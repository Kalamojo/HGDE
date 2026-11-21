"""Strategy "direct": the LLM does the whole clustering itself (no representation, no k-means).

    every call  the LLM sees ALL items: as labelled contact sheets (6 x 6 tiles, each item's id under its
                picture), grouped by its current clustering, and/or as descriptions
    round 0     it puts every item into exactly k groups (k = 3, the number Jay's k-means is given)
    update      it sees all hints so far and which ones its clustering violates; up to --drafts steps, in
                which it may TEST a new clustering against all hints before committing one

For the harness the clustering is turned into one-hot vectors (group c -> unit vector e_c), so all metrics
and Jay's feedback work unchanged. On one-hot vectors a hint is satisfied exactly when A and P share a
group and N doesn't (distances are 0 or sqrt 2, against the margin 0.2).

Fairness: hinted items can be placed exactly, so ARI on all items includes memorization. Compare direct
runs by test ARI.
"""

from __future__ import annotations

import io
import math

import numpy as np
from pydantic import BaseModel, Field

from .data import Item, load_png, render_triplets
from .harness import RoundInput
from .llm import MAX_VIOLATED_SHOWN, Image, LLMStrategy, Text, hint_lines, score_against_hints

SHEET_COLS = SHEET_ROWS = 6       # tiles per contact sheet: 36 items per image
TILE_PX = 128


# ---------- state and reply formats ----------

class Grouping(BaseModel):
    """The strategy's whole state (saved as schema_round_XX.json); also the round-0 reply format."""
    approach: str = ""
    groups: dict[str, list[str]]        # group name -> anonymous item ids
    notes: str = ""


class ReviseReply(BaseModel):
    action: str = Field(pattern=r"^(test|commit|commit_draft)$")
    approach: str = ""
    groups: dict[str, list[str]] | None = None      # for test / commit
    notes: str = ""
    draft: int | None = None                        # for commit_draft


# ---------- prompts ----------

GOAL = (
    "A user has an ideal way of clustering this collection in mind, which they have not told you. Your job is to "
    "cluster the items so that your clustering matches the one the user wants. After each clustering, the user "
    "gives hints about which items should be closer together and which should be further apart. Generalize "
    "these hints to identify the hidden pattern the user is clustering by, and zero in on it: the clustering has "
    "to be right for every item in the collection, including the many items the hints never mention. How you do "
    "this is up to you."
)


def rules(k: int) -> str:
    return (f"Rules: put every item into exactly one of exactly {k} groups. A hint \"A should be closer to B than "
            "to C\" is satisfied when A and B are in the same group and C is in a different group.\n\n"
            "A clustering is written as JSON:\n"
            '{"approach": "<in plain words: how you are clustering and why>", '
            '"groups": {"<group name>": ["item_xxx", ...], ...}, '
            '"notes": "<anything you want to remember for later rounds>"}\n'
            f"with exactly {k} groups and every item id exactly once.\n")


def revise_prompt(s: DirectStrategy, g: Grouping, inp: RoundInput, current: dict, past: list[str],
                  items_shown: list, drafts_text: list[str], step: int) -> list:
    violated = hint_lines(current["violated"][:MAX_VIOLATED_SHOWN], s.items)
    if len(current["violated"]) > len(violated):
        violated.append(f"... and {len(current['violated']) - len(violated)} more")
    if step == s.cfg.drafts:
        budget = (f"This is your last step this round (step {step} of {s.cfg.drafts}): whatever you reply is applied. "
                  "Reply {\"action\": \"commit\", <clustering>} or {\"action\": \"commit_draft\", \"draft\": <number>} "
                  "to apply a clustering you tested.")
    else:
        budget = (f"You have {s.cfg.drafts} steps this round; this is step {step}. You can TEST a clustering first: you "
                  "are told how many hints it satisfies and which it violates. Reply {\"action\": \"test\", <clustering>} "
                  "to test, {\"action\": \"commit\", <clustering>} to apply one now, or {\"action\": \"commit_draft\", "
                  "\"draft\": <number>} to apply a clustering you already tested. On the last step your reply is "
                  "applied. <clustering> means the approach, groups and notes fields.")
    parts = [Text(
        f"{GOAL}\n\n{rules(s.k)}\n"
        f"Your notes so far: {g.notes or '(none)'}\n\n"
        "What you did in earlier rounds:\n" + ("\n".join(past) or "(this is the first round of hints)") + "\n\n"
        "Hints from earlier rounds:\n" + ("\n".join(render_triplets(inp.earlier_triplets, s.items)) or "(none)") + "\n\n"
        f"Hints this round (round {inp.round}):\n" + "\n".join(render_triplets(inp.new_triplets, s.items)) + "\n\n"
        f"Your current clustering satisfies {current['sat_new']:.0%} of this round's hints and {current['sat_all']:.0%} "
        f"of all hints so far. Distinct hints it violates ({len(current['violated'])}, this round's first):\n"
        + ("\n".join(violated) or "(none)") + "\n\nAll items, shown by your current clustering:"
    )] + items_shown
    tail = "\n\n"
    if drafts_text:
        tail += "Clusterings you tested this round:\n" + "\n\n".join(drafts_text) + "\n\n"
    parts.append(Text(tail + budget + "\nReply with only the JSON."))
    return parts


def validate_grouping(groups: dict[str, list[str]], all_ids: list[str], k: int) -> str | None:
    """What is wrong with a clustering (None = valid): exactly k non-empty groups, every id exactly once."""
    listed = [a for ids in groups.values() for a in ids]
    problems = []
    if sum(1 for ids in groups.values() if ids) != k:
        problems.append(f"there must be exactly {k} non-empty groups")
    if sorted(listed) != sorted(all_ids):
        dup = sorted({a for a in listed if listed.count(a) > 1})
        unknown = sorted(set(listed) - set(all_ids))
        missing = sorted(set(all_ids) - set(listed))
        problems.append(f"every item exactly once: duplicated {dup[:20]}, unknown {unknown[:20]}, "
                        f"missing {len(missing)} (e.g. {missing[:20]})")
    return "; ".join(problems) or None


def contact_sheet(items: list[Item], cache_dir) -> bytes:
    """One PNG with the items tiled SHEET_COLS wide, each item's id printed under its tile."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    label_h = TILE_PX // 7
    cols, rows = min(SHEET_COLS, len(items)), math.ceil(len(items) / SHEET_COLS)
    sheet = PILImage.new("RGB", (cols * TILE_PX, rows * (TILE_PX + label_h)), "white")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=label_h - 3)
    for j, it in enumerate(items):
        x, y = (j % SHEET_COLS) * TILE_PX, (j // SHEET_COLS) * (TILE_PX + label_h)    # top-left of tile j
        with PILImage.open(io.BytesIO(load_png(it.image_path, TILE_PX, cache_dir))) as im:
            sheet.paste(im.convert("RGB"), (x + (TILE_PX - im.width) // 2, y + (TILE_PX - im.height) // 2))
        draw.text((x + TILE_PX // 2, y + TILE_PX + label_h // 2), it.anon_id, fill="black", font=font, anchor="mm")
    buf = io.BytesIO()
    sheet.quantize(colors=256).save(buf, format="PNG")      # ~3.5x smaller, so 13 sheets fit in one request
    return buf.getvalue()


# ---------- the strategy ----------

class DirectStrategy(LLMStrategy):
    state_cls = Grouping
    name = "direct"

    def __init__(self, sel, client, cfg, out_dir, cache_dir):
        super().__init__(sel, client, cfg, out_dir, cache_dir)
        self.ids = [it.anon_id for it in self.items]
        self.last_changes = {}

    def all_items_parts(self, blocks: list[tuple[str, list[Item]]]) -> list:
        """(title, items) blocks: a header listing the block's ids, then its items as contact sheets
        (36 per image) and/or "id: description" lines."""
        parts = []
        for title, members in blocks:
            members = sorted(members, key=lambda it: it.anon_id)
            parts.append(Text(f"\n\n=== {title} ({len(members)} items): {', '.join(it.anon_id for it in members)}\n"))
            if self.cfg.inputs != "text":
                per = SHEET_COLS * SHEET_ROWS
                for s in range(0, len(members), per):
                    parts.append(Image(contact_sheet(members[s:s + per], self.cache_dir), label=f"sheet {title} {s // per + 1}"))
            if self.cfg.inputs != "image":
                parts.append(Text("\n" + "\n".join(f"{it.anon_id}: {it.text}" for it in members) + "\n"))
        return parts

    def check_groups(self, groups) -> str | None:
        return validate_grouping(groups or {}, self.ids, self.k)

    def one_hot(self, g: Grouping) -> np.ndarray:
        """Item i in the c-th non-empty group -> row i is the unit vector e_c."""
        V = np.zeros((self.n, self.k))
        for c, ids in enumerate(ids for ids in g.groups.values() if ids):
            V[[self.by_anon[a] for a in ids], c] = 1.0
        return V

    def start(self) -> Grouping:
        parts = [Text(f"{GOAL}\n\n{rules(self.k)}\nBelow are all {self.n} items. There are no hints yet: they start "
                      "after your first clustering. Reply with only the clustering JSON.\n\nThe items:")]
        parts += self.all_items_parts([("all items", list(self.items))])
        reply = self.ask(parts, Grouping, "direct_design", {"item_ids": self.ids, "k": self.k},
                         lambda r: self.check_groups(r.groups))
        if reply is None:
            raise RuntimeError("direct clustering failed twice; see llm_calls.jsonl")
        return reply

    def embed(self, g: Grouping) -> np.ndarray:
        return self.one_hot(g)

    def cluster(self, V: np.ndarray) -> np.ndarray:
        """No k-means: the clusters are the LLM's groups, read back from the one-hot vectors."""
        return V.argmax(axis=1)

    def update(self, g: Grouping, inp: RoundInput):
        new = inp.new_triplets
        all_triplets = inp.earlier_triplets + new
        current = score_against_hints(self.one_hot(g), all_triplets, new)
        past = self.past_rounds(inp, lambda ch: f"{ch.get('drafts_tested', 0)} clusterings tested")
        # every item, shown by the current grouping (built once: the contact sheets are slow to make)
        items_shown = self.all_items_parts([(f"group {name}", [self.items[self.by_anon[a]] for a in ids])
                                            for name, ids in g.groups.items()])

        def ask_step(step, drafts):
            def check(reply):
                if reply.action == "commit_draft":
                    return self.step_error(reply, len(drafts))
                return self.check_groups(reply.groups)

            prompt = revise_prompt(self, g, inp, current, past, items_shown, [self.draft_text(d) for d in drafts], step)
            reply = self.ask(prompt, ReviseReply, "direct_revise",
                             {"round": inp.round, "n_tested": len(drafts), "item_ids": self.ids, "k": self.k}, check)
            if reply is None:                   # still invalid after the retry: the current grouping stays
                return None
            if reply.action == "commit_draft":
                return reply.action, reply.draft
            return reply.action, Grouping(approach=reply.approach, groups=reply.groups, notes=reply.notes or g.notes)

        chosen, how, drafts = self.draft_loop(ask_step, lambda c: score_against_hints(self.one_hot(c), all_triplets, new))
        chosen = chosen or g
        self.last_changes = {
            "approach": chosen.approach, "committed": how, "drafts_tested": len(drafts),
            "group_sizes": [len(ids) for ids in chosen.groups.values()],
            "satisfied_before": {"all": current["sat_all"], "new": current["sat_new"]},
            "drafts": [{"draft": d["draft"], "sat_all": d["sat_all"], "sat_new": d["sat_new"],
                        "approach": d["version"].approach} for d in drafts],
        }
        return chosen, self.last_changes

    def draft_text(self, d: dict) -> str:
        groups = "\n".join(f"  {name} ({len(ids)}): {', '.join(ids)}" for name, ids in d["version"].groups.items())
        shown = hint_lines(d["violated"][:MAX_VIOLATED_SHOWN], self.items)
        return (f"Draft {d['draft']}: approach: {d['version'].approach}\n{groups}\nResult: satisfies {d['sat_new']:.0%} "
                f"of this round's hints and {d['sat_all']:.0%} of all hints so far. Distinct hints still violated "
                f"({len(d['violated'])}):\n" + ("\n".join(shown) or "  (none)"))

    def log_columns(self, g: Grouping) -> dict:
        return {"group_sizes": "/".join(str(len(ids)) for ids in g.groups.values()),
                "drafts_tested": self.last_changes.get("drafts_tested", 0), **self.usage_columns()}

    def summary(self, g: Grouping, changes: dict) -> str:
        return (f"{changes.get('drafts_tested', 0)} tested, committed {changes.get('committed')}, groups "
                f"{'/'.join(str(len(ids)) for ids in g.groups.values())}: {g.approach[:60]}, {self.spent()}")
