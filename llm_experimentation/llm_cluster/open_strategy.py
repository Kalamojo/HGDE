"""Strategy "open": the LLM is told only the goal and designs its own way of representing items.

    round 0   the LLM sees 40 sample items and writes a METHOD: plain-language instructions that turn ONE
              item into 1-32 numbers (what each number means is up to it), optionally naming up to 12
              reference items to compare against
    embed     separate executor calls apply the method to every item; the executor sees only the method,
              the reference items and its batch (never the hints). The numbers go straight to the run's
              clusterer (HDBSCAN or k-means), with no z-scoring: the LLM controls the scale.
    update    up to --drafts steps: the LLM may TEST a draft method (executed on every hinted item and
              scored against all hints) before committing one

It mirrors Jay's triplet model: a function applied to each item alone (the method), capacity to bend
around specific items (reference items), and several passes over all triplets per round (test drafts).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from pydantic import BaseModel, Field
from sklearn.preprocessing import normalize

from .data import Item, render_triplets
from .harness import RoundInput
from .llm import MAX_VIOLATED_SHOWN, JsonlCache, LLMStrategy, Text, hint_lines, score_against_hints

MAX_REFERENCE_ITEMS = 12


# ---------- state and reply formats ----------

class Method(BaseModel):
    """The strategy's whole state (saved as schema_round_XX.json)."""
    approach: str = Field(min_length=3, max_length=4000)       # in plain words: what it is doing and why
    instructions: str = Field(min_length=3, max_length=12000)  # what the executor does for one item
    outputs: list[str] = Field(min_length=1, max_length=32)    # meaning of each number, in order
    reference_items: list[str] = Field(default=[], max_length=MAX_REFERENCE_ITEMS)
    notes: str = ""

    @property
    def key(self) -> str:
        """Cache key for its vectors: only what the executor sees (not the approach or notes)."""
        return hashlib.sha1(json.dumps([self.instructions.strip(), self.outputs, self.reference_items]).encode()).hexdigest()[:12]


class ReviseReply(BaseModel):
    action: str = Field(pattern=r"^(test|commit|commit_draft)$")
    method: Method | None = None        # for test / commit
    draft: int | None = None            # for commit_draft


class ExecuteRow(BaseModel):
    item: str
    vector: list[float]


class ExecuteReply(BaseModel):
    rows: list[ExecuteRow]


# ---------- prompts ----------
# Nothing here suggests HOW to represent the items: the LLM is told the goal, the mechanics its output
# goes through, and the feedback, and chooses the rest itself.

GOAL = (
    "A user has an ideal way of grouping this collection in mind, which they have not told you. Your job is to "
    "change how the items are represented, so that an unsupervised clustering algorithm groups them the way "
    "the user wants. After each grouping, the user gives hints about which items should be closer together "
    "and which should be further apart. Generalize from these hints to identify the hidden pattern the user "
    "is grouping by: the grouping has to be right for every item in the collection, including the many items "
    "the hints never mention. How you do this is up to you."
)

MECHANICS = (
    "How your representation is used:\n"
    "- You write a METHOD that turns any single item into a list of numbers.\n"
    "- An independent executor applies your method to every item, one item at a time. For each item it sees "
    "only your method, that item, and the reference items you name (if any). It never sees the user's hints "
    "or any other item.\n"
    "- You decide what the numbers are, how many there are (1 to 32), and how the executor computes them. "
    f"You may name up to {MAX_REFERENCE_ITEMS} reference items (by id); the executor is shown them with every item.\n"
    "- Each item's list of numbers is then scaled to length 1, and k-means groups the items by Euclidean "
    "distance. The number of groups is fixed and is not told to you.\n"
    "- A hint \"A should be closer to B than to C\" counts as satisfied when, after the scaling, A's distance "
    "to C is larger than its distance to B by at least 0.2.\n\n"
    "Write a method as JSON:\n"
    '{"approach": "<in plain words: what you are doing and why>", '
    '"instructions": "<exactly what the executor should do for one item>", '
    '"outputs": ["<what number 1 is>", "<what number 2 is>", ...], '
    '"reference_items": ["item_xxx", ...], '
    '"notes": "<anything you want to remember for later rounds>"}\n'
)


def design_prompt(s: OpenStrategy, sample: list[Item]) -> list:
    parts = [Text(f"{GOAL}\n\n{MECHANICS}\nBelow are {len(sample)} items sampled from the collection of {s.n}. "
                  "There are no hints yet: they start after the first grouping. Write your first method; reply "
                  "with only the method JSON.\n\nThe items:")]
    for item in sample:
        parts += s.item_parts(item)
    return parts


def revise_prompt(s: OpenStrategy, m: Method, inp: RoundInput, involved: list[int], current: np.ndarray,
                  sat: dict, past: list[str], drafts_text: list[str], step: int) -> list:
    if step == s.cfg.drafts:
        budget = (f"This is your last step this round (step {step} of {s.cfg.drafts}): whatever you reply is "
                  "applied. Reply {\"action\": \"commit\", \"method\": {...}} or {\"action\": \"commit_draft\", "
                  "\"draft\": <number>} to apply one of the drafts you tested.")
    else:
        budget = (f"You have {s.cfg.drafts} steps this round; this is step {step}. You can TEST a draft method "
                  "first: it is executed on every item mentioned in any hint so far, and you are told how many hints "
                  "it satisfies and which it violates. Reply {\"action\": \"test\", \"method\": {...}} to test, "
                  "{\"action\": \"commit\", \"method\": {...}} to apply a method now, or {\"action\": "
                  "\"commit_draft\", \"draft\": <number>} to apply a draft you already tested. On the last step "
                  "your reply is applied.")
    parts = [Text(
        f"{GOAL}\n\n{MECHANICS}\n"
        f"Your current method (applied to every item):\n{m.model_dump_json(indent=1)}\n\n"
        "What you did in earlier rounds:\n" + ("\n".join(past) or "(this is the first round of hints)") + "\n\n"
        "Hints from earlier rounds:\n" + ("\n".join(render_triplets(inp.earlier_triplets, s.items)) or "(none)") + "\n\n"
        f"Hints this round (round {inp.round}):\n" + "\n".join(render_triplets(inp.new_triplets, s.items)) + "\n\n"
        f"With your current method, {sat['sat_new']:.0%} of this round's hints and {sat['sat_all']:.0%} of all hints "
        "so far are satisfied.\n\nThe items mentioned this round, with their current numbers:"
    )]
    for i in involved:
        parts += s.item_parts(s.items[i], extra="current numbers: [" + ", ".join(f"{v:.3g}" for v in current[i]) + "]")
    tail = "\n\n"
    if drafts_text:
        tail += "Drafts you tested this round:\n" + "\n\n".join(drafts_text) + "\n\n"
    parts.append(Text(tail + budget + "\nReply with only the JSON."))
    return parts


def executor_prompt(s: OpenStrategy, m: Method, batch: list[Item]) -> list:
    outputs = "\n".join(f"{j + 1}. {o}" for j, o in enumerate(m.outputs))
    parts = [Text("Apply the following method to each item below, one item at a time.\n\n"
                  f"Method:\n{m.instructions}\n\n"
                  f"For each item, output exactly {len(m.outputs)} numbers, in this order:\n{outputs}\n")]
    if m.reference_items:
        parts.append(Text("\nReference items (use them only as the method says):"))
        for r in m.reference_items:
            parts += s.item_parts(s.items[s.by_anon[r]])
    parts.append(Text("\n\nReply with JSON of the form:\n"
                      '{"rows": [{"item": "<item id>", "vector": [<number>, ...]}, ...]}\n'
                      f"with one row for each of the {len(batch)} items below and exactly {len(m.outputs)} numbers "
                      "per row.\n\nThe items to process:"))
    for item in batch:
        parts += s.item_parts(item)
    return parts


# ---------- the strategy ----------

class OpenStrategy(LLMStrategy):
    state_cls = Method
    name = "open"

    def __init__(self, sel, client, cfg, out_dir, cache_dir):
        super().__init__(sel, client, cfg, out_dir, cache_dir)
        self.vectors = JsonlCache(out_dir / "vectors.jsonl", "vector")
        self.last_changes = {}              # for this round's history.csv row

    def check_method(self, m: Method) -> str | None:
        """Reference items must be real, distinct item ids."""
        unknown = [r for r in m.reference_items if r not in self.by_anon]
        if unknown:
            return f"unknown reference item ids {unknown[:10]}"
        if len(set(m.reference_items)) != len(m.reference_items):
            return "reference items must be distinct"
        return None

    def method_matrix(self, m: Method, idx=None) -> np.ndarray:
        """The method's vectors for items `idx` (default all), executing it where not cached. Two passes
        over whatever is missing; a vector still missing becomes zeros (counted in `filled`). Rows
        outside `idx` stay 0 (only used when testing drafts)."""
        idx = range(self.n) if idx is None else sorted(set(idx))
        for _ in range(2):
            todo = self.vectors.missing(m.key, idx)
            if todo:
                self.run_batches(todo, lambda batch: self.execute_batch(m, batch))
        missing = self.vectors.missing(m.key, idx)
        self.filled += len(missing) * len(m.outputs)
        self.vectors.put_many([(m.key, i, [0.0] * len(m.outputs)) for i in missing])
        M = np.zeros((self.n, len(m.outputs)))
        for i in idx:
            M[i] = self.vectors.get(m.key, i)
        return M

    def execute_batch(self, m: Method, batch: list[int]):
        """One executor call; keeps every row with the right number of finite numbers."""
        items = [self.items[i] for i in batch]
        reply = self.ask(executor_prompt(self, m, items), ExecuteReply, "open_execute",
                         {"item_ids": [it.anon_id for it in items], "d": len(m.outputs)})
        if reply is None:
            return
        wanted = {it.anon_id: it.idx for it in items}
        self.vectors.put_many([(m.key, wanted[r.item], r.vector) for r in reply.rows if r.item in wanted
                               and len(r.vector) == len(m.outputs) and np.all(np.isfinite(r.vector))])

    def start(self) -> Method:
        sample = self.design_sample()
        reply = self.ask(design_prompt(self, sample), Method, "open_design",
                         {"ref_pool": [it.anon_id for it in sample]}, self.check_method)
        if reply is None:
            raise RuntimeError("open method design failed twice; see llm_calls.jsonl")
        return reply

    def embed(self, m: Method) -> np.ndarray:
        return self.method_matrix(m)

    def update(self, m: Method, inp: RoundInput):
        new = inp.new_triplets
        all_triplets = inp.earlier_triplets + new
        involved = list(dict.fromkeys(int(i) for t in new for i in t))       # this round's hinted items
        current = self.method_matrix(m, involved)
        sat = score_against_hints(inp.E, all_triplets, new)
        past = self.past_rounds(inp, lambda ch: f"{'changed' if ch.get('method_changed') else 'kept'} the method "
                                                f"({ch.get('drafts_tested', 0)} drafts tested)")

        def ask_step(step, drafts):
            def check(reply):
                if reply.action != "commit_draft" and reply.method is None:
                    return "a method is required for test / commit"
                return self.step_error(reply, len(drafts)) or (reply.method and self.check_method(reply.method))

            prompt = revise_prompt(self, m, inp, involved, current, sat, past, [self.draft_text(d) for d in drafts], step)
            reply = self.ask(prompt, ReviseReply, "open_revise",
                             {"round": inp.round, "n_tested": len(drafts),
                              "ref_pool": [self.items[i].anon_id for i in involved]}, check)
            if reply is None:
                return None
            return reply.action, reply.draft if reply.action == "commit_draft" else reply.method

        def test(method):
            """Score a draft against all hints, executing it only on hinted items (no z-scoring, so an
            item's vector doesn't depend on other items)."""
            hinted = sorted({int(i) for t in all_triplets for i in t})
            return score_against_hints(normalize(self.method_matrix(method, hinted)), all_triplets, new)

        chosen, how, drafts = self.draft_loop(ask_step, test)
        chosen = chosen or m
        self.last_changes = {
            "approach": chosen.approach, "committed": how, "method_changed": chosen.key != m.key,
            "drafts_tested": len(drafts), "satisfied_before": {"all": sat["sat_all"], "new": sat["sat_new"]},
            "drafts": [{"draft": d["draft"], "sat_all": d["sat_all"], "sat_new": d["sat_new"],
                        "method": d["version"].model_dump()} for d in drafts],
        }
        return chosen, self.last_changes

    def draft_text(self, d: dict) -> str:
        """How a tested draft and its result are shown to the LLM at later steps."""
        shown = hint_lines(d["violated"][:MAX_VIOLATED_SHOWN], self.items)
        return (f"Draft {d['draft']}:\n{d['version'].model_dump_json(indent=1)}\nResult: satisfies {d['sat_new']:.0%} of "
                f"this round's hints and {d['sat_all']:.0%} of all hints so far. Distinct hints still violated "
                f"({len(d['violated'])}):\n" + ("\n".join(shown) or "  (none)"))

    def log_columns(self, m: Method) -> dict:
        return {"n_features": len(m.outputs), "n_reference_items": len(m.reference_items),
                "drafts_tested": self.last_changes.get("drafts_tested", 0),
                "method_changed": self.last_changes.get("method_changed", False), **self.usage_columns()}

    def summary(self, m: Method, changes: dict) -> str:
        return f"{changes.get('drafts_tested', 0)} drafts tested, committed {changes.get('committed')}: {m.approach[:70]}, {self.spent()}"
