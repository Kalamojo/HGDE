"""The experiment loop: design schema -> score -> k-means -> Jay's feedback -> revise schema -> ...

Mirrors tests/triplet_by_column.py round for round: same items, same k-means call, same feedback
function and rng stream, same metrics (src/utils/metrics.all_metrics) and the same history columns,
so the curves overlay results/triplet_by_column/<run>/history.csv directly.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ValidationError
from sklearn.preprocessing import normalize

from . import prompts
from .client import BudgetExceeded, QuotaExhausted, Text, extract_json
from .feedback import next_feedback, render_triplets
from .items import REPO_ROOT, Selection, load_jay_module
from .schema import MAX_WEIGHT, Feature, Schema, ScoreStore, raw_matrix, score_key, vectorize

from archive.clustering import cluster_embeddings  # noqa: E402  (repo root is on sys.path via .items)
from archive.scripts.triplet_learning_testing import triplet_stats, write_csv  # noqa: E402
from src.utils.metrics import all_metrics  # noqa: E402


@dataclass
class Config:
    inputs: str
    strategy: str = "schema"       # "schema" (A: feature schema) | "hypothesis" (B: hypothesis + rescore)
    rounds: int = 30
    triplets_per_round: int = 32
    goal_ari: float = 0.95
    seed: int = 42
    design_sample: int = 40
    score_batch: int = 20
    workers: int = 4
    image_px: int = 256
    max_tokens: int = 32000
    resume: bool = False


@dataclass
class Counters:
    parse_failures: int = 0
    filled_scores: int = 0
    rejected_changes: list = field(default_factory=list)


class Experiment:
    def __init__(self, sel: Selection, client, cfg: Config, out_dir: Path, cache_dir: Path):
        self.sel, self.client, self.cfg, self.out = sel, client, cfg, out_dir
        self.items = sel.items
        self.n = len(self.items)
        self.k = sel.n_classes
        self.by_anon = {it.anon_id: it.idx for it in self.items}
        self.renderer = prompts.ItemRenderer(cfg.inputs, cfg.image_px, cache_dir)
        self.system = prompts.system_prompt(cfg.inputs)
        self.store = ScoreStore(out_dir / "scores.jsonl")
        self.counters = Counters()
        self.history: list[dict] = []
        self.edits: list[dict] = []
        self.all_triplets: list = []

    # ---------- LLM calls ----------

    def call_json(self, parts: list, reply_model: type[BaseModel], task: str, meta: dict, check=None):
        """One call, validated against reply_model (+ optional check(reply) -> error str|None); one retry
        with the error appended. Returns None if both attempts fail."""
        attempt_parts = parts
        for attempt in range(2):
            text = self.client.complete(self.system, attempt_parts, self.cfg.max_tokens, task=task, meta=meta)
            try:
                reply = reply_model.model_validate(extract_json(text))
                problem = check(reply) if check else None
                if problem is None:
                    return reply
                error = problem
            except (ValueError, ValidationError) as e:
                error = str(e)[:1500]
            self.counters.parse_failures += 1
            attempt_parts = parts + [Text(f"\n\nYour previous reply was not valid: {error}\n"
                                          "Reply again with only the corrected JSON object.")]
        return None

    def design_sample(self):
        rng = np.random.default_rng([self.cfg.seed, 2])
        sample_idx = sorted(rng.choice(self.n, min(self.cfg.design_sample, self.n), replace=False))
        return [self.items[i] for i in sample_idx]

    def design(self) -> Schema:
        if self.cfg.strategy == "hypothesis":
            return self.design_hypothesis()
        parts = prompts.design_prompt(self.design_sample(), self.n, self.renderer)

        def check(reply):
            try:
                Schema(features=reply.features)
            except ValidationError as e:
                return str(e)
            return None

        reply = self.call_json(parts, prompts.DesignReply, "design", {}, check)
        if reply is None:
            raise RuntimeError("schema design failed twice; see llm_calls.jsonl")
        return Schema(features=[f.model_copy(update={"weight": 1.0}) for f in reply.features], notes=reply.notes)

    @staticmethod
    def check_hypothesis(reply) -> str | None:
        try:
            Schema(features=reply.dimensions)
        except ValidationError as e:
            return str(e)
        if sum(f.weight > 0 for f in reply.dimensions) < 2:
            return "at least 2 dimensions need a weight above 0"
        return None

    def design_hypothesis(self) -> Schema:
        parts = prompts.hypothesis_design_prompt(self.design_sample(), self.n, self.renderer)
        reply = self.call_json(parts, prompts.HypothesisReply, "hyp_design", {}, self.check_hypothesis)
        if reply is None:
            raise RuntimeError("hypothesis design failed twice; see llm_calls.jsonl")
        return Schema(features=reply.dimensions, notes=reply.notes, criterion=reply.criterion.strip())

    def score(self, features: list[Feature], criterion: str = ""):
        """Score every item on every given feature that isn't cached yet (under `criterion` for B)."""
        todo = sorted({i for f in features for i in self.store.missing(f, self.n, criterion)})
        if not todo:
            return
        feats = [f for f in features if self.store.missing(f, self.n, criterion)]
        batches = [todo[s:s + self.cfg.score_batch] for s in range(0, len(todo), self.cfg.score_batch)]

        def run(batch):
            batch_items = [self.items[i] for i in batch]
            ids = [it.anon_id for it in batch_items]
            fids = [f.id for f in feats]

            def check(reply):
                got = {r.item: r.scores for r in reply.rows}
                missing_items = [a for a in ids if a not in got]
                missing_scores = [(a, f) for a in ids if a in got for f in fids if f not in got[a]]
                if missing_items or missing_scores:
                    return (f"missing rows for {missing_items[:10]} and scores for {missing_scores[:10]}; "
                            "every item needs a score for every feature")
                return None

            reply = self.call_json(prompts.score_prompt(feats, batch_items, self.renderer, criterion), prompts.ScoreReply,
                                   "score", {"item_ids": ids, "feature_ids": fids}, check)
            got = {r.item: r.scores for r in reply.rows} if reply else {}
            rows = []
            for it in batch_items:
                for f in feats:
                    value = got.get(it.anon_id, {}).get(f.id)
                    if value is None:          # still missing after the retry: neutral value, counted
                        value = 0.5
                        self.counters.filled_scores += 1
                    rows.append((score_key(f, criterion), it.idx, float(np.clip(value, 0.0, 1.0))))
            self.store.put_many(rows)

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            list(pool.map(run, batches))

    def revise(self, schema: Schema, round_number: int, new_triplets, earlier_triplets, E, pred):
        involved_idx = list(dict.fromkeys([t[0] for t in new_triplets] + [t[1] for t in new_triplets]
                                          + [t[2] for t in new_triplets]))
        involved = [self.items[i] for i in involved_idx]
        M = raw_matrix(schema, self.store, self.n)
        values = {i: ", ".join(f"{f.id}={M[i, j]:.2f}" for j, f in enumerate(schema.features)) for i in involved_idx}
        sat_new = triplet_stats(E, new_triplets, pred)["distance_triplet_satisfaction"]
        sat_all = triplet_stats(E, earlier_triplets + new_triplets, pred)["distance_triplet_satisfaction"]
        if self.cfg.strategy == "hypothesis":
            parts = prompts.hypothesis_revise_prompt(
                schema, round_number, render_triplets(new_triplets, self.items),
                render_triplets(earlier_triplets, self.items), involved, values, sat_all, sat_new, self.renderer)
            reply = self.call_json(parts, prompts.HypothesisReply, "hyp_revise",
                                   {"feature_ids": [f.id for f in schema.features], "round": round_number},
                                   self.check_hypothesis)
            if reply is None:
                self.counters.rejected_changes.append(f"round {round_number}: revision failed twice")
                return schema, {"error": "revision failed twice; hypothesis unchanged"}
            new = Schema(features=[f.model_copy(update={"weight": float(np.clip(f.weight, 0.0, MAX_WEIGHT))})
                                   for f in reply.dimensions],
                         notes=reply.notes or schema.notes, criterion=reply.criterion.strip())
            return new, {"criterion": new.criterion, "criterion_changed": new.criterion != schema.criterion,
                         "dimensions": [f"{f.id} (w={f.weight:g}): {f.definition}" for f in new.features]}
        parts = prompts.revise_prompt(schema, round_number, render_triplets(new_triplets, self.items),
                                      render_triplets(earlier_triplets, self.items), involved, values,
                                      sat_all, sat_new, self.renderer)
        reply = self.call_json(parts, prompts.ReviseReply, "revise",
                               {"feature_ids": [f.id for f in schema.features], "round": round_number})
        if reply is None:
            return schema, {"error": "revision failed twice; schema unchanged"}
        return self.apply_revision(schema, reply)

    def apply_revision(self, schema: Schema, reply: prompts.ReviseReply):
        feats = {f.id: f for f in schema.features}
        rejected = []
        for fid, w in reply.weights.items():
            if fid in feats:
                feats[fid] = feats[fid].model_copy(update={"weight": float(np.clip(w, 0.0, MAX_WEIGHT))})
            else:
                rejected.append(f"weight for unknown feature {fid}")
        for fid in reply.drop:
            if fid in feats:
                del feats[fid]
            else:
                rejected.append(f"drop of unknown feature {fid}")
        added = []
        for f in reply.add:
            if f.id in feats:
                rejected.append(f"add with existing id {f.id}")
                continue
            feats[f.id] = f
            added.append(f.id)
        if sum(f.weight > 0 for f in feats.values()) < 2:
            rejected.append("revision left fewer than 2 active features; schema unchanged")
            self.counters.rejected_changes += rejected
            return schema, {"rejected": rejected}
        self.counters.rejected_changes += rejected
        new = Schema(features=list(feats.values()), notes=reply.notes or schema.notes)
        return new, {"weights": reply.weights, "drop": reply.drop, "add": added, "rejected": rejected}

    # ---------- evaluation ----------

    def embed_and_cluster(self, schema: Schema):
        self.score(schema.features, schema.criterion)
        V = vectorize(schema, self.store, self.n)
        pred, _ = cluster_embeddings(V, algorithm="kmeans", n_clusters=self.k, random_state=self.cfg.seed)
        E = normalize(V, norm="l2")   # the space run_kmeans actually clusters in (it L2-normalizes)
        return V, E, np.asarray(pred)

    def metrics(self, E, pred) -> dict:
        s = self.sel
        return all_metrics(E, pred, s.y, s.test_mask if s.test_mask.any() else None, s.auc_pairs)

    def usage_cols(self, schema: Schema) -> dict:
        log = self.client.log
        return {"n_features": len(schema.features), "n_active_features": sum(f.weight > 0 for f in schema.features),
                "llm_parse_failures": self.counters.parse_failures, "filled_scores": self.counters.filled_scores,
                "llm_calls": log.calls, "input_tokens": log.input_tokens, "output_tokens": log.output_tokens,
                "usd_total": round(log.usd, 4)}

    # ---------- main loop ----------

    def run(self):
        cfg, sel = self.cfg, self.sel
        print(f"Items={self.n}, classes={self.k}, held-out test items={int(sel.test_mask.sum())}, "
              f"inputs={cfg.inputs}, strategy={cfg.strategy}")
        stop_reason = "completed all rounds"
        all_triplets = self.all_triplets
        used_anchors: set = set()
        if cfg.resume:
            schema, V, E, pred, baseline, first_round = self.restore(used_anchors)
        else:
            try:
                schema = self.design()
                V, E, pred = self.embed_and_cluster(schema)
            except (BudgetExceeded, QuotaExhausted) as e:
                print(f"Stopped before round 0: {e}")
                return
            baseline = pred.copy()
            first_round = 1
            self.history.append({"round": 0, **self.metrics(E, pred), "triplets_total": 0, "new_triplets_before": "",
                                 "new_triplets_after": "", "all_triplets_satisfied": "", "criterion_changed": "",
                                 **self.usage_cols(schema)})
            self.save(schema, V, 0, baseline, pred)
            self.report(self.history[-1], f"criterion: {schema.criterion[:70]}" if schema.criterion else "")

        for r in range(first_round, cfg.rounds + 1):
            if float(self.history[-1]["ari"]) >= cfg.goal_ari:
                stop_reason = "reached goal ARI"
                break
            fb = next_feedback(pred, sel.y, sel.rng, used_anchors, cfg.triplets_per_round, sel.test_mask)
            if fb is None:
                stop_reason = "no remaining label/cluster disagreement"
                break
            anchor, kind, new_triplets = fb
            used_anchors.add(anchor)
            before = triplet_stats(E, new_triplets, pred)
            try:
                schema, changes = self.revise(schema, r, new_triplets, all_triplets, E, pred)
                all_triplets.extend(new_triplets)
                V, E, pred = self.embed_and_cluster(schema)
            except BudgetExceeded as e:
                stop_reason = str(e)
                break
            except QuotaExhausted as e:
                stop_reason = (f"daily token quota used up ({e}). Round {r} was not saved; continue later with "
                               "the same command plus --resume")
                break
            after = triplet_stats(E, new_triplets, pred)
            overall = triplet_stats(E, all_triplets, pred)
            self.history.append({"round": r, **self.metrics(E, pred), "triplets_total": len(all_triplets),
                                 "new_triplets_before": before["distance_triplet_satisfaction"],
                                 "new_triplets_after": after["distance_triplet_satisfaction"],
                                 "all_triplets_satisfied": overall["distance_triplet_satisfaction"],
                                 "criterion_changed": bool(changes.get("criterion_changed", False)),
                                 **self.usage_cols(schema)})
            it = self.items[anchor]
            self.edits.append({"round": r, "anchor_index": int(anchor), "anon_id": it.anon_id, "name": it.name,
                               "target_label": it.label, "feedback_type": kind,
                               "triplets_added": len(new_triplets), "changes": changes, "criterion": schema.criterion,
                               "notes": schema.notes})
            self.save(schema, V, r, baseline, pred)
            if cfg.strategy == "hypothesis":
                what = f"criterion: {schema.criterion[:70]}" if changes.get("criterion_changed") else "criterion unchanged"
            else:
                what = f"+{len(changes.get('add', []))} features"
            self.report(self.history[-1], f"{it.name} ({kind}), {what}")
        print(f"Stopped: {stop_reason}")
        (self.out / "stop_reason.txt").write_text(stop_reason, encoding="utf-8")

    # ---------- resume ----------

    def restore(self, used_anchors: set):
        """Rebuild the state at the end of the last saved round from the output folder.

        Everything the next round depends on is on disk: the schema, the score cache and the triplets.
        Jay's feedback rng is rebuilt by replaying select_feedback on each saved round's clusters
        (k-means on the saved vectors is deterministic), and the replay is checked against the saved
        triplets. LLM calls carry no conversation state, so the continued run matches an uninterrupted one."""
        hist_path = self.out / "history.csv"
        if not hist_path.exists():
            raise SystemExit(f"--resume: no history.csv in {self.out}; nothing to resume (start without --resume)")
        with hist_path.open(encoding="utf-8") as f:
            self.history = [{k: _parse_cell(v) for k, v in row.items()} for row in csv.DictReader(f)]
        edits_path = self.out / "edits.json"
        self.edits = json.loads(edits_path.read_text(encoding="utf-8")) if edits_path.exists() else []
        last = int(self.history[-1]["round"])
        self.edits = [e for e in self.edits if int(e["round"]) <= last]
        saved = np.load(self.out / "triplets.npy").reshape(-1, 3)

        def cluster(r):
            V = np.load(self.out / f"V_round_{r:02d}.npy")
            pred, _ = cluster_embeddings(V, algorithm="kmeans", n_clusters=self.k, random_state=self.cfg.seed)
            return V, np.asarray(pred)

        V, pred = cluster(0)
        baseline = pred.copy()
        offset = 0
        for r in range(1, last + 1):
            fb = next_feedback(pred, self.sel.y, self.sel.rng, used_anchors, self.cfg.triplets_per_round,
                               self.sel.test_mask)
            if fb is None:
                raise SystemExit(f"--resume: replay found no feedback for round {r}; folder doesn't match settings")
            anchor, _, new_triplets = fb
            got = np.asarray(new_triplets, dtype=np.int64).reshape(-1, 3)
            if not np.array_equal(got, saved[offset:offset + len(got)]):
                raise SystemExit(f"--resume: replaying round {r}'s feedback doesn't match triplets.npy; "
                                 "the folder doesn't match these settings")
            used_anchors.add(anchor)
            self.all_triplets.extend(new_triplets)
            offset += len(got)
            V, pred = cluster(r)
        if offset != len(saved):
            raise SystemExit(f"--resume: triplets.npy has {len(saved)} triplets, replay produced {offset}")
        schema = Schema.model_validate_json((self.out / f"schema_round_{last:02d}.json").read_text(encoding="utf-8"))
        row = self.history[-1]
        self.counters.parse_failures = int(row.get("llm_parse_failures") or 0)
        self.counters.filled_scores = int(row.get("filled_scores") or 0)
        log = self.client.log                       # so tokens, $ and --max-usd stay cumulative over the whole run
        log.calls, log.input_tokens, log.output_tokens = (int(row.get(c) or 0) for c in
                                                          ("llm_calls", "input_tokens", "output_tokens"))
        E = normalize(V, norm="l2")
        print(f"Resumed after round {last}: {len(self.all_triplets)} triplets so far, ARI={float(row['ari']):.4f}, "
              f"spent so far ${log.usd:.2f}")
        return schema, V, E, pred, baseline, last + 1

    def report(self, row: dict, extra: str = ""):
        usd = row["usd_total"]
        cost = f"${usd:.2f}" if usd == usd else f"{row['input_tokens']}/{row['output_tokens']} tok"
        print(f"Round {row['round']}: ARI={row['ari']:.4f}, test ARI={row['test_ari']:.4f}, "
              f"test pair AUC={row['test_pair_auc']:.3f}, F1={row['f1']:.4f}, features={row['n_active_features']}, "
              f"spent {cost}" + (f" | {extra}" if extra else ""))

    def save(self, schema: Schema, V, r: int, baseline, pred):
        """Written every round, so a crash or budget stop keeps everything up to that point."""
        jay = load_jay_module()
        (self.out / f"schema_round_{r:02d}.json").write_text(schema.model_dump_json(indent=2), encoding="utf-8")
        np.save(self.out / f"V_round_{r:02d}.npy", V)
        np.save(self.out / "triplets.npy", np.asarray(self.all_triplets, dtype=np.int64).reshape(-1, 3))
        write_csv(self.out / "history.csv", self.history)
        write_csv(self.out / "assignments.csv", [
            {"row_index": it.idx, "dex_num": it.dex_num, "name": it.name, "anon_id": it.anon_id,
             "label": it.label, "is_test": bool(self.sel.test_mask[it.idx]),
             "cluster_before": int(baseline[it.idx]), "cluster_after": int(pred[it.idx])} for it in self.items])
        (self.out / "edits.json").write_text(json.dumps(self.edits, indent=2, default=str), encoding="utf-8")
        if len(self.history) > 1:
            jay.plot_history(self.out / "scores.png", self.history)


def _parse_cell(v: str):
    """history.csv cell back to int / float where it was one (so the plot and goal check work)."""
    if v in ("", "True", "False"):
        return v
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v
