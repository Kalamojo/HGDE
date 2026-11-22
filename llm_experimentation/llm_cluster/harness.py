"""The test harness: one interactive-clustering loop that any method can be plugged into.

It knows nothing about LLMs. A method is a `Strategy`, and the harness does everything that must be the
same across methods so their curves can be compared:

    round 0:  state = strategy.start()
    each round:
        V = strategy.embed(state)            item vectors (n_items x d)
        pred = strategy.cluster(V)           cluster labels (default: Jay's k-means call)
        metrics(V, pred)                     src/utils/metrics.all_metrics, incl. test ARI on held-out items
        hints = Jay's select_feedback(pred)  the simulated user (data.next_feedback)
        state = strategy.update(state, hints)
    until --rounds, the goal ARI, no more hints, or a StopRun (e.g. out of budget).

How this maps onto Kolade's src/utils framework (ClusterState.recluster_round):

    src/utils                                  here
    -----------------------------------------  -----------------------------------------------
    Benchmark (items, labels, test_mask)       data.Selection (Jay's selection, exactly)
    benchmark.sample_triplets(pred)            data.next_feedback (Jay's select_feedback)
    Clusterer.run(embeddings)                  Strategy.cluster(V)
    ClusterEval.update + train_model           Strategy.update(state, RoundInput)
    model(original_embeddings)                 Strategy.embed(state)
    ClusterState.log_results / all_metrics     Harness.metrics / all_metrics (the same function)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import normalize

from .data import Selection, load_jay_module, next_feedback

from archive.clustering import cluster_embeddings  # noqa: E402  (repo root is on sys.path via .data)
from archive.scripts.triplet_learning_testing import triplet_stats, write_csv  # noqa: E402
from src.utils.metrics import all_metrics  # noqa: E402


@dataclass
class HarnessConfig:
    """Settings of the loop itself; the same for every method being compared."""
    rounds: int = 30
    triplets_per_round: int = 32
    goal_ari: float = 0.95          # stop early once ARI on all items reaches this
    resume: bool = False


class StopRun(Exception):
    """Raised by a strategy to end the run cleanly. The round in progress is not saved."""


@dataclass
class RoundInput:
    """What a strategy may use when it updates."""
    round: int
    new_triplets: list              # this round's (A, P, N) index triplets
    earlier_triplets: list          # all triplets from earlier rounds
    E: np.ndarray                   # current vectors, L2-normalized (the space k-means clusters in)
    pred: np.ndarray                # current cluster labels
    history: list[dict]             # history.csv rows so far
    edits: list[dict]               # edits.json entries so far


def kmeans(V: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Jay's clustering call, unchanged (L2-normalizes the rows, then KMeans with n_init=20)."""
    return np.asarray(cluster_embeddings(V, algorithm="kmeans", n_clusters=k, random_state=seed)[0])


class Strategy:
    """The interface a method implements. `state` is its whole learned state between rounds, as a
    pydantic model so it can be saved every round (schema_round_XX.json) and loaded on --resume."""

    state_cls: type[BaseModel]
    name = "strategy"
    k: int                          # clusters to make; subclasses set it
    seed: int                       # for k-means; subclasses set it

    def start(self) -> BaseModel:
        """Round 0: the initial state."""
        raise NotImplementedError

    def embed(self, state) -> np.ndarray:
        """Vectors for all items under this state."""
        raise NotImplementedError

    def cluster(self, V: np.ndarray) -> np.ndarray:
        """Cluster labels from vectors. Default: Jay's k-means call."""
        return kmeans(V, self.k, self.seed)

    def update(self, state, inp: RoundInput) -> tuple[BaseModel, dict]:
        """Learn from this round's hints. Returns (new state, what changed, for edits.json)."""
        raise NotImplementedError

    def log_columns(self, state) -> dict:
        """Extra history.csv columns."""
        return {}

    def summary(self, state, changes: dict) -> str:
        """A short console line about this round."""
        return ""

    def restore(self, last_row: dict) -> None:
        """--resume: restore running counters from the last history row."""


class Harness:
    def __init__(self, sel: Selection, strategy: Strategy, cfg: HarnessConfig, out_dir: Path):
        self.sel, self.strategy, self.cfg, self.out = sel, strategy, cfg, out_dir
        self.items = sel.items
        self.history: list[dict] = []       # -> history.csv
        self.edits: list[dict] = []         # -> edits.json
        self.all_triplets: list = []        # -> triplets.npy
        self.used_anchors: set = set()      # Jay's feedback never picks the same anchor twice

    def metrics(self, E, pred) -> dict:
        s = self.sel
        return all_metrics(E, pred, s.y, s.test_mask if s.test_mask.any() else None, s.auc_pairs)

    def represent(self, state):
        """state -> (V, E = V with unit-length rows, pred)."""
        V = self.strategy.embed(state)
        return V, normalize(V), self.strategy.cluster(V)

    def run(self):
        cfg, sel, strat = self.cfg, self.sel, self.strategy
        print(f"Items={len(self.items)}, classes={sel.n_classes}, test items={int(sel.test_mask.sum())}, strategy={strat.name}")
        stop_reason = "completed all rounds"
        if cfg.resume:
            state, V, E, pred, baseline, first_round = self.restore()
        else:
            try:
                state = strat.start()
                V, E, pred = self.represent(state)
            except StopRun as e:
                print(f"Stopped before round 0: {e}")
                return
            baseline, first_round = pred.copy(), 1      # baseline: round-0 clusters, for assignments.csv
            self.history.append({"round": 0, **self.metrics(E, pred), "triplets_total": 0, "new_triplets_before": "",
                                 "new_triplets_after": "", "all_triplets_satisfied": "", **strat.log_columns(state)})
            self.save(state, V, 0, baseline, pred)
            self.report(self.history[-1], strat.summary(state, {}))

        for r in range(first_round, cfg.rounds + 1):
            if float(self.history[-1]["ari"]) >= cfg.goal_ari:
                stop_reason = "reached goal ARI"
                break
            fb = next_feedback(pred, sel.y, sel.rng, self.used_anchors, cfg.triplets_per_round, sel.test_mask)
            if fb is None:
                stop_reason = "no remaining label/cluster disagreement"
                break
            anchor, kind, new_triplets = fb
            self.used_anchors.add(anchor)
            before = triplet_stats(E, new_triplets)      # share of the new hints satisfied before the update
            try:
                state, changes = strat.update(state, RoundInput(r, new_triplets, list(self.all_triplets), E, pred,
                                                                self.history, self.edits))
                V, E, pred = self.represent(state)
            except StopRun as e:
                stop_reason = f"{e}. Round {r} was not saved; continue with the same command plus --resume"
                break
            self.all_triplets.extend(new_triplets)
            self.history.append({"round": r, **self.metrics(E, pred), "triplets_total": len(self.all_triplets),
                                 "new_triplets_before": before["distance_triplet_satisfaction"],
                                 "new_triplets_after": triplet_stats(E, new_triplets)["distance_triplet_satisfaction"],
                                 "all_triplets_satisfied": triplet_stats(E, self.all_triplets)["distance_triplet_satisfaction"],
                                 **strat.log_columns(state)})
            it = self.items[anchor]
            self.edits.append({"round": r, "anchor_index": int(anchor), "anon_id": it.anon_id, "name": it.name,
                               "target_label": it.label, "feedback_type": kind, "changes": changes})
            self.save(state, V, r, baseline, pred)
            self.report(self.history[-1], f"{it.name} ({kind}), {strat.summary(state, changes)}")
        print(f"Stopped: {stop_reason}")
        (self.out / "stop_reason.txt").write_text(stop_reason, encoding="utf-8")

    def report(self, row: dict, extra: str = ""):
        print(f"Round {row['round']}: ARI={row['ari']:.4f}, test ARI={row['test_ari']:.4f}, "
              f"test pair AUC={row['test_pair_auc']:.3f}" + (f" | {extra}" if extra else ""))

    def save(self, state, V, r: int, baseline, pred):
        """Write everything after every round, so a crash or a budget stop keeps all finished rounds."""
        (self.out / f"schema_round_{r:02d}.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
        np.save(self.out / f"V_round_{r:02d}.npy", V)
        np.save(self.out / "triplets.npy", np.asarray(self.all_triplets, dtype=np.int64).reshape(-1, 3))
        write_csv(self.out / "history.csv", self.history)
        write_csv(self.out / "assignments.csv", [
            {"row_index": it.idx, "dex_num": it.dex_num, "name": it.name, "anon_id": it.anon_id, "label": it.label,
             "is_test": bool(self.sel.test_mask[it.idx]), "cluster_before": int(baseline[it.idx]),
             "cluster_after": int(pred[it.idx])} for it in self.items])
        (self.out / "edits.json").write_text(json.dumps(self.edits, indent=2, default=str), encoding="utf-8")
        # the feedback rng and used anchors: all --resume needs to continue the feedback exactly
        (self.out / "feedback_state.json").write_text(json.dumps(
            {"rng": self.sel.rng.bit_generator.state, "used_anchors": sorted(int(a) for a in self.used_anchors)}),
            encoding="utf-8")
        if len(self.history) > 1:
            load_jay_module().plot_history(self.out / "scores.png", self.history)

    def restore(self):
        """--resume: load the end of the last saved round from the output folder."""
        if not (self.out / "feedback_state.json").exists():
            raise SystemExit(f"--resume: {self.out} has no saved round to continue from")
        with (self.out / "history.csv").open(encoding="utf-8") as f:
            self.history = [{k: _parse_cell(v) for k, v in row.items()} for row in csv.DictReader(f)]
        self.edits = json.loads((self.out / "edits.json").read_text(encoding="utf-8"))
        self.all_triplets = [tuple(t) for t in np.load(self.out / "triplets.npy").tolist()]
        fb = json.loads((self.out / "feedback_state.json").read_text(encoding="utf-8"))
        self.sel.rng.bit_generator.state = fb["rng"]
        self.used_anchors = set(fb["used_anchors"])
        with (self.out / "assignments.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        baseline = np.array([int(r["cluster_before"]) for r in rows])
        pred = np.array([int(r["cluster_after"]) for r in rows])
        last = int(self.history[-1]["round"])
        V = np.load(self.out / f"V_round_{last:02d}.npy")
        state = self.strategy.state_cls.model_validate_json(
            (self.out / f"schema_round_{last:02d}.json").read_text(encoding="utf-8"))
        self.strategy.restore(self.history[-1])
        print(f"Resumed after round {last}: {len(self.all_triplets)} triplets so far")
        return state, V, normalize(V), pred, baseline, last + 1


def _parse_cell(v: str):
    """A history.csv cell back to a number where it was one."""
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v
