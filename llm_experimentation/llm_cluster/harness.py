"""The test harness: one interactive-clustering loop that any method can be plugged into.

It knows nothing about LLMs. A method is a `Strategy`, and the harness does everything that must be the
same across methods so their curves can be compared:

    round 0:  state = strategy.start()
    each round:
        V = strategy.embed(state)                 item vectors (n_items x d)
        pred = strategy.cluster(V, clusterer)     cluster labels (default: the src/utils Clusterer on V)
        metrics(V, pred)                          src/utils/metrics.all_metrics, incl. test ARI on held-out items
        hints = benchmark.sample_triplets(pred)   the simulated user (data.next_feedback)
        state = strategy.update(state, hints)
    until --rounds, the goal ARI, no more hints, or a StopRun (e.g. out of budget).

How this maps onto Kolade's src/utils framework (ClusterState.recluster_round). The data, feedback,
clusterer and metrics are the same src/utils objects; only the learning step differs:

    src/utils                                  here
    -----------------------------------------  -----------------------------------------------
    Benchmark (items, labels, test_mask)       data.Selection wraps the same PokemonBenchmark
    benchmark.sample_triplets(pred)            data.next_feedback calls it
    Clusterer.run(embeddings)                  Strategy.cluster(V, clusterer) calls it (direct: the LLM's groups)
    ClusterEval.update + adapter.learn         Strategy.update(state, RoundInput)
    adapter.predict / model(embeddings)        Strategy.embed(state)
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

from .data import Selection, next_feedback

from src.utils.clusterings import Clusterer  # noqa: E402  (repo root is on sys.path via .data)
from src.utils.metrics import all_metrics  # noqa: E402

MARGIN = 0.2        # a hint counts as satisfied when d(A, P) + MARGIN <= d(A, N)


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
    E: np.ndarray                   # current vectors, L2-normalized (the space the clusterer runs in)
    pred: np.ndarray                # current cluster labels
    history: list[dict]             # history.csv rows so far
    edits: list[dict]               # edits.json entries so far


def satisfied(E: np.ndarray, triplets) -> float:
    """Share of (A, P, N) triplets with d(A, P) + MARGIN <= d(A, N) in E (nan if there are none)."""
    if not len(triplets):
        return float("nan")
    a, p, n = np.asarray(triplets, dtype=int).T
    return float(np.mean(np.linalg.norm(E[a] - E[p], axis=1) + MARGIN <= np.linalg.norm(E[a] - E[n], axis=1) + 1e-6))


def write_csv(path: Path, rows: list[dict]):
    """rows -> a CSV file, with the first row's keys as the header."""
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot_history(path: Path, history: list[dict]):
    """ARI and test ARI per round, saved as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [row["round"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, [row["ari"] for row in history], marker="o", label="ARI (all items)")
    ax.plot(rounds, [row["test_ari"] for row in history], marker="s", label="test ARI (held-out items)")
    ax.set(xlabel="Feedback round", ylabel="ARI", title="Clustering vs. the hidden labels")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class Strategy:
    """The interface a method implements. `state` is its whole learned state between rounds, as a
    pydantic model so it can be saved every round (schema_round_XX.json) and loaded on --resume."""

    state_cls: type[BaseModel]
    name = "strategy"

    def start(self) -> BaseModel:
        """Round 0: the initial state."""
        raise NotImplementedError

    def embed(self, state) -> np.ndarray:
        """Vectors for all items under this state."""
        raise NotImplementedError

    def cluster(self, V: np.ndarray, clusterer: Clusterer) -> np.ndarray:
        """Cluster labels from vectors. Default: the run's src/utils Clusterer on the L2-normalized rows."""
        return clusterer.run(normalize(V))

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
    def __init__(self, sel: Selection, strategy: Strategy, cfg: HarnessConfig, out_dir: Path, clusterer: Clusterer):
        self.sel, self.strategy, self.cfg, self.out, self.clusterer = sel, strategy, cfg, out_dir, clusterer
        self.items = sel.items
        self.history: list[dict] = []       # -> history.csv
        self.edits: list[dict] = []         # -> edits.json
        self.all_triplets: list = []        # -> triplets.npy

    def metrics(self, E, pred) -> dict:
        s = self.sel
        return all_metrics(E, pred, s.y, s.test_mask if s.test_mask.any() else None, s.auc_pairs)

    def represent(self, state):
        """state -> (V, E = V with unit-length rows, pred)."""
        V = self.strategy.embed(state)
        return V, normalize(V), self.strategy.cluster(V, self.clusterer)

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
            fb = next_feedback(sel.benchmark, pred, cfg.triplets_per_round)
            if fb is None:
                stop_reason = "no remaining label/cluster disagreement"
                break
            anchor, target_cluster, new_triplets = fb
            before = satisfied(E, new_triplets)          # share of the new hints satisfied before the update
            try:
                state, changes = strat.update(state, RoundInput(r, new_triplets, list(self.all_triplets), E, pred,
                                                                self.history, self.edits))
                V, E, pred = self.represent(state)
            except StopRun as e:
                stop_reason = f"{e}. Round {r} was not saved; continue with the same command plus --resume"
                break
            self.all_triplets.extend(new_triplets)
            self.history.append({"round": r, **self.metrics(E, pred), "triplets_total": len(self.all_triplets),
                                 "new_triplets_before": before,
                                 "new_triplets_after": satisfied(E, new_triplets),
                                 "all_triplets_satisfied": satisfied(E, self.all_triplets),
                                 **strat.log_columns(state)})
            it = self.items[anchor]
            self.edits.append({"round": r, "anchor_index": int(anchor), "anon_id": it.anon_id, "name": it.name,
                               "target_label": it.label, "target_cluster": target_cluster, "changes": changes})
            self.save(state, V, r, baseline, pred)
            self.report(self.history[-1], f"{it.name} (to cluster {target_cluster}), {strat.summary(state, changes)}")
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
        # the feedback rng (benchmark.rng): all --resume needs to continue the feedback exactly
        (self.out / "feedback_state.json").write_text(json.dumps({"rng": self.sel.benchmark.rng.getstate()}),
                                                      encoding="utf-8")
        if len(self.history) > 1:
            plot_history(self.out / "scores.png", self.history)

    def restore(self):
        """--resume: load the end of the last saved round from the output folder."""
        if not (self.out / "feedback_state.json").exists():
            raise SystemExit(f"--resume: {self.out} has no saved round to continue from")
        with (self.out / "history.csv").open(encoding="utf-8") as f:
            self.history = [{k: _parse_cell(v) for k, v in row.items()} for row in csv.DictReader(f)]
        self.edits = json.loads((self.out / "edits.json").read_text(encoding="utf-8"))
        self.all_triplets = [tuple(t) for t in np.load(self.out / "triplets.npy").tolist()]
        version, internal, gauss = json.loads((self.out / "feedback_state.json").read_text(encoding="utf-8"))["rng"]
        self.sel.benchmark.rng.setstate((version, tuple(internal), gauss))     # JSON turned the tuples into lists
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
