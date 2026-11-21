"""llm_cluster/harness.py, tested with a tiny non-LLM strategy."""

import csv

import numpy as np
from pydantic import BaseModel

import helpers
from llm_cluster.harness import Harness, HarnessConfig, StopRun, Strategy


class Step(BaseModel):
    step: int = 0


class ToyStrategy(Strategy):
    """Vectors move from random noise to the true classes over 6 updates. Raises StopRun in round `stop_at`."""
    state_cls = Step

    def __init__(self, sel, stop_at=None):
        self.k, self.seed, self.stop_at = 3, 0, stop_at
        self.truth = np.eye(3)[np.unique(sel.y, return_inverse=True)[1]]
        self.noise = np.random.default_rng(0).normal(size=self.truth.shape)

    def start(self):
        return Step()

    def embed(self, state):
        t = min(1.0, state.step / 6)
        return (1 - t) * self.noise + t * 5 * self.truth

    def update(self, state, inp):
        if inp.round == self.stop_at:
            raise StopRun("out of budget")
        return Step(step=state.step + 1), {}


def run(sel, out, rounds, resume=False, stop_at=None, goal=1.5):
    out.mkdir(exist_ok=True)
    Harness(sel, ToyStrategy(sel, stop_at), HarnessConfig(rounds, 8, goal, resume), out).run()
    with (out / "history.csv").open(encoding="utf-8") as f:
        return [r["round"] for r in csv.DictReader(f)]


def test_run_saves_every_round_and_stops_at_goal(toy_sel, tmp_path):
    assert run(toy_sel, tmp_path, rounds=2) == ["0", "1", "2"]
    assert len(np.load(tmp_path / "triplets.npy")) == 16 and (tmp_path / "V_round_02.npy").exists()
    rounds = run(helpers.make_toy_sel(tmp_path), tmp_path / "goal", rounds=20, goal=0.95)
    assert len(rounds) < 21 and (tmp_path / "goal" / "stop_reason.txt").read_text() == "reached goal ARI"


def test_stoprun_keeps_earlier_rounds(toy_sel, tmp_path):
    assert run(toy_sel, tmp_path, rounds=5, stop_at=2) == ["0", "1"]
    assert "Round 2 was not saved" in (tmp_path / "stop_reason.txt").read_text()


def test_resume_matches_an_uninterrupted_run(tmp_path):
    run(helpers.make_toy_sel(tmp_path), tmp_path / "full", rounds=3)
    run(helpers.make_toy_sel(tmp_path), tmp_path / "split", rounds=1)
    assert run(helpers.make_toy_sel(tmp_path), tmp_path / "split", rounds=3, resume=True) == ["0", "1", "2", "3"]
    assert np.array_equal(np.load(tmp_path / "full" / "triplets.npy"), np.load(tmp_path / "split" / "triplets.npy"))
