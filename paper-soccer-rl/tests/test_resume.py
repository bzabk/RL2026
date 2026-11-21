"""Najważniejszy test pod klaster: przerwany i wznowiony trening == trening bez przerwy."""
import numpy as np

from paper_soccer.training.runner import TrainConfig, train
from paper_soccer.utils.io import load_pickle


def _cfg(out, **kw):
    base = dict(algo="tabular_q", board="4x4", seed=3, out_dir=str(out), max_env_steps=6000,
                milestones=(2000, 4000), eval_games=10, log_every=50, eps_decay_frac=0.5)
    base.update(kw)
    return TrainConfig(**base)


def test_resume_gives_identical_result(tmp_path):
    straight = tmp_path / "straight"
    assert train(_cfg(straight)) == "done"

    resumed = tmp_path / "resumed"
    assert train(_cfg(resumed, stop_after_episodes=40)) == "stopped"      # "ubity job"
    assert not (resumed / "done.json").exists()
    assert train(_cfg(resumed, resume=True)) == "done"                    # wznowienie

    a = load_pickle(straight / "final.pkl")
    b = load_pickle(resumed / "final.pkl")
    assert a["env_steps"] == b["env_steps"] and a["episodes"] == b["episodes"]
    assert a["decisions"] == b["decisions"]
    assert a["q"].keys() == b["q"].keys()
    for k in a["q"]:
        assert np.array_equal(a["q"][k], b["q"][k])


def test_outputs_exist_and_done_run_is_skipped(tmp_path):
    out = tmp_path / "run"
    assert train(_cfg(out)) == "done"
    for name in ["config.json", "ckpt.pkl", "final.pkl", "train.csv", "eval.csv", "done.json",
                 "milestone_2000.pkl", "milestone_4000.pkl"]:
        assert (out / name).exists(), name
    lines = (out / "eval.csv").read_text().strip().splitlines()
    assert lines[0].startswith("algo,board,seed")
    assert len(lines) > 1
    # ponowne uruchomienie ukończonego runu niczego nie psuje
    before = (out / "eval.csv").read_text()
    assert train(_cfg(out, resume=True)) == "done"
    assert (out / "eval.csv").read_text() == before


def test_existing_checkpoint_without_resume_is_refused(tmp_path):
    out = tmp_path / "run"
    train(_cfg(out, stop_after_episodes=10))
    try:
        train(_cfg(out))
        assert False, "powinno odmówić nadpisania checkpointu"
    except SystemExit:
        pass
