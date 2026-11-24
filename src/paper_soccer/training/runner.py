"""Pętla treningowa odporna na przerwanie joba (checkpointy + --resume).

Układ katalogu runu (``out_dir``)::

    config.json            # pełna konfiguracja + wersja kodu (git) - odtwarzalność
    ckpt.pkl               # ostatni checkpoint (agent + stan RNG + liczniki) - do --resume
    milestone_<N>.pkl      # agent po N krokach środowiska (do E2 / turnieju)
    final.pkl              # agent po zakończeniu
    train.csv              # postęp treningu
    eval.csv               # ewaluacja vs random/greedy (obie strony), w punktach kontrolnych
    done.json              # znacznik "ukończone" (ponowne sbatch nic nie psuje)
"""
from __future__ import annotations

import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..agents import make_agent
from ..evaluation.arena import evaluate_vs
from ..game import PaperSoccerGame, parse_board
from ..opponents import GreedyOpponent, RandomOpponent, make_opponent
from ..utils.io import CsvLogger, load_pickle, save_pickle_atomic, write_json_atomic

TRAIN_FIELDS = ["episodes", "env_steps", "decisions", "eps", "avg_moves", "train_win_rate",
                "n_states", "wall_s"]
EVAL_FIELDS = ["algo", "board", "seed", "episodes", "env_steps", "decisions", "train_wall_s",
               "n_states", "opponent", "side", "games", "opening_moves", "wins", "win_rate", "avg_moves",
               "avg_bounces", "reasons"]


@dataclass
class TrainConfig:
    algo: str = "tabular_q"
    board: str = "4x4"
    gate_size: int = 2
    seed: int = 0
    out_dir: str = "results/run"
    max_env_steps: int = 1_000_000
    # eksploracja
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.5          # ułamek budżetu, w którym eps spada liniowo
    # przeciwnicy treningowi
    greedy_frac: float = 0.5             # reszta = random
    # hiperparametry agenta
    alpha: float = 0.1
    gamma: float = 0.99
    # ewaluacja / logi / checkpointy
    milestones: tuple = (10_000, 50_000, 100_000, 500_000, 1_000_000)
    eval_games: int = 200                # na stronę, na przeciwnika
    eval_opening_moves: int = 2          # losowe otwarcie w ewaluacji (0 = ścisły protokół deterministyczny)
    log_every: int = 1000                # epizodów
    ckpt_every_s: float = 300.0
    time_limit_min: float = 0.0          # 0 = bez limitu; ustaw < limit Slurma
    resume: bool = False
    stop_after_episodes: int = 0         # tylko do testów (symuluje przerwanie joba)
    extra: dict = field(default_factory=dict)


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True, cwd=Path(__file__).parent).strip()
    except Exception:
        return "unknown"


class _StopFlag:
    """SIGTERM/SIGUSR1 (scancel, limit czasu) -> dokończ epizod, zapisz checkpoint, wyjdź."""

    def __init__(self):
        self.set = False
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            try:
                signal.signal(sig, self._handler)
            except (ValueError, OSError):  # np. nie w głównym wątku
                pass

    def _handler(self, signum, frame):
        self.set = True


def train(cfg: TrainConfig) -> str:
    """Zwraca 'done' albo 'stopped' (przerwano - uruchom ponownie z resume=True)."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if (out / "done.json").exists():
        print(f"[train] {out} już ukończony - pomijam")
        return "done"

    board = parse_board(cfg.board)
    game = PaperSoccerGame(board[0], board[1], cfg.gate_size)
    agent = make_agent(cfg.algo, alpha=cfg.alpha, gamma=cfg.gamma)
    rng = np.random.default_rng(cfg.seed)
    wall_before = 0.0
    next_milestone = 0
    ms = sorted(cfg.milestones)

    ckpt_path = out / "ckpt.pkl"
    if cfg.resume and ckpt_path.exists():
        ck = load_pickle(ckpt_path)
        agent.load_state_dict(ck["agent"])
        rng.bit_generator.state = ck["rng_state"]
        wall_before = ck["wall_s"]
        next_milestone = ck["next_milestone"]
        print(f"[train] wznowiono: episodes={agent.episodes} env_steps={agent.env_steps}")
    elif not cfg.resume and ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} istnieje - użyj --resume albo innego --out-dir")

    write_json_atomic({**asdict(cfg), "git_rev": _git_rev()}, out / "config.json")

    train_log = CsvLogger(out / "train.csv", TRAIN_FIELDS)
    eval_log = CsvLogger(out / "eval.csv", EVAL_FIELDS)
    opponents = [RandomOpponent(), GreedyOpponent()]
    eval_opponents = [RandomOpponent(), GreedyOpponent()]
    stop = _StopFlag()
    t0 = time.monotonic()

    def wall() -> float:
        return wall_before + (time.monotonic() - t0)

    def save_ckpt():
        save_pickle_atomic({"agent": agent.state_dict(), "rng_state": rng.bit_generator.state,
                            "wall_s": wall(), "next_milestone": next_milestone}, ckpt_path)

    last_eval = {"steps": -1}

    def evaluate(tag_steps: int):
        if last_eval["steps"] == agent.env_steps:   # ten sam stan już ewaluowany (np. milestone = koniec)
            return
        last_eval["steps"] = agent.env_steps
        for opp in eval_opponents:
            for row in evaluate_vs(agent, opp, board, cfg.eval_games, seed=cfg.seed,
                                   gate_size=cfg.gate_size, opening_moves=cfg.eval_opening_moves):
                eval_log.log({"algo": cfg.algo, "board": cfg.board, "seed": cfg.seed,
                              "episodes": agent.episodes, "env_steps": agent.env_steps,
                              "decisions": agent.decisions, "train_wall_s": round(wall(), 2),
                              "n_states": agent.n_states, **row})
        print(f"[eval] env_steps={agent.env_steps} (punkt {tag_steps})", flush=True)

    window_moves, window_wins, last_ckpt = [], [], time.monotonic()
    status = "done"
    decay_steps = max(1.0, cfg.eps_decay_frac * cfg.max_env_steps)
    episodes_this_run = 0

    while agent.env_steps < cfg.max_env_steps:
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * agent.env_steps / decay_steps)
        side = -1 if agent.episodes % 2 == 0 else 1          # równowaga stron
        opp = opponents[1] if rng.random() < cfg.greedy_frac else opponents[0]
        info = agent.train_episode(game, opp, side, rng, eps)
        episodes_this_run += 1
        window_moves.append(info["moves"])
        window_wins.append(info["win"])

        if agent.episodes % cfg.log_every == 0:
            train_log.log({"episodes": agent.episodes, "env_steps": agent.env_steps,
                           "decisions": agent.decisions, "eps": round(eps, 4),
                           "avg_moves": round(float(np.mean(window_moves)), 2),
                           "train_win_rate": round(float(np.mean(window_wins)), 4),
                           "n_states": agent.n_states, "wall_s": round(wall(), 1)})
            window_moves, window_wins = [], []

        while next_milestone < len(ms) and agent.env_steps >= ms[next_milestone]:
            save_pickle_atomic(agent.state_dict(), out / f"milestone_{ms[next_milestone]}.pkl")
            evaluate(ms[next_milestone])
            next_milestone += 1

        if time.monotonic() - last_ckpt > cfg.ckpt_every_s:
            save_ckpt()
            last_ckpt = time.monotonic()

        over_limit = cfg.time_limit_min > 0 and (time.monotonic() - t0) > cfg.time_limit_min * 60
        test_stop = cfg.stop_after_episodes and episodes_this_run >= cfg.stop_after_episodes
        if stop.set or over_limit or test_stop:
            status = "stopped"
            break

    save_ckpt()
    if status == "done":
        save_pickle_atomic(agent.state_dict(), out / "final.pkl")
        evaluate(agent.env_steps)
        write_json_atomic({"env_steps": agent.env_steps, "episodes": agent.episodes,
                           "wall_s": wall(), "n_states": agent.n_states}, out / "done.json")
    else:
        print(f"[train] przerwano na env_steps={agent.env_steps}; wznów z --resume", flush=True)
    train_log.close()
    eval_log.close()
    return status
