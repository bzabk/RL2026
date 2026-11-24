#!/usr/bin/env python3
"""Uruchamia trening jednego agenta (lokalnie albo z Slurma).

Przykład lokalny (szybki test):
    python scripts/train.py --algo tabular_q --board 4x4 --seed 0 \
        --max-env-steps 50000 --milestones 10000,50000 --out-dir results/tabular_q_4x4_s0

Wznowienie po przerwaniu:  dodaj --resume (te same argumenty).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_soccer.training.runner import TrainConfig, train  # noqa: E402


def main():
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default=d.algo)
    p.add_argument("--board", default=d.board, help="np. 4x4, 6x8, 8x10")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--out-dir", default=None, help="domyślnie results/<algo>_<board>_s<seed>")
    p.add_argument("--max-env-steps", type=int, default=d.max_env_steps)
    p.add_argument("--milestones", default=",".join(map(str, d.milestones)),
                   help="kroki środowiska, po których zapisać milestone + ewaluację")
    p.add_argument("--eval-games", type=int, default=d.eval_games)
    p.add_argument("--eval-opening-moves", type=int, default=d.eval_opening_moves,
                   help="losowe pierwsze N ruchów w ewaluacji (0 = ścisłe, deterministyczne partie)")
    p.add_argument("--alpha", type=float, default=d.alpha)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--eps-start", type=float, default=d.eps_start)
    p.add_argument("--eps-end", type=float, default=d.eps_end)
    p.add_argument("--eps-decay-frac", type=float, default=d.eps_decay_frac)
    p.add_argument("--greedy-frac", type=float, default=d.greedy_frac)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--ckpt-every-s", type=float, default=d.ckpt_every_s)
    p.add_argument("--time-limit-min", type=float, default=d.time_limit_min,
                   help="po tylu minutach zapisz checkpoint i wyjdź (ustaw < limit Slurma)")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    out_dir = a.out_dir or f"results/{a.algo}_{a.board}_s{a.seed}"
    cfg = TrainConfig(
        algo=a.algo, board=a.board, seed=a.seed, out_dir=out_dir,
        max_env_steps=a.max_env_steps,
        milestones=tuple(int(x) for x in a.milestones.split(",") if x),
        eval_games=a.eval_games, eval_opening_moves=a.eval_opening_moves, alpha=a.alpha, gamma=a.gamma,
        eps_start=a.eps_start, eps_end=a.eps_end, eps_decay_frac=a.eps_decay_frac,
        greedy_frac=a.greedy_frac, log_every=a.log_every, ckpt_every_s=a.ckpt_every_s,
        time_limit_min=a.time_limit_min, resume=a.resume,
    )
    status = train(cfg)
    sys.exit(0 if status == "done" else 3)  # 3 = przerwano, uruchom ponownie z --resume


if __name__ == "__main__":
    main()
