#!/usr/bin/env python3
"""Podgląd partii w terminalu (debug reguł): greedy vs random albo wytrenowany agent.

    python scripts/play_demo.py --board 4x4 --seed 1
    python scripts/play_demo.py --board 4x4 --agent results/tabular_q_4x4_s0/final.pkl
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_soccer.agents.tabular_q import TabularQAgent  # noqa: E402
from paper_soccer.game import PaperSoccerGame, parse_board  # noqa: E402
from paper_soccer.opponents import GreedyOpponent, RandomOpponent  # noqa: E402
from paper_soccer.utils.io import load_pickle  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--board", default="4x4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--agent", default=None, help="plik final.pkl / milestone_*.pkl agenta tablicowego")
    a = p.parse_args()

    x, y = parse_board(a.board)
    game = PaperSoccerGame(x, y)
    rng = np.random.default_rng(a.seed)
    if a.agent:
        first = TabularQAgent()
        first.load_state_dict(load_pickle(a.agent))
    else:
        first = GreedyOpponent()
    players = {-1: first, 1: RandomOpponent()}
    game.reset()
    end = False
    while not end:
        who = game.current_player
        act = players[who].act(game, rng)
        _, _, end, _ = game.step(act, copy_grid=False)
    print(game.render())
    print("zwycięzca:", game.winner, "| powód:", game.termination_reason)


if __name__ == "__main__":
    main()
