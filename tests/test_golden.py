"""Test charakterystyczny: nowy silnik musi odtworzyć 120 partii nagranych na ORYGINALNYM env.py
(patrz ``original env.py in Git history``). Jeśli ten test spadnie, refaktor zmienił reguły gry."""
import json
from pathlib import Path

import numpy as np

from paper_soccer.game import PaperSoccerGame

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "golden_games.json").read_text())


def test_golden_games_replay_identically():
    assert len(GOLDEN) == 120
    for rec in GOLDEN:
        game = PaperSoccerGame(rec["x"], rec["y"], rec["g"])
        game.reset()
        players, end, grid, rewards = [], False, None, None
        for a in rec["actions"]:
            assert not end
            players.append(int(game.current_player))
            grid, rewards, end, _ = game.step(a)
        assert end
        assert players == rec["players"]
        assert {str(k): v for k, v in rewards.items()} == rec["rewards"]
        assert list(game.cur_pos) == rec["final_pos"]
        assert int(game.current_player) == rec["final_player"]
        assert int(grid.sum()) == rec["grid_sum"]
        h = int((grid.astype(np.int64) * np.arange(grid.size).reshape(grid.shape)).sum())
        assert h == rec["grid_hash"]
