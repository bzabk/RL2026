"""Rozgrywanie partii i ewaluacja agenta przeciwko stałym przeciwnikom (obie strony)."""
from __future__ import annotations

import numpy as np

from ..game import PaperSoccerGame


def play_game(players: dict, game: PaperSoccerGame, rng, deterministic: bool = True,
              opening_moves: int = 0) -> dict:
    """``players``: {-1: agent, 1: agent}. Zwraca wynik partii (z zapisem akcji do odtworzenia).

    ``opening_moves``: pierwsze N ruchów (obu graczy) losowe legalne - różnicuje partie, gdy obie
    polityki są deterministyczne (inaczej "100 partii" to w praktyce jedna partia).
    """
    game.reset()
    actions, bounces = [], 0
    end = False
    while not end:
        who = game.current_player
        if len(actions) < opening_moves:
            a = int(rng.choice(game.legal_actions()))
        else:
            a = players[who].act(game, rng, deterministic=deterministic)
        actions.append(int(a))
        _, _, end, nxt = game.step(a, copy_grid=False)
        if not end and nxt == who:
            bounces += 1
    return {
        "winner": game.winner,
        "reason": game.termination_reason,
        "n_moves": len(actions),
        "bounces": bounces,
        "actions": actions,
    }


def evaluate_vs(agent, opponent, board: tuple[int, int], n_games: int, seed: int,
                gate_size: int = 2, opening_moves: int = 0) -> list[dict]:
    """Win rate agenta przeciwko ``opponent``; ``n_games`` partii na KAŻDĄ stronę.

    Używa własnego rng (zależnego od ``seed``), więc nie zaburza losowości treningu.
    """
    game = PaperSoccerGame(board[0], board[1], gate_size)
    rows = []
    for side in (-1, 1):
        rng = np.random.default_rng([seed, 7919, side + 1])
        wins = moves = bounces = 0
        reasons: dict[str, int] = {}
        for _ in range(n_games):
            res = play_game({side: agent, -side: opponent}, game, rng, opening_moves=opening_moves)
            wins += int(res["winner"] == side)
            moves += res["n_moves"]
            bounces += res["bounces"]
            reasons[res["reason"]] = reasons.get(res["reason"], 0) + 1
        rows.append({
            "opponent": opponent.name,
            "side": side,
            "games": n_games,
            "opening_moves": opening_moves,
            "wins": wins,
            "win_rate": wins / n_games,
            "avg_moves": moves / n_games,
            "avg_bounces": bounces / n_games,
            "reasons": ";".join(f"{k}={v}" for k, v in sorted(reasons.items())),
        })
    return rows
