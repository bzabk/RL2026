"""Przeciwnicy testowi i treningowi.

Wspólny interfejs (taki sam jak agenci): ``act(game, rng, deterministic=True) -> int``.
Losowość zawsze przez przekazany ``rng`` (numpy Generator) - dzięki temu partie są odtwarzalne.
"""
from __future__ import annotations

import numpy as np


class RandomOpponent:
    name = "random"

    def act(self, game, rng, deterministic: bool = True) -> int:
        return int(rng.choice(game.legal_actions()))


class GreedyOpponent:
    """Prosta heurystyka (drabinka trudności: random < greedy).

    Kolejność reguł dla każdego legalnego ruchu ``a`` (gracz ``me`` na ruchu):
      1. ruch dający natychmiastową wygraną  -> wybierz go,
      2. ruch kończący grę porażką            -> unikaj,
      3. (opcjonalnie) ruch po którym przeciwnik ma natychmiastową wygraną -> duża kara,
      4. wynik = postęp w stronę bramki rywala + bonus za odbicie (zachowuję ruch)
                 + mała premia za liczbę wyjść z nowego punktu (unikanie ślepych uliczek).
    Remisy rozstrzyga losowy szum z ``rng``.
    """

    name = "greedy"

    def __init__(self, check_reply: bool = True):
        self.check_reply = check_reply

    def act(self, game, rng, deterministic: bool = True) -> int:
        me = game.current_player
        goal_y = game.goal_row(me)
        y0 = game.cur_pos[0]
        best_a, best_score = None, -np.inf
        for a in game.legal_actions():
            a = int(a)
            g2, rewards, end = game.peek(a)
            if end:
                if rewards[me] == 1:
                    return a
                continue  # natychmiastowa porażka
            new_y = g2.cur_pos[0]
            progress = abs(y0 - goal_y) - abs(new_y - goal_y)  # +1 / 0 / -1
            bounce = g2.current_player == me
            exits = int(g2.action_masks().sum())
            score = progress + (1.5 if bounce else 0.0) + 0.1 * exits
            if self.check_reply and not bounce:
                opp = g2.current_player
                for b in g2.legal_actions():
                    _, r2, e2 = g2.peek(int(b))
                    if e2 and r2[opp] == 1:
                        score -= 10.0
                        break
            score += 1e-3 * rng.random()
            if score > best_score:
                best_a, best_score = a, score
        if best_a is None:  # każdy ruch przegrywa - zagraj cokolwiek legalnego
            best_a = int(rng.choice(game.legal_actions()))
        return best_a


def make_opponent(name: str):
    if name == "random":
        return RandomOpponent()
    if name == "greedy":
        return GreedyOpponent()
    raise ValueError(f"nieznany przeciwnik: {name}")
