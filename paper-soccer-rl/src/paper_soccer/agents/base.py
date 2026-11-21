"""Wspólny interfejs agentów.

Każdy agent (też przeciwnicy z ``opponents.py``) ma:

    act(game, rng, deterministic=True) -> int     # indeks LEGALNEJ akcji 0..7

Agenci uczący się dodatkowo mają (używa ich ``training/runner.py``):

    train_episode(game, opponent, side, rng, eps) -> dict   # statystyki epizodu
    state_dict() / load_state_dict(d)                       # zapis / wznowienie
    n_states (property, opcjonalnie)                        # do logów
"""
from __future__ import annotations

import numpy as np


def argmax_random_tiebreak(values: np.ndarray, rng) -> int:
    """argmax z losowym rozstrzyganiem remisów (przez przekazany rng)."""
    best = values.max()
    return int(rng.choice(np.flatnonzero(values == best)))
