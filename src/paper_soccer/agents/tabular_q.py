"""Tabular Q-learning (baseline nr 1).

Konwencja "od decyzji do decyzji" (adapter jednoagentowy z planu, sekcja 4.3):
uczący się agent gra jako ``side``; ruchy przeciwnika są rozgrywane wewnątrz epizodu.
Wartość Q jest zawsze z perspektywy agenta, więc zwykły target Q-learningu jest poprawny
(bez zmiany znaku). Odbicie = kolejna decyzja tego samego gracza - też obsłużone poprawnie.

Klucz stanu: ``game.state_key()`` = (zapakowane bity planszy, pozycja, gracz).
"""
from __future__ import annotations

import numpy as np

from .base import argmax_random_tiebreak


class TabularQAgent:
    name = "tabular_q"

    def __init__(self, alpha: float = 0.1, gamma: float = 0.99):
        self.alpha = alpha
        self.gamma = gamma
        self.q: dict[bytes, np.ndarray] = {}
        # liczniki budżetu (plan, sekcja 5: ruchy obu graczy + decyzje agenta osobno)
        self.env_steps = 0
        self.decisions = 0
        self.episodes = 0

    @property
    def n_states(self) -> int:
        return len(self.q)

    # ---------------------------------------------------------------- Q-tablica
    def _row(self, game, key: bytes) -> np.ndarray:
        q = self.q.get(key)
        if q is None:
            q = np.zeros(8, dtype=np.float32)
            q[~game.action_masks()] = -np.inf  # nielegalne akcje nigdy nie wygrają argmax
            self.q[key] = q
        return q

    # ---------------------------------------------------------------- polityka
    def act(self, game, rng, deterministic: bool = True) -> int:
        q = self.q.get(game.state_key())
        if q is None:  # stan nieznany: losowy legalny ruch
            return int(rng.choice(game.legal_actions()))
        return argmax_random_tiebreak(q, rng)

    def _act_eps(self, game, q, rng, eps: float) -> int:
        if rng.random() < eps:
            return int(rng.choice(game.legal_actions()))
        return argmax_random_tiebreak(q, rng)

    # ---------------------------------------------------------------- trening
    def train_episode(self, game, opponent, side: int, rng, eps: float) -> dict:
        game.reset()
        prev_q, prev_a = None, None
        moves = 0
        rewards = {1: 0, -1: 0}
        end = False
        while not end:
            if game.current_player == side:
                q = self._row(game, game.state_key())
                if prev_q is not None:  # r=0, bootstrap z nowej decyzji
                    prev_q[prev_a] += self.alpha * (self.gamma * q.max() - prev_q[prev_a])
                a = self._act_eps(game, q, rng, eps)
                prev_q, prev_a = q, a
                self.decisions += 1
            else:
                a = opponent.act(game, rng, deterministic=False)
            _, rewards, end, _ = game.step(a, copy_grid=False)
            moves += 1
        if prev_q is not None:  # nagroda terminalna z perspektywy agenta
            prev_q[prev_a] += self.alpha * (rewards[side] - prev_q[prev_a])
        self.env_steps += moves
        self.episodes += 1
        return {"moves": moves, "win": int(rewards[side] == 1)}

    # ---------------------------------------------------------------- zapis
    def state_dict(self) -> dict:
        return {
            "q": self.q,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "env_steps": self.env_steps,
            "decisions": self.decisions,
            "episodes": self.episodes,
        }

    def load_state_dict(self, d: dict) -> None:
        self.q = d["q"]
        self.alpha = d["alpha"]
        self.gamma = d["gamma"]
        self.env_steps = d["env_steps"]
        self.decisions = d["decisions"]
        self.episodes = d["episodes"]
