"""Rejestr agentów. Nowy algorytm = nowy plik + jedna linijka tutaj."""
from __future__ import annotations

from .tabular_q import TabularQAgent

# Kolejne (zgodnie z planem): linear_q, double_dqn, dyna_double_dqn, maskable_ppo, a2c
# Agenci z PyTorchem importowani leniwie, żeby tabular/linear działały bez torch.
AGENTS = {
    "tabular_q": TabularQAgent,
}


def make_agent(algo: str, **kwargs):
    if algo not in AGENTS:
        raise ValueError(f"nieznany algorytm '{algo}'. Dostępne: {sorted(AGENTS)}")
    return AGENTS[algo](**kwargs)
