# Paper Soccer RL

Porównanie agentów RL w Paper Soccer (plan: `docs/paper_soccer_init_plan.md`,
algorytmy: `docs/paper_soccer_algorithms.md`).

## Struktura

```text
src/paper_soccer/
├── game.py            # czysty silnik reguł
├── opponents.py       # random, greedy (heurystyka); interfejs act(game, rng)
├── agents/
│   ├── base.py        # wspólny interfejs + argmax z losowym remisem
│   ├── tabular_q.py   # gotowe
│   └── __init__.py    # rejestr: nowy agent = nowy plik + 1 linijka
├── evaluation/arena.py  # play_game, evaluate_vs (obie strony)
├── training/runner.py   # pętla treningowa: checkpointy, --resume, milestones, logi CSV
└── utils/io.py          # zapis atomowy, CsvLogger
scripts/
├── train.py             # punkt wejścia (--algo --board --seed ...)
└── play_demo.py         # podgląd partii ASCII
cluster/eden/            # setup, Slurm, sync
tests/                   # pytest; fixtures/golden_games.json = partie nagrane na ORYGINALNYM env.py
```

## Lokalnie

```bash
uv sync --extra dev
uv run --extra dev pytest -q                     # ~2 s, wszystko ma przejść przed treningiem
uv run python scripts/train.py --board 4x4 --seed 0 --max-env-steps 200000 --milestones 10000,50000,200000
uv run python scripts/play_demo.py --board 4x4 --seed 1
```

## Eden (Slurm)

See [Eden setup and training](cluster/eden/README.md).

## Co jest w katalogu runu (`results/<algo>_<board>_s<seed>/`)

`config.json` (parametry + git rev), `train.csv`, `eval.csv` (win rate vs random/greedy, obie strony,
w punktach `--milestones`), `milestone_<N>.pkl`, `ckpt.pkl`, `final.pkl`, `done.json`.

## Dodawanie kolejnego algorytmu

1. `src/paper_soccer/agents/<nazwa>.py` z klasą mającą `act`, `train_episode`, `state_dict`,
   `load_state_dict`, `n_states`, liczniki `env_steps/decisions/episodes`.
2. Wpis w `agents/__init__.py`.
3. Test: legalność akcji + test wznowienia (wzór: `tests/test_resume.py`).

## Dokumentacja

- [Algorytmy](docs/paper_soccer_algorithms.md)
- [Plan eksperymentów](docs/paper_soccer_init_plan.md)

`uv.lock` wygeneruj przez `uv sync --extra dev`. Stary silnik pozostaje w historii Git.
