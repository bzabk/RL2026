# Paper Soccer RL

Porównanie agentów RL w Paper Soccer (plan: `docs/paper_soccer_init_plan.md`,
algorytmy: `docs/paper_soccer_algorithms.md`).

## Struktura

```text
src/paper_soccer/
├── game.py            # czysty silnik reguł (refaktor legacy/env_original.py)
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
├── play_demo.py         # podgląd partii ASCII
├── setup_env.sh         # venv na loginie Edena
├── sync_results.sh      # rsync wyników na Twój komputer
└── slurm/{train.sbatch,submit.sh}
tests/                   # pytest; golden_games.json = partie nagrane na ORYGINALNYM env.py
legacy/env_original.py   # oryginał (nie ruszać; do porównań)
```

## Lokalnie

```bash
pip install -r requirements.txt && pip install -e .
pytest -q                                    # ~2 s, wszystko ma przejść przed treningiem
python scripts/train.py --board 4x4 --seed 0 --max-env-steps 200000 --milestones 10000,50000,200000
python scripts/play_demo.py --board 4x4 --seed 1
```

## Eden (Slurm)

1. Logowanie spoza sieci wydziału (przez jump host) - dodaj do `~/.ssh/config`:
   ```
   Host eden
       HostName eden.mini.pw.edu.pl
       User EDEN_LOGIN
       ProxyJump MINI_LOGIN@ssh.mini.pw.edu.pl
   ```
2. `ssh eden`, potem `git clone <repo> paper-soccer-rl && cd paper-soccer-rl && bash scripts/setup_env.sh`
   (na węźle logowania; **nie liczymy tam niczego**).
3. W `scripts/slurm/train.sbatch` wpisz swoje `--account=` (nazwa grupy).
4. Szybki test na węźle: `srun -p student -A GROUP --cpus-per-task=2 --mem=4G --time=00:15:00 --pty bash`
   (w `tmux`), tam `pytest -q` i krótki `train.py`.
5. Joby: `bash scripts/slurm/submit.sh tabular_q 4x4 "0 1 2 3 4"`; podgląd `squeue -u $USER`,
   logi w `slurm_logs/`.
6. Przerwany job (limit czasu): ten sam `submit.sh` ponownie - `--resume` dokończy od checkpointu,
   ukończone runy (`done.json`) są pomijane.
7. Na swoim komputerze: `bash scripts/sync_results.sh` (tylko logi) lub `--all` (z modelami).

## Co jest w katalogu runu (`results/<algo>_<board>_s<seed>/`)

`config.json` (parametry + git rev), `train.csv`, `eval.csv` (win rate vs random/greedy, obie strony,
w punktach `--milestones`), `milestone_<N>.pkl`, `ckpt.pkl`, `final.pkl`, `done.json`.

## Dodawanie kolejnego algorytmu

1. `src/paper_soccer/agents/<nazwa>.py` z klasą mającą `act`, `train_episode`, `state_dict`,
   `load_state_dict`, `n_states`, liczniki `env_steps/decisions/episodes`.
2. Wpis w `agents/__init__.py`.
3. Test: legalność akcji + test wznowienia (wzór: `tests/test_resume.py`).
