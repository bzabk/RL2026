#!/usr/bin/env bash
# Użycie (zwykle przez submit.sh):  sbatch cluster/eden/train.sbatch <algo> <board> <seed> [max_env_steps]
#
# ZMIEŃ: --account (nazwa Waszej grupy na Edenie). Partycja z tutoriala Edena: student.
# Tabular/Linear Q to czysty CPU -> gpu:0. Dla Double DQN / Dyna zmień na --gres=gpu:1.
#SBATCH --partition=student
#SBATCH --account=GROUP_NAME
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=ps_train
#SBATCH --output=slurm_logs/%x-%j.log

set -euo pipefail

ALGO=${1:?algo}
BOARD=${2:?board, np. 4x4}
SEED=${3:?seed}
STEPS=${4:-1000000}

# katalog, z którego zrobiono sbatch (korzeń repo); katalog slurm_logs musi istnieć przed sbatch
cd "$SLURM_SUBMIT_DIR"
source "$HOME/venvs/paper-soccer/bin/activate"

OUT="results/${ALGO}_${BOARD}_s${SEED}"

# --resume: jeśli job był przerwany/uciął go limit czasu, ponowny sbatch dokończy od checkpointu.
# --time-limit-min: zapis checkpointu i czyste wyjście ~10 min przed limitem Slurma (2h = 120 min).
python scripts/train.py \
    --algo "$ALGO" --board "$BOARD" --seed "$SEED" \
    --max-env-steps "$STEPS" \
    --out-dir "$OUT" --resume --time-limit-min 110 \
    || { code=$?; [ "$code" -eq 3 ] && echo "PRZERWANO (limit czasu) - uruchom ponownie ten sam sbatch" && exit 0; exit "$code"; }
