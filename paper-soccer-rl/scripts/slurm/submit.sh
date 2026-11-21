#!/usr/bin/env bash
# Wysyła po jednym jobie na seed.  Uruchamiaj z korzenia repo, NA WĘŹLE LOGOWANIA Edena.
#
#   bash scripts/slurm/submit.sh tabular_q 4x4 "0 1 2 3 4"
#   bash scripts/slurm/submit.sh tabular_q 8x10 "0 1 2" 5000000
set -euo pipefail
ALGO=${1:?algo}
BOARD=${2:?board}
SEEDS=${3:-"0 1 2"}
STEPS=${4:-1000000}

mkdir -p slurm_logs results   # Slurm nie tworzy katalogu na logi sam
for s in $SEEDS; do
  sbatch --job-name="${ALGO}_${BOARD}_s${s}" scripts/slurm/train.sbatch "$ALGO" "$BOARD" "$s" "$STEPS"
done
echo "Podgląd: squeue -u \$USER   |   log: tail -f slurm_logs/*.log"
