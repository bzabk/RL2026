#!/usr/bin/env bash
# Uruchamiaj NA SWOIM KOMPUTERZE (nie na klastrze). Ściąga wyniki z Edena do ./results/.
# Wymaga aliasu "eden" w ~/.ssh/config (patrz README). Dociąga tylko nowe/zmienione pliki.
#
#   bash cluster/eden/sync_results.sh            # logi i configi (bez dużych modeli)
#   bash cluster/eden/sync_results.sh --all      # razem z checkpointami
set -euo pipefail
REMOTE_DIR=${REMOTE_DIR:-RL2026}      # ścieżka repo na klastrze (względem home)
EXCLUDES=(--exclude '*.pkl' --exclude '*.pt' --exclude '*.tmp')
[ "${1:-}" = "--all" ] && EXCLUDES=(--exclude '*.tmp')
mkdir -p results
rsync -avz --progress "${EXCLUDES[@]}" "eden:${REMOTE_DIR}/results/" ./results/
