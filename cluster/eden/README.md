# Eden (Slurm)

Polecenia wykonuj z głównego katalogu repozytorium (`~/RL2026` na Edenie).

1. Logowanie spoza sieci wydziału (przez jump host) - dodaj do `~/.ssh/config`:
   ```
   Host eden
       HostName eden.mini.pw.edu.pl
       User EDEN_LOGIN
       ProxyJump MINI_LOGIN@ssh.mini.pw.edu.pl
   ```
2. `ssh eden`, potem `git clone <repo> RL2026 && cd RL2026 && bash cluster/eden/setup_env.sh`
   (na węźle logowania; **nie liczymy tam niczego**).
3. W `cluster/eden/train.sbatch` wpisz swoje `--account=` (nazwa grupy).
4. Szybki test na węźle: `srun -p student -A GROUP --cpus-per-task=2 --mem=4G --time=00:15:00 --pty bash`
   (w `tmux`), tam `pytest -q` i krótki `train.py`.
5. Joby: `bash cluster/eden/submit.sh tabular_q 4x4 "0 1 2 3 4"`; podgląd `squeue -u $USER`,
   logi w `slurm_logs/`.
6. Przerwany job (limit czasu): ten sam `submit.sh` ponownie - `--resume` dokończy od checkpointu,
   ukończone runy (`done.json`) są pomijane.
7. Na swoim komputerze: `bash cluster/eden/sync_results.sh` (tylko logi) lub `--all` (z modelami).


Jeśli repozytorium na Edenie jest w innym katalogu, ustaw `REMOTE_DIR` przy synchronizacji. Przed wysłaniem zadań przełącz się na gałąź zawierającą migrację.
