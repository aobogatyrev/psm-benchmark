#!/usr/bin/env bash
# X4 (causal forest) — run separately where econml is available (it does not build on an aarch64 Linux VM)
set -u; cd "$(dirname "$0")"; OUT=results/final_x4; mkdir -p $OUT; LOG=$OUT/run_x4.log
python3 run_experiment.py --dataset lalonde_dw_psid --configs X4 --seeds 10 --jobs 2 --out $OUT >> $LOG 2>&1
for a in $(seq 1 10 91); do b=$((a+9)); python3 run_experiment.py --dataset ihdp --reps "${a}-${b}" --configs X4 --jobs 2 --out $OUT >> $LOG 2>&1; done
python3 run_experiment.py --dataset lalonde_dw_cps --configs X4 --seeds 5 --jobs 2 --out $OUT >> $LOG 2>&1
echo "[x4 done] $(date)" >> $LOG
