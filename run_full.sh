#!/usr/bin/env bash
# run_full.sh — the full run for the article, step by step with resume.
#
# Usage on macOS (Terminal, from the psm-benchmark directory):
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt doubleml econml
#   caffeinate -i bash run_full.sh          # caffeinate keeps the Mac awake; log: results/final/run_full.log
# A repeated run skips the steps already completed (results/final/done.log). JOBS=N sets the number of processes.
#
# Every step is a separate call of run_experiment.py; results are appended to results/final/runs.csv
# (upsert by key), so an interruption costs only the current step. The X4 steps (causal forest)
# run only if econml can be imported (it does not build on Linux/aarch64).
set -u
cd "$(dirname "$0")"
OUT=results/final; mkdir -p "$OUT"
LOG="$OUT/run_full.log"; DONE="$OUT/done.log"; touch "$DONE"
JOBS="${JOBS:-$( (nproc || sysctl -n hw.ncpu) 2>/dev/null | head -1)}"
PY="${PY:-python3}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -x ".venv/bin/python3" ]; then PY=".venv/bin/python3"; fi
HAS_X4=0; "$PY" -c "import econml" 2>/dev/null && HAS_X4=1
echo "python=$PY jobs=$JOBS econml=$HAS_X4" | tee -a "$LOG"

step () {  # step <name> <args...>
  local name="$1"; shift
  if grep -qx "$name" "$DONE"; then echo "[skip] $name" | tee -a "$LOG"; return; fi
  echo "[start] $(date '+%F %T') $name" | tee -a "$LOG"
  if "$PY" run_experiment.py "$@" --jobs "$JOBS" --out "$OUT" >> "$LOG" 2>&1; then
    echo "$name" >> "$DONE"; echo "[done]  $(date '+%F %T') $name" | tee -a "$LOG"
  else
    echo "[FAIL]  $(date '+%F %T') $name (see $LOG)" | tee -a "$LOG"
  fi
}

# 1. LaLonde-DW (PSID-1): core set, 10 seeds, B = 500 — main table T3
step lalonde_core_lr   --dataset lalonde_dw_psid --configs M0,M1,M2,M3,M4,M5,M6,M7,M8,M1+ --seeds 10 --bootstrap 500
step lalonde_core_gbm  --dataset lalonde_dw_psid --configs M9,M10,M11   --seeds 10 --bootstrap 500
step lalonde_core_rf   --dataset lalonde_dw_psid --configs M12,M13,M14  --seeds 10 --bootstrap 500
# 2. LaLonde-DW: extensions X1–X3 (+ X4 if econml is available)
step lalonde_ext       --dataset lalonde_dw_psid --configs X1,X2,X3     --seeds 10 --bootstrap 500
[ $HAS_X4 = 1 ] && step lalonde_x4 --dataset lalonde_dw_psid --configs X4 --seeds 10
# 3. IHDP 1–100: core set + X1–X3, in chunks of 10 replications, B = 200
for a in $(seq 1 10 91); do b=$((a+9))
  step "ihdp_core_${a}_${b}" --dataset ihdp --reps "${a}-${b}" --configs core --bootstrap 200
done
for a in $(seq 1 10 91); do b=$((a+9))
  step "ihdp_ext_${a}_${b}"  --dataset ihdp --reps "${a}-${b}" --configs X1,X2,X3 --bootstrap 200
done
if [ $HAS_X4 = 1 ]; then for a in $(seq 1 10 91); do b=$((a+9))
  step "ihdp_x4_${a}_${b}"   --dataset ihdp --reps "${a}-${b}" --configs X4
done; fi
# 4. Sensitivity grid (F5) — LaLonde-DW, 10 seeds, B = 200
step lalonde_sens      --dataset lalonde_dw_psid --configs sensitivity --seeds 10 --bootstrap 200
# 5. CPS-1 (sensitivity to the control sample): LR configurations 5 seeds B = 200; ML — 3 seeds B = 100
step cps_core_lr       --dataset lalonde_dw_cps --configs M0,M1,M2,M3,M4,M5,M6,M7,M8,M1+ --seeds 5 --bootstrap 200
step cps_core_ml       --dataset lalonde_dw_cps --configs M9,M10,M11,M12,M13,M14 --seeds 3 --bootstrap 100
# 6. Afterwards: tables and figures
"$PY" make_tables.py  --results "$OUT" >> "$LOG" 2>&1
"$PY" make_figures.py --results "$OUT" --lang en >> "$LOG" 2>&1

echo "[all done] $(date '+%F %T')" | tee -a "$LOG"
