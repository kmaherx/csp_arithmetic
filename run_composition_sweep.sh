#!/bin/bash
# Sweep composition eval over all 3 persona pairs.
# Logs each run separately under results/composition/<pair>.log
set -e

cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.hf_token)
export HF_HOME=/workspace/.cache/huggingface/

LOG_DIR=results/composition
mkdir -p "$LOG_DIR"

run() {
    local label="$1"; shift
    local logf="$LOG_DIR/${label}.log"
    echo ">>> [$(date +%H:%M:%S)] START $label" | tee -a "$LOG_DIR/sweep.log"
    if .venv/bin/python -u "$@" >> "$logf" 2>&1; then
        echo "<<< [$(date +%H:%M:%S)] DONE  $label" | tee -a "$LOG_DIR/sweep.log"
    else
        echo "!!! [$(date +%H:%M:%S)] FAIL  $label (see $logf)" | tee -a "$LOG_DIR/sweep.log"
        exit 1
    fi
}

for pair in pirate+poet pirate+prophet poet+prophet; do
    run "$pair" run_composition.py --pair "$pair"
done

echo ">>> [$(date +%H:%M:%S)] COMPOSITION SWEEP COMPLETE" | tee -a "$LOG_DIR/sweep.log"
