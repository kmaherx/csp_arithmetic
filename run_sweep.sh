#!/bin/bash
# Train remaining CSPs and run eval for all personas. Sequential.
# Logs each run separately under results/<persona>_<polarity>.log
set -e

cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.hf_token)
export HF_HOME=/workspace/.cache/huggingface/

LOG_DIR=results
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

# pirate pos already done; pirate neg uses cached teacher responses
run pirate_neg train.py --persona pirate --polarity neg

# poet (teacher gen happens on first run, cached for second)
run poet_pos train.py --persona poet --polarity pos
run poet_neg train.py --persona poet --polarity neg

# prophet
run prophet_pos train.py --persona prophet --polarity pos
run prophet_neg train.py --persona prophet --polarity neg

# Evaluation per persona
for persona in pirate poet prophet; do
    run "eval_${persona}" evaluate.py --persona "$persona"
done

echo ">>> [$(date +%H:%M:%S)] SWEEP COMPLETE" | tee -a "$LOG_DIR/sweep.log"
