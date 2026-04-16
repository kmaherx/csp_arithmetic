#!/bin/bash
# Train sp_pos CSPs for a list of personas (passed as args), serially.
# Used to populate the CSP embedding set for the PCA / assistant-axis analysis.
#
# Usage:
#   bash run_axis_sweep.sh wizard samurai vampire ...
#
# Each persona gets:
#   - its own teacher response cache (results/<persona>/cached_responses.json)
#   - an sp_pos.pt checkpoint at results/<persona>/sp_pos.pt
#   - a training log at results/<persona>_pos.log
# No neg-polarity training; no eval. Embeddings only — we just need the
# checkpoints for PCA / geometric analysis.

set -e

cd "$(dirname "$0")"
export HF_TOKEN=$(cat /workspace/.hf_token)
export HF_HOME=/workspace/.cache/huggingface/

LOG_DIR=results
mkdir -p "$LOG_DIR"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <persona1> <persona2> ..."
    exit 1
fi

SWEEP_LOG="$LOG_DIR/axis_sweep.log"
echo ">>> [$(date +%H:%M:%S)] AXIS SWEEP START (${#} personas)" | tee -a "$SWEEP_LOG"
echo "    personas: $*" | tee -a "$SWEEP_LOG"

for persona in "$@"; do
    logf="$LOG_DIR/${persona}_pos.log"
    echo ">>> [$(date +%H:%M:%S)] START $persona" | tee -a "$SWEEP_LOG"
    if .venv/bin/python -u train.py --persona "$persona" --polarity pos >> "$logf" 2>&1; then
        echo "<<< [$(date +%H:%M:%S)] DONE  $persona" | tee -a "$SWEEP_LOG"
    else
        echo "!!! [$(date +%H:%M:%S)] FAIL  $persona (see $logf)" | tee -a "$SWEEP_LOG"
    fi
done

echo ">>> [$(date +%H:%M:%S)] AXIS SWEEP COMPLETE" | tee -a "$SWEEP_LOG"
