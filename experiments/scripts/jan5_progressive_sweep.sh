#!/bin/sh

# Sweep over progressive reward structures for pinpad-easy
# Tests: progressive and progressive_steep with various hyperparameters
SWEEP_OUTPUT=$(wandb sweep experiments/pinpad-easy-progressive-sweep.yml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent [^ ]*' | awk '{print $NF}')
# Print ID to debug
echo "Detected Sweep ID: $SWEEP_ID"
wandb agent $SWEEP_ID
