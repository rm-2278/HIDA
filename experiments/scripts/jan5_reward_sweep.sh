#!/bin/sh

# Sweep over different reward structures for pinpad-easy
# Tests: flat, progressive, progressive_steep, sequence_bonus, decaying, sparse
SWEEP_OUTPUT=$(wandb sweep experiments/pinpad-easy-reward-sweep.yml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent [^ ]*' | awk '{print $NF}')
# Print ID to debug
echo "Detected Sweep ID: $SWEEP_ID"
wandb agent $SWEEP_ID
