#!/bin/sh

# Sweep over sparse reward structure for pinpad-easy
# Tests: sparse reward with high exploration parameters
SWEEP_OUTPUT=$(wandb sweep experiments/pinpad-easy-sparse-sweep.yml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent [^ ]*' | awk '{print $NF}')
# Print ID to debug
echo "Detected Sweep ID: $SWEEP_ID"
wandb agent $SWEEP_ID
