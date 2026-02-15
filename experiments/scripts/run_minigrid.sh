#!/bin/bash
# Minigrid 400k step experiments
# Launches wandb sweep for MiniGrid environments

SWEEP_OUTPUT=$(wandb sweep experiments/configs/minigrid-400k.yml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent [^ ]*' | awk '{print $NF}')
echo "Detected Sweep ID: $SWEEP_ID"
wandb agent $SWEEP_ID
