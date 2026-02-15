#!/bin/sh

# Test hierarchy levels 1 and 2 for pinpad-easy configuration
# Tests both flat (hierarchy=1) and 2-level hierarchical approaches
SWEEP_OUTPUT=$(wandb sweep experiments/configs/pinpad-easy-hierarchy-test.yml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent [^ ]*' | awk '{print $NF}')
# Print ID to debug
echo "Detected Sweep ID: $SWEEP_ID"
wandb agent $SWEEP_ID