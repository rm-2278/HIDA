# Layer-Specific Exploration Parameters

## Overview

This feature allows configuring different exploration tendency (entropy) values for each layer in the hierarchical reinforcement learning setup. Each subactor in the hierarchy can now have its own exploration parameters.

## Motivation

In hierarchical RL, different layers may benefit from different levels of exploration:
- **Lower layers** (closer to environment): Often benefit from lower exploration for more stable, deterministic actions
- **Higher layers** (abstract goals): May benefit from higher exploration to discover diverse subgoals

## Supported Parameters

The following exploration parameters now support per-layer configuration:
- `actor_entropy`: Policy entropy coefficient (main exploration parameter)
- `actor_state_entropy`: State entropy coefficient

## Configuration Format

### Single Value (Backward Compatible)
All layers use the same value:
```yaml
actor_entropy: '3e-4'
actor_state_entropy: 0.0
```

### Per-Layer Values
Specify a list with one value per layer:
```yaml
max_hierarchy: 3
actor_entropy: ['3e-4', '1e-3', '5e-3']
actor_state_entropy: [0.0, 0.1, 0.2]
```

This configures:
- Layer 0 (Subactor-0): `actor_entropy=3e-4`, `actor_state_entropy=0.0`
- Layer 1 (Subactor-1): `actor_entropy=1e-3`, `actor_state_entropy=0.1`
- Layer 2 (Subactor-2): `actor_entropy=5e-3`, `actor_state_entropy=0.2`

### Shorter Lists
If the list is shorter than `max_hierarchy`, the last value is reused:
```yaml
max_hierarchy: 3
actor_entropy: ['3e-4', '1e-3']
```

Results in:
- Layer 0: `3e-4`
- Layer 1: `1e-3`
- Layer 2: `1e-3` (reuses last value)

## Usage Examples

### Example 1: Increasing Exploration with Abstraction
Lower layers are more deterministic, higher layers explore more:
```yaml
max_hierarchy: 3
actor_entropy: ['3e-4', '1e-3', '3e-3']
```

### Example 2: Decreasing Exploration with Abstraction
Lower layers explore more, higher layers are more deterministic:
```yaml
max_hierarchy: 3
actor_entropy: ['1e-3', '5e-4', '3e-4']
```

### Example 3: Mixed Configuration
Single value for one parameter, per-layer for another:
```yaml
max_hierarchy: 3
actor_entropy: ['3e-4', '1e-3', '5e-3']  # Per-layer
actor_state_entropy: 0.0                  # Single value for all layers
```

## Command Line Usage

When using command line arguments:
```bash
python hieros/train.py \
  --configs pinpad-easy \
  --max_hierarchy 3 \
  --actor_entropy "['3e-4', '1e-3', '5e-3']"
```

## Implementation Details

### Code Changes
- Added `layer_idx` parameter to `SubActor.__init__()` (default: 0)
- Modified entropy handling to detect list/tuple and select appropriate value
- Updated subactor creation to pass layer index
- Minimal changes to existing code (~20 lines modified)

### Backward Compatibility
- ✅ Existing single-value configurations work unchanged
- ✅ No breaking changes to existing experiments
- ✅ All existing tests pass

## Testing

Run the included tests to verify functionality:
```bash
# Structure tests
python test_layer_entropy.py

# Configuration handling tests
python test_entropy_config.py
```

## Example Experiment

See `experiments/example_layer_entropy.yml` for a complete example configuration.

## Notes

- Values are processed during `SubActor` initialization
- Each layer gets its own entropy lambda function
- The `layer_idx` is assigned sequentially (0, 1, 2, ...)
- List indices match layer indices exactly
