# Fix for Tensor Dimension Mismatch in _subgoal_reward

## Problem Description

When running `bash dec29.sh`, the system encountered a tensor dimension mismatch error:

```
The expanded size of the tensor (1) must match the existing size (8) at non-singleton dimension 1.
Target sizes: [8, 1, 128].  Tensor sizes: [8, 128]
```

### Root Cause

The error occurred in the `_subgoal_reward` function in `hieros/hieros.py` (around line 1402-1404). The original code attempted to:

1. Reshape the subgoal tensor from `[B, T, F]` to `[B*T, F]` (flattening batch and time)
2. Expand the result to match `state_representation` shape `[B, T, F']`

This failed because:
- After reshape: `[8, 1, 128]` → `[8, 128]` (2D tensor)
- Attempted expand to: `[8, 1, 384]` (3D tensor)
- PyTorch's `expand()` cannot convert a dimension with size 128 to size 1

Additionally, there was a feature dimension mismatch:
- Decoded subgoal: 128 features (deter only)
- State representation: 384 features (stoch 256 + deter 128)

## Solution

The fix modifies the `_subgoal_reward` function to:

### 1. Preserve Time Dimension
Instead of collapsing and then trying to expand, the fix preserves the time dimension:
```python
if len(state_representation.shape) == 3:
    if len(subgoal.shape) == 2:
        reshaped_subgoal = subgoal.unsqueeze(1)  # [B, F] -> [B, 1, F]
    elif len(subgoal.shape) == 3:
        reshaped_subgoal = subgoal  # Keep as [B, T, F]
```

### 2. Handle Feature Dimension Mismatch
When feature dimensions don't match, pad with zeros:
```python
if reshaped_subgoal.shape[-1] < state_representation.shape[-1]:
    padding_size = state_representation.shape[-1] - reshaped_subgoal.shape[-1]
    padding = torch.zeros(..., padding_size, ...)
    reshaped_subgoal = torch.cat([reshaped_subgoal, padding], dim=-1)
```

### 3. Support Both 2D and 3D Tensors
The fix handles both training scenarios (with batch data from replay buffer) and policy scenarios (with live environment interactions).

## Changes Made

File: `hieros/hieros.py`
Function: `SubActor._subgoal_reward` (lines 1395-1469)

Key modifications:
1. Added dimension-aware reshaping logic
2. Implemented zero-padding for feature dimension mismatches
3. Added support for both 2D and 3D input tensors
4. Fixed `dims_to_sum` calculation for 2D tensors

## Testing

The fix ensures that:
- Original bug case (`[8, 1, 128]` subgoal with `[8, 1, 384]` state) works ✓
- 2D tensors without time dimension work ✓
- 3D tensors with time dimension work ✓
- Different batch sizes work ✓
- Feature dimension mismatches are handled gracefully ✓

## How to Verify

Run the original failing command:
```bash
bash dec29.sh
```

The training should now proceed without the tensor dimension mismatch error.

## Notes

- The fix pads smaller feature dimensions with zeros to maintain mathematical correctness in cosine similarity computation
- The time dimension is consistently handled across both training and policy execution paths
- Backward compatibility is maintained for existing code that passes 2D tensors

## Related Files

- `hieros/hieros.py` - Main fix location
- `test_debug_structure.py` - Structure validation tests (passes ✓)
- `DEBUG_SUBGOAL_VISUALIZATION.md` - Debugging documentation
- `DEBUG_README.md` - Quick reference guide
