# Debugging Subgoal Visualization in Hieros

## Overview

This document explains how to use the debugging functionality added to help diagnose tensor shape mismatch errors in the `subgoal_debug_visualization` feature of Hieros.

## The Problem

When using `subgoal_debug_visualization: True` in the config, you may encounter an error like:

```
Exception in hieros policy: The expanded size of the tensor (128) must match the existing size (8) 
at non-singleton dimension 2. Target sizes: [8, 1, 128]. Tensor sizes: [64, 8, 8]
```

This error occurs due to tensor dimension mismatches in the subgoal reward computation code.

## The Solution

### 1. Debug Function

A new debug function `debug_subgoal_visualization_shapes()` has been added to `hieros/hieros.py`. This function:

- **Validates tensor shapes** at each step of the subgoal reward computation
- **Logs detailed shape information** for all tensors involved
- **Detects common errors** like batch size mismatches or incorrect time dimensions
- **Provides actionable error messages** to help identify the root cause

### 2. How to Enable Debug Logging

To enable detailed shape logging, set `debug: True` in your config file:

```yaml
# In your config (e.g., hieros/configs.yaml)
debug: True
subgoal_debug_visualization: True
```

When enabled, the debug function will print detailed information like:

```
================================================================================
DEBUG: Subgoal Visualization Shapes for Subactor 0
================================================================================
Cached subgoal shape: [4, 8, 8]
  Expected: [batch_size, subgoal_dim_1, subgoal_dim_2]
  Example: [4, 8, 8] or [64, 8, 8]

Decoded subgoal shape: [4, 1280]
  Expected: [batch_size, decoded_features]
  Example: [4, 1280] or [64, 1280]

Subgoal with time shape: [4, 1, 1280]
  Expected: [batch_size, 1, decoded_features]
  Example: [4, 1, 1280] or [64, 1, 1280]

Subactor state keys: ['deter', 'stoch']
Subactor state shapes:
  deter: [4, 256]
  stoch: [4, 1024]

State with time shapes:
  deter: [4, 1, 256]
  stoch: [4, 1, 1024]

Batch size: 4

✅ All shapes are valid!
================================================================================
```

### 3. Enhanced Error Messages

When a dimension mismatch error occurs, you'll now see an enhanced error message:

```
================================================================================
ERROR: Tensor dimension mismatch detected!
================================================================================
Error message: <original error>

This error often occurs when tensor shapes don't match.
To debug, enable debug mode in config: debug: True
This will print detailed shape information for all tensors.

Common causes:
  1. Mismatch between cached_subgoal batch size and state batch size
  2. Incorrect subgoal_shape in config
  3. decode_subgoal function not handling shapes correctly
================================================================================
```

## Understanding the Shapes

### Key Tensor Shapes

1. **cached_subgoal**: `[batch_size, subgoal_dim_1, subgoal_dim_2]`
   - Example: `[4, 8, 8]` for 4 environments with 8x8 subgoal shape
   - This comes from the subgoal cache after padding

2. **decoded_subgoal**: `[batch_size, decoded_features]`
   - Example: `[4, 1280]` 
   - Result of passing cached_subgoal through `decode_subgoal()`
   - decoded_features = dyn_deter + dyn_stoch*dyn_discrete (e.g., 256 + 32*32 = 1280)

3. **subgoal_with_time**: `[batch_size, 1, decoded_features]`
   - Example: `[4, 1, 1280]`
   - Result of `decoded_subgoal.unsqueeze(1)` to add time dimension

4. **state_with_time**: dict with tensors of shape `[batch_size, 1, feature_dim]`
   - Example: `{'deter': [4, 1, 256], 'stoch': [4, 1, 1024]}`
   - Each state tensor has time dimension added with `unsqueeze(1)`

### The Original Bug

The original code tried to use:
```python
batch_size = next(iter(subactor_state[0].values())).shape[0]
subgoal_with_batch = cached_subgoal.unsqueeze(0).expand(batch_size, *cached_subgoal.shape)
```

This was incorrect because:
1. `cached_subgoal` already has a batch dimension (first dimension)
2. `unsqueeze(0)` adds a new dimension at position 0: `[1, batch, 8, 8]`
3. `expand(batch_size, *cached_subgoal.shape)` tries to expand to `[batch, batch, 8, 8]` which is wrong
4. This caused the dimension mismatch error

### The Fix

The current code correctly uses:
```python
decoded_subgoal = subactor.decode_subgoal(cached_subgoal, isfirst=False)
subgoal_with_time = decoded_subgoal.unsqueeze(1)
```

This works because:
1. `cached_subgoal` is passed directly to `decode_subgoal` (it already has correct batch dimension)
2. `decode_subgoal` returns `[batch, features]`
3. `unsqueeze(1)` adds time dimension to get `[batch, 1, features]`
4. This matches the shape of `state_with_time`

## Common Issues and Solutions

### Issue 1: Batch Size Mismatch

**Symptoms:**
```
❌ cached_subgoal batch size 64 doesn't match state batch size 4
```

**Cause:** The cached_subgoal has a different batch size than the current state.

**Solution:** This could indicate:
- Config mismatch between when the cache was filled and current execution
- Different number of environments (`envs.amount`) between runs
- Cache was filled with a different configuration

**Fix:** Clear the cache or ensure consistent config:
```python
# Reset cache if needed
self._subgoal_cache_idx = 0
```

### Issue 2: Wrong Subgoal Shape

**Symptoms:**
```
❌ cached_subgoal should have 3 dimensions, got 4
```

**Cause:** The subgoal_shape in config doesn't match what was actually cached.

**Solution:** Check your config:
```yaml
subgoal_shape: [8, 8]  # Should match what's expected
```

### Issue 3: Time Dimension Error

**Symptoms:**
```
❌ subgoal_with_time should have time dimension of size 1, got 2
```

**Cause:** The time dimension was added incorrectly.

**Solution:** Ensure you're using `unsqueeze(1)` not `repeat()` or other operations:
```python
subgoal_with_time = decoded_subgoal.unsqueeze(1)  # Correct
# Not: decoded_subgoal.unsqueeze(1).repeat(1, 2, 1)  # Wrong!
```

## Testing

Two test scripts are provided:

### 1. Structure Test (No Dependencies)

```bash
python tests/test_debug_structure.py
```

This validates:
- The debug function exists and has correct structure
- The function is properly integrated into the code
- Enhanced error handling is present
- The Python syntax is valid

### 2. Functional Test (Requires torch)

```bash
python tests/test_subgoal_debug.py
```

This tests:
- Correct tensor shapes (should pass validation)
- Incorrect batch sizes (should detect errors)
- Incorrect time dimensions (should detect errors)
- Logging can be disabled

## Debugging Workflow

1. **Enable debug mode:**
   ```yaml
   debug: True
   subgoal_debug_visualization: True
   ```

2. **Run your training:**
   ```bash
   python hieros/train.py --configs hieros/configs.yaml ...
   ```

3. **Check the output for shape information:**
   - If shapes are valid, you'll see `✅ All shapes are valid!`
   - If shapes are invalid, you'll see detailed error messages

4. **Common fixes:**
   - Adjust `subgoal_shape` in config to match your model
   - Ensure `envs.amount` is consistent
   - Check `dyn_deter` and `dyn_stoch` config values

5. **If the problem persists:**
   - Check the decode_subgoal implementation
   - Verify subgoal_autoencoder architecture
   - Ensure the world model is producing expected state shapes

## Additional Resources

- **Error message history:** Check `self._metrics["exception_count"]` to see how often errors occur
- **Manual debugging:** Add your own print statements around line 530 in hieros/hieros.py
- **Config validation:** Use the debug config preset: `--configs hieros/configs.yaml debug`

## Summary

The debugging functionality helps you:
- ✅ Quickly identify tensor shape mismatches
- ✅ Understand what shapes are expected vs. actual
- ✅ Get actionable suggestions for fixing issues
- ✅ Validate that your fix is working correctly

Enable `debug: True` in your config whenever you use `subgoal_debug_visualization: True` to get the most helpful diagnostic information.
