# Debug Tools for Subgoal Visualization

This directory contains debugging tools to help diagnose tensor dimension mismatch errors in the Hieros subgoal visualization feature.

## Problem

When using `subgoal_debug_visualization: True`, users may encounter errors like:

```
Exception in hieros policy: The expanded size of the tensor (128) must match the existing size (8) 
at non-singleton dimension 2. Target sizes: [8, 1, 128]. Tensor sizes: [64, 8, 8]
```

## Solution

We've added comprehensive debugging functionality to help diagnose and fix these issues.

## Files Added

1. **`hieros/hieros.py`** - Modified with:
   - `debug_subgoal_visualization_shapes()` function for detailed shape logging
   - Integration with `config.debug` flag
   - Enhanced error messages for dimension mismatches

2. **`DEBUG_SUBGOAL_VISUALIZATION.md`** - Comprehensive documentation covering:
   - Understanding the error
   - How to enable debug logging
   - Explanation of tensor shapes
   - Common issues and solutions
   - Debugging workflow

3. **`test_debug_structure.py`** - Structure validation tests:
   - Validates debug function exists
   - Checks integration is correct
   - Verifies enhanced error handling
   - No dependencies required

4. **`test_subgoal_debug.py`** - Functional tests (requires torch):
   - Tests with correct shapes
   - Tests with incorrect batch sizes
   - Tests with incorrect time dimensions
   - Tests logging can be disabled

5. **`examples_debug_usage.py`** - Usage examples showing:
   - Basic usage
   - Conditional debugging
   - Error detection
   - Integration patterns

## Quick Start

### Enable Debug Mode

In your config file (e.g., `hieros/configs.yaml`):

```yaml
debug: True
subgoal_debug_visualization: True
```

### Run Your Code

When the code runs, you'll see detailed shape information printed for each subactor:

```
================================================================================
DEBUG: Subgoal Visualization Shapes for Subactor 0
================================================================================
Cached subgoal shape: [4, 8, 8]
Decoded subgoal shape: [4, 1280]
Subgoal with time shape: [4, 1, 1280]
...
✅ All shapes are valid!
================================================================================
```

### Run Tests

```bash
# Structure tests (no dependencies)
python tests/test_debug_structure.py

# Functional tests (requires torch)
python tests/test_subgoal_debug.py

# Usage examples (requires torch)
python docs/examples_debug_usage.py
```

## Key Features

✅ **Detailed Shape Logging** - See exact shapes of all tensors involved
✅ **Automatic Validation** - Detects common shape mismatches
✅ **Helpful Error Messages** - Get actionable suggestions for fixing issues
✅ **Zero Performance Impact** - Only runs when `debug=True`
✅ **Easy Integration** - Just set `debug: True` in config

## Documentation

See `DEBUG_SUBGOAL_VISUALIZATION.md` for complete documentation including:
- Detailed explanation of the bug and fix
- Understanding tensor shapes
- Common issues and solutions
- Step-by-step debugging workflow

## Testing

All tests pass successfully:

```bash
$ python tests/test_debug_structure.py
...
🎉 All tests passed!
Total: 4/4 tests passed
```

## Support

If you continue to experience issues:

1. Enable `debug: True` in your config
2. Check the detailed shape output
3. Compare with expected shapes in the documentation
4. Look for batch size mismatches
5. Verify `subgoal_shape` in config matches your model

## Contributing

If you find other shape-related issues or have suggestions for improving the debug functionality, please open an issue or PR.

---

Created as part of the fix for: "Bug in subgoal_visualization_debug"
