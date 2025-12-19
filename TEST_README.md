# Test Files for Evaluation Logging Bug Fix

This directory contains test files that demonstrate and verify the fix for the evaluation logging bug.

## The Bug

Evaluation metrics were being logged to JSONL files and W&B at a later step than when the evaluation actually occurred. This made it appear that evaluation happened later than it did, causing confusion in training analysis.

## Test Files

### 1. `test_eval_timing_simple.py` - Bug Demonstration

**Purpose:** Clearly demonstrates the timing bug and how the fix resolves it.

**Run:**
```bash
python test_eval_timing_simple.py
```

**What it shows:**
- **Buggy behavior:** Evaluation at step 30 logged at step 50 (20-step delay)
- **Fixed behavior:** Evaluation at step 30 logged at step 30 (correct timing)

**Key Output:**
```
🐛 BUG: Eval score logged at step 50,
        but evaluation actually occurred at step 30!
        Delay: 20 steps

✓ FIXED: Eval score logged at step 30,
         exactly when evaluation occurred!
```

### 2. `test_no_duplicates.py` - Duplicate Check

**Purpose:** Verifies that the fix doesn't cause metrics to be logged twice.

**Run:**
```bash
python test_no_duplicates.py
```

**What it shows:**
- Eval metrics are logged exactly once (at step 30)
- Training metrics are logged separately (at step 50)
- No duplicate or mixed metrics in any log entry

**Key Output:**
```
✓ PASS: Eval metrics appear exactly once
  - Logged at step: 30
  - Expected step: 30
  - ✓ Correct step!
```

### 3. `test_eval_logging_bug.py` - Full Integration Test

**Purpose:** More comprehensive test with mock environments (requires dependencies).

**Note:** This test requires the full embodied library and dependencies. Use the simpler tests above for quick verification.

## Running Tests

**Quick verification (no dependencies required):**
```bash
python test_eval_timing_simple.py
python test_no_duplicates.py
```

**Expected output:**
- Both tests should pass with ✓ marks
- The timing test shows the bug and the fix clearly
- The duplicates test confirms metrics are logged correctly once

## The Fix

The fix is in `embodied/run/train_eval.py`, lines 167-168:

```python
if should_eval(step):
    driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
    # Log evaluation metrics immediately at the correct step
    logger.add(metrics.result())
    logger.write()
```

These 2 lines ensure that:
1. Eval metrics are added to the logger at the current (correct) step
2. They are written immediately to all outputs (JSONL, W&B, TensorBoard)
3. The metrics are cleared (via `result(reset=True)`) so they don't appear again

## Documentation

- `SUMMARY.md` - Comprehensive answers to the original questions
- `EVAL_LOGGING_BUG_FIX.md` - Detailed technical analysis

## Results

✅ Bug identified and documented  
✅ Minimal fix implemented (2 lines)  
✅ Tests demonstrate bug and verify fix  
✅ No duplicate metrics  
✅ No security vulnerabilities  
✅ Code review passed  

The fix ensures that both JSONL and W&B logs show evaluation metrics at the correct step, making training analysis accurate and reliable.
