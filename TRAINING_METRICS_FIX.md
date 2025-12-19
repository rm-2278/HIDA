# Training Metrics Recording Fix

## Summary

This document describes the fixes applied to resolve issues with irregular metric recording, missing score data, and missing evaluation statistics during training.

## Issues Fixed

### 1. Irregular Metric Recording

**Problem**: Metrics were being recorded at irregular steps (e.g., 4, 220, 308, 396) instead of at regular intervals.

**Root Cause**: The `should_log` scheduler was using `embodied.when.Clock`, which is time-based (wall-clock seconds) rather than step-based. This caused logging to occur based on elapsed time, leading to irregular step intervals depending on training speed.

**Solution**: Changed `should_log` from `embodied.when.Clock(args.log_every)` to `embodied.when.Every(args.log_every)` across all training implementations.

**Expected Behavior**: With `--log_every=100` and `--action_repeat=4`, logging now occurs at regular step intervals: 0, 100, 200, 300, 400 (in environment steps) or 0, 25, 50, 75, 100 (in training steps after action_repeat division).

### 2. Missing Score Data

**Problem**: No scores were being recorded in `scores.jsonl`.

**Root Cause**: The `scores.jsonl` logger was configured with pattern `"episode/score"`, which only captured training episode scores. Evaluation episodes log with prefix `"eval_episode/score"`, which wasn't captured. In short training runs where training episodes might not complete, but evaluation episodes do, no scores would be recorded.

**Solution**: Changed the pattern from `"episode/score"` to `r".*episode/score"` to capture both:
- Training episode scores: `episode/score`
- Evaluation episode scores: `eval_episode/score`

**Expected Behavior**: `scores.jsonl` now records scores from both training and evaluation episodes whenever they complete.

### 3. Missing Evaluation Statistics

**Problem**: No evaluation statistics were being recorded.

**Root Cause**: The training loop was running 100 steps per iteration (hardcoded). With `--eval_every=100` (which becomes 25 steps after action_repeat division), the loop would skip evaluation checkpoints. For example, starting at step 0, training 100 steps would reach step 100 and exit before checking should_eval at steps 25, 50, or 75.

**Solution**: Made the training loop chunk size adaptive based on `eval_every`:
```python
chunk_size = min(100, max(1, args.eval_every)) if args.eval_every > 0 else 100
```

**Expected Behavior**: 
- With `--eval_every=100 --action_repeat=4`, eval_every becomes 25 steps
- Chunk size becomes min(100, 25) = 25
- Training loop checks should_eval at steps 0, 25, 50, 75, 100
- Evaluation runs at steps 25, 50, 75, 100 (since eval_initial=False by default)
- Evaluation metrics are properly logged to `metrics.jsonl`

## Files Modified

1. `embodied/run/train_eval.py` - Main fix for the reported issue
2. `embodied/run/train.py` - Basic training loop
3. `embodied/run/eval_only.py` - Evaluation-only mode
4. `embodied/run/train_holdout.py` - Training with holdout set
5. `embodied/run/train_save.py` - Training with model saving
6. `embodied/run/parallel.py` - Parallel training (actor and learner)
7. `hieros/train.py` - Logger configuration for score pattern

## Testing

To verify these fixes work correctly with the original command:

```bash
python hieros/train.py --configs atari100k --task=atari_pong \
  --steps=400 --eval_every=100 --eval_eps=1 \
  --batch_size=4 --batch_length=32 --log_every=100
```

Expected results:
- **Regular logging**: Metrics recorded at steps 0, 100, 200, 300, 400 (env steps)
- **Score recording**: Episode scores appear in `scores.jsonl` with keys like `episode/score` and `eval_episode/score`
- **Evaluation statistics**: Evaluation runs at steps 100, 200, 300, 400 (env steps) and evaluation metrics appear in `metrics.jsonl` with keys like `eval_episode/avg_score`, `eval_episode/length`, etc.

## Configuration Notes

### Action Repeat
Most Atari configs use `action_repeat=4`, which means:
- `--steps=400` becomes 100 training steps
- `--eval_every=100` becomes 25 training steps  
- `--log_every=100` becomes 25 training steps

This division happens in `hieros/train.py` lines 78-81.

### Evaluation Initial
By default, `eval_initial=False`, which means evaluation does NOT run at step 0. To enable initial evaluation, add `--eval_initial=True` to the command.

### Logger Outputs
The logger creates three main output files:
- `metrics.jsonl` - All scalar metrics (pattern: `.*`)
- `scores.jsonl` - Episode scores only (pattern: `.*episode/score`)
- Terminal output - Filtered metrics for display

## Additional Notes

The `should_save` scheduler remains using `Clock` (time-based) as model checkpointing is typically better suited to wall-clock time rather than training steps, to avoid excessive saves during fast training or insufficient saves during slow training.
