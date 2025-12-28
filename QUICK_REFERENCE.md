# Quick Reference - Reproducibility Fixes

## What Was Fixed

### 1. Replay Buffer Seeding ⭐ PRIMARY FIX
**The Issue:** Replay buffers were using a hardcoded seed (0), not the configured seed.
**The Fix:** Pass `config.seed` to all replay buffer constructors.
**Impact:** This was causing different episode sampling between runs, making training non-reproducible.

### 2. Environment Seeding ⭐ CRITICAL FIX  
**The Issue:** All 8 parallel environments were getting the same seed (42, 42, 42, ...).
**The Fix:** Each environment now gets a unique seed (42, 43, 44, 45, ...).
**Impact:** This was causing identical behavior across environments, reducing diversity.

### 3. PyTorch Determinism
**Added:** 
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `PYTHONHASHSEED` environment variable
- Enhanced CUDA determinism settings

## How to Test

### Simple Test (2 minutes)
```bash
# Run 1
python hieros/train.py --configs seed --wandb_prefix=test1 --wandb_name=seed --seed=42

# Run 2  
python hieros/train.py --configs seed --wandb_prefix=test2 --wandb_name=seed --seed=42

# Compare - should see identical metrics
diff logs/*/metrics.jsonl
```

### Metrics to Check
If reproducibility is working, these should be **exactly identical** between runs:
- `Subactor-0/model_loss`
- `episode/score`
- `train/reward`

## New Hyperparameters for pinpad-easy

### Use the New Config
```bash
python hieros/train.py --configs pinpad-easy-director --task=pinpad-easy_three
```

### What Changed
| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| train_ratio | 8 | 128 | 16x more learning per step (Director uses 128-512) |
| batch_length | 16 | 32 | Better temporal credit for sequences |
| batch_size | 8 | 16 | More stable gradients |
| imag_horizon | 16 | 32 | Better long-term planning |
| actor_entropy | 3e-4 | 1e-3 | More exploration for discrete actions |

### Why This Helps
- **Higher train_ratio:** Director's key insight - learn more from each experience
- **Longer sequences:** Critical for tasks requiring temporal understanding
- **More exploration:** Essential for discovering correct action sequences

## Expected Results

✅ **Before fixes:** Different metrics every run, even with same seed
✅ **After fixes:** Identical metrics across runs with same seed
✅ **With new hyperparameters:** Better performance on pinpad-easy tasks

## Questions?

See `REPRODUCIBILITY_FIXES.md` for detailed explanation and testing procedures.
