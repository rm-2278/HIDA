"""
Test to demonstrate the evaluation logging bug.

The bug: Evaluation metrics are logged at a later step than when the evaluation
actually occurred. This happens because logger.write() is not called immediately
after evaluation completes.
"""

import json
import tempfile
from pathlib import Path
import sys
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

import embodied


class MockAgent:
    """Mock agent for testing."""
    
    def policy(self, obs, state=None, mode='train'):
        # Return random actions
        return {'action': np.array([0])}, state
    
    def train(self, batch, state):
        return {}, state, {}
    
    def report(self, batch):
        return {}
    
    def sync(self):
        pass
    
    def save(self):
        pass
    
    def dataset(self, generator):
        # Simple generator that yields dummy batches
        while True:
            yield {'dummy': np.array([1.0])}


class MockEnv:
    """Mock environment for testing."""
    
    def __init__(self, num_episodes=2):
        self.num_episodes = num_episodes
        self.episode_count = 0
        self.step_count = 0
        
    @property
    def obs_space(self):
        return {'observation': embodied.Space(np.float32, (4,))}
    
    @property
    def act_space(self):
        return {'action': embodied.Space(np.int32, (), 0, 2)}
    
    def __len__(self):
        return 1
    
    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 10  # End episode after 10 steps
        
        obs = {'observation': np.random.randn(4).astype(np.float32)}
        reward = 1.0 if not done else 10.0  # Bonus reward at end
        
        if done:
            self.episode_count += 1
            self.step_count = 0
            obs['is_last'] = True
            obs['is_terminal'] = True
        else:
            obs['is_last'] = False
            obs['is_terminal'] = False
            
        obs['reward'] = reward
        obs['action'] = action.get('action', np.array([0]))
        
        return obs


class MockReplay:
    """Mock replay buffer for testing."""
    
    def __init__(self):
        self._data = []
        self.stats = {}
        
    def add(self, transition, worker=0):
        self._data.append(transition)
        
    def __len__(self):
        return len(self._data)
    
    @property
    def dataset(self):
        while True:
            if self._data:
                yield {'dummy': np.array([1.0])}


def test_eval_logging_timing():
    """
    Test that demonstrates the evaluation logging bug.
    
    The test sets up a minimal training loop and checks when eval metrics
    are written to the JSONL file. The bug is that eval metrics appear at
    a later step than when evaluation actually occurred.
    """
    print("\n" + "="*70)
    print("TEST: Evaluation Logging Timing Bug")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logdir = Path(tmpdir) / "logs"
        logdir.mkdir(parents=True)
        
        # Set up minimal configuration
        class Args:
            logdir = str(logdir)
            expl_until = 0  # No exploration
            train_ratio = 0.5
            batch_steps = 8
            log_every = 50  # Log every 50 steps
            save_every = 1000000  # Never save
            eval_every = 30  # Evaluate every 30 steps
            eval_initial = False  # Don't evaluate at start
            eval_eps = 1  # One eval episode
            sync_every = 100
            train_fill = 10
            eval_fill = 10
            log_keys_video = []
            log_keys_sum = r'^$'
            log_keys_mean = r'^$'
            log_keys_max = r'^$'
            log_zeros = True
            steps = 100  # Run for 100 steps
            from_checkpoint = None
            store_checkpoints = False
            
        args = Args()
        
        # Create logger with JSONL output
        step = embodied.Counter()
        outputs = [
            embodied.logger.JSONLOutput(logdir, filename='metrics.jsonl'),
        ]
        logger = embodied.Logger(step, outputs)
        
        # Create mock components
        agent = MockAgent()
        train_env = MockEnv()
        eval_env = MockEnv()
        train_replay = MockReplay()
        eval_replay = MockReplay()
        
        # Import and patch the train_eval function to track when eval happens
        from embodied.run import train_eval
        
        # Track evaluation steps
        eval_steps = []
        original_driver_eval_call = None
        
        # We'll manually trace through what should happen:
        # 1. Training runs until step 30
        # 2. Evaluation triggers at step 30
        # 3. Evaluation completes (still at step ~30-40 range)
        # 4. Training continues, step advances
        # 5. At step 50, should_log triggers and metrics are written
        # 6. Eval metrics from step 30 are written with step ~50-60
        
        print("\nRunning minimal training loop...")
        print(f"Eval will trigger at step: {args.eval_every}")
        print(f"Logging will trigger at step: {args.log_every}")
        print(f"Training for: {args.steps} steps\n")
        
        # Run the training
        try:
            train_eval.train_eval(agent, train_env, eval_env, train_replay, 
                                 eval_replay, logger, args)
        except Exception as e:
            print(f"Training completed (or stopped): {e}")
        
        # Read the JSONL file
        metrics_file = logdir / 'metrics.jsonl'
        if not metrics_file.exists():
            print(f"ERROR: Metrics file not found at {metrics_file}")
            return False
            
        metrics_by_step = {}
        with open(metrics_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                step_num = data['step']
                metrics_by_step[step_num] = data
                
        print(f"Found {len(metrics_by_step)} logging entries")
        print("\nLogging entries by step:")
        for step_num in sorted(metrics_by_step.keys()):
            data = metrics_by_step[step_num]
            has_eval = any(k.startswith('eval_episode') for k in data.keys())
            eval_keys = [k for k in data.keys() if 'eval' in k.lower()]
            print(f"  Step {step_num}: {'[HAS EVAL METRICS]' if has_eval else ''} {eval_keys[:3]}")
            
        # Check for the bug
        print("\n" + "-"*70)
        print("BUG ANALYSIS:")
        print("-"*70)
        
        # Find when eval metrics appear
        eval_metric_steps = []
        for step_num, data in metrics_by_step.items():
            if any(k.startswith('eval_episode') for k in data.keys()):
                eval_metric_steps.append(step_num)
                
        if eval_metric_steps:
            print(f"\nEval metrics appear at steps: {eval_metric_steps}")
            print(f"Expected eval to occur at step: ~{args.eval_every}")
            
            first_eval_logged = eval_metric_steps[0]
            expected_eval_step = args.eval_every
            
            if first_eval_logged > expected_eval_step + 10:
                print(f"\n🐛 BUG CONFIRMED:")
                print(f"   Eval occurred at step ~{expected_eval_step}")
                print(f"   But metrics logged at step {first_eval_logged}")
                print(f"   Delay: ~{first_eval_logged - expected_eval_step} steps")
                print(f"\n   This happens because logger.write() is not called")
                print(f"   immediately after evaluation completes.")
                return True
            else:
                print(f"\n✓ No obvious timing bug detected")
                print(f"   (Eval metrics logged close to when eval occurred)")
                return False
        else:
            print("\nNo eval metrics found in logs")
            print("(Training may not have reached eval_every step)")
            return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("EVALUATION LOGGING BUG TEST")
    print("="*70)
    print("\nThis test demonstrates that evaluation metrics are logged")
    print("at a later step than when the evaluation actually occurred.")
    print()
    
    bug_found = test_eval_logging_timing()
    
    print("\n" + "="*70)
    if bug_found:
        print("RESULT: Bug confirmed - eval metrics logged at wrong step")
    else:
        print("RESULT: Test completed")
    print("="*70 + "\n")
