"""
Simple demonstration of the evaluation logging bug.

This test simulates the key parts of the training loop to show that
evaluation metrics are logged at a later step than when they were collected.
"""

import json
import tempfile
from pathlib import Path


class StepCounter:
    """Simulates the step counter."""
    def __init__(self):
        self.value = 0
    
    def __int__(self):
        return self.value
    
    def increment(self):
        self.value += 1


class Metrics:
    """Simulates the metrics accumulator."""
    def __init__(self):
        self.data = {}
        
    def add(self, metrics_dict, prefix=None):
        for key, value in metrics_dict.items():
            full_key = f"{prefix}/{key}" if prefix else key
            self.data[full_key] = value
            
    def result(self):
        return self.data.copy()
    
    def reset(self):
        self.data.clear()


class Logger:
    """Simulates the logger."""
    def __init__(self, step, output_file):
        self.step = step
        self.output_file = output_file
        self.pending_metrics = []
        
    def add(self, metrics_dict):
        current_step = int(self.step)
        for key, value in metrics_dict.items():
            self.pending_metrics.append((current_step, key, value))
            
    def write(self):
        if not self.pending_metrics:
            return
        
        # Group by step
        by_step = {}
        for step, key, value in self.pending_metrics:
            if step not in by_step:
                by_step[step] = {'step': step}
            by_step[step][key] = value
            
        # Write to file
        with open(self.output_file, 'a') as f:
            for step in sorted(by_step.keys()):
                f.write(json.dumps(by_step[step]) + '\n')
                
        self.pending_metrics.clear()


def simulate_training_with_bug():
    """
    Simulates the current buggy behavior where eval metrics are logged
    at a later step than when they were collected.
    """
    print("\n" + "="*70)
    print("SIMULATION: Current (Buggy) Behavior")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = Path(tmpdir) / "metrics_buggy.jsonl"
        
        step = StepCounter()
        metrics = Metrics()
        logger = Logger(step, logfile)
        
        eval_every = 30
        log_every = 50
        
        print(f"\nConfiguration:")
        print(f"  - Eval triggers at: step {eval_every}")
        print(f"  - Logging triggers at: step {log_every}")
        print(f"  - Training steps per iteration: 10\n")
        
        # Simulate training loop
        while step.value < 100:
            # Check if should eval
            if step.value == eval_every:
                print(f"[Step {step.value}] Evaluation triggered")
                # Simulate evaluation collecting metrics
                eval_score = 42.0
                metrics.add({'avg_score': eval_score}, prefix='eval_episode')
                print(f"[Step {step.value}] Eval collected score: {eval_score}")
                print(f"[Step {step.value}] ⚠️  NO logger.write() called here!")
                
            # Simulate training steps
            for _ in range(10):
                step.increment()
                
            # Check if should log
            if step.value >= log_every and step.value < log_every + 10:
                print(f"[Step {step.value}] Logging triggered")
                logger.add(metrics.result())
                metrics.reset()
                logger.write()
                print(f"[Step {step.value}] Metrics written to file")
                break
                
        # Read and display results
        print(f"\n{'='*70}")
        print("Results written to file:")
        print("="*70)
        
        with open(logfile, 'r') as f:
            for line in f:
                data = json.loads(line)
                has_eval = 'eval_episode/avg_score' in data
                if has_eval:
                    print(f"Step {data['step']}: {data}")
                    print(f"\n🐛 BUG: Eval score logged at step {data['step']},")
                    print(f"        but evaluation actually occurred at step {eval_every}!")
                    print(f"        Delay: {data['step'] - eval_every} steps\n")


def simulate_training_fixed():
    """
    Simulates the fixed behavior where eval metrics are logged
    immediately at the correct step.
    """
    print("\n" + "="*70)
    print("SIMULATION: Fixed Behavior")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = Path(tmpdir) / "metrics_fixed.jsonl"
        
        step = StepCounter()
        metrics = Metrics()
        logger = Logger(step, logfile)
        
        eval_every = 30
        log_every = 50
        
        print(f"\nConfiguration:")
        print(f"  - Eval triggers at: step {eval_every}")
        print(f"  - Logging triggers at: step {log_every}")
        print(f"  - Training steps per iteration: 10\n")
        
        # Simulate training loop
        while step.value < 100:
            # Check if should eval
            if step.value == eval_every:
                print(f"[Step {step.value}] Evaluation triggered")
                # Simulate evaluation collecting metrics
                eval_score = 42.0
                metrics.add({'avg_score': eval_score}, prefix='eval_episode')
                print(f"[Step {step.value}] Eval collected score: {eval_score}")
                
                # FIX: Write immediately after eval
                print(f"[Step {step.value}] ✓ logger.write() called immediately!")
                logger.add(metrics.result())
                logger.write()
                metrics.reset()
                
            # Simulate training steps
            for _ in range(10):
                step.increment()
                
            # Check if should log
            if step.value >= log_every and step.value < log_every + 10:
                print(f"[Step {step.value}] Logging triggered")
                logger.add(metrics.result())
                metrics.reset()
                logger.write()
                print(f"[Step {step.value}] Metrics written to file")
                break
                
        # Read and display results
        print(f"\n{'='*70}")
        print("Results written to file:")
        print("="*70)
        
        with open(logfile, 'r') as f:
            for line in f:
                data = json.loads(line)
                has_eval = 'eval_episode/avg_score' in data
                if has_eval:
                    print(f"Step {data['step']}: {data}")
                    print(f"\n✓ FIXED: Eval score logged at step {data['step']},")
                    print(f"         exactly when evaluation occurred!\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("DEMONSTRATION: Evaluation Logging Timing Bug")
    print("="*70)
    print("\nThis demonstrates that evaluation metrics are currently logged")
    print("at a later step than when the evaluation actually occurs.\n")
    
    simulate_training_with_bug()
    simulate_training_fixed()
    
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nThe bug occurs because logger.write() is not called immediately")
    print("after evaluation completes. Instead, metrics sit in the metrics")
    print("object until the next should_log() trigger, which happens at a")
    print("later step.")
    print("\nFix: Add logger.write() immediately after evaluation (line 165)")
    print("     in embodied/run/train_eval.py")
    print("="*70 + "\n")
