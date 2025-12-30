#!/usr/bin/env python3
"""Test script for position heatmap visualization."""

import numpy as np
import sys
import pathlib

# Add project paths
sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent / "embodied"))

from embodied.envs.pinpad import PinPad
from embodied.envs import pinpad_easy

def test_pinpad_heatmap():
    """Test PinPad heatmap generation."""
    print("Testing PinPad position tracking...")
    
    # Create environment
    env = PinPad(task="three", length=100, seed=42)
    
    # Reset environment
    obs = env.step({"action": 0, "reset": True})
    print(f"Initial position: {env.player}")
    
    # Take some random steps
    np.random.seed(42)
    for i in range(50):
        action = np.random.randint(0, 5)
        obs = env.step({"action": action, "reset": False})
    
    # Get statistics
    stats = env.get_position_stats()
    print("\nPosition Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Generate heatmap
    heatmap = env.get_position_heatmap()
    print(f"\nHeatmap shape: {heatmap.shape}")
    print(f"Heatmap dtype: {heatmap.dtype}")
    print(f"Heatmap min/max: {heatmap.min()}/{heatmap.max()}")
    
    # Check that heatmap has reasonable values
    assert heatmap.shape == (64, 64, 3), f"Expected shape (64, 64, 3), got {heatmap.shape}"
    assert heatmap.dtype == np.uint8, f"Expected dtype uint8, got {heatmap.dtype}"
    assert heatmap.min() >= 0 and heatmap.max() <= 255, "Heatmap values out of range"
    
    print("\n✓ PinPad heatmap test passed!")
    return True

def test_pinpad_easy_heatmap():
    """Test PinPadEasy heatmap generation."""
    print("\nTesting PinPadEasy position tracking...")
    
    # Create environment
    env = pinpad_easy.PinPadEasy(task="three", length=100, seed=42)
    
    # Reset environment
    obs = env.step({"action": 0, "reset": True})
    print(f"Initial position: {env.player}")
    
    # Take some random steps
    np.random.seed(42)
    for i in range(50):
        action = np.random.randint(0, 5)
        obs = env.step({"action": action, "reset": False})
    
    # Get statistics
    stats = env.get_position_stats()
    print("\nPosition Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Generate heatmap
    heatmap = env.get_position_heatmap()
    print(f"\nHeatmap shape: {heatmap.shape}")
    print(f"Heatmap dtype: {heatmap.dtype}")
    print(f"Heatmap min/max: {heatmap.min()}/{heatmap.max()}")
    
    # Check that heatmap has reasonable values
    assert heatmap.shape == (64, 64, 3), f"Expected shape (64, 64, 3), got {heatmap.shape}"
    assert heatmap.dtype == np.uint8, f"Expected dtype uint8, got {heatmap.dtype}"
    assert heatmap.min() >= 0 and heatmap.max() <= 255, "Heatmap values out of range"
    
    print("\n✓ PinPadEasy heatmap test passed!")
    return True

def test_coverage_increases():
    """Test that coverage increases as agent explores."""
    print("\nTesting coverage increase...")
    
    env = PinPad(task="three", length=1000, seed=42)
    env.step({"action": 0, "reset": True})
    
    # Initial statistics
    stats1 = env.get_position_stats()
    initial_coverage = stats1["coverage_ratio"]
    
    # Move around
    np.random.seed(42)
    for i in range(200):
        action = np.random.randint(0, 5)
        env.step({"action": action, "reset": False})
    
    # Check that coverage increased
    stats2 = env.get_position_stats()
    final_coverage = stats2["coverage_ratio"]
    
    print(f"Initial coverage: {initial_coverage:.2%}")
    print(f"Final coverage: {final_coverage:.2%}")
    
    assert final_coverage >= initial_coverage, "Coverage should not decrease"
    print("\n✓ Coverage increase test passed!")
    return True

if __name__ == "__main__":
    try:
        test_pinpad_heatmap()
        test_pinpad_easy_heatmap()
        test_coverage_increases()
        print("\n" + "="*50)
        print("All tests passed! ✓")
        print("="*50)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
