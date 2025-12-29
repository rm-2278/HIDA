#!/usr/bin/env python3
"""
Test script for subgoal visualization debug functionality.

This script tests the debug_subgoal_visualization_shapes function
by simulating the tensor shapes involved in subgoal reward computation.

Usage:
    python test_subgoal_debug.py
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent / "hieros"))

import torch
from hieros.hieros import debug_subgoal_visualization_shapes


def test_correct_shapes():
    """Test with correct tensor shapes (should pass validation)."""
    print("\n" + "="*80)
    print("TEST 1: Testing with CORRECT tensor shapes")
    print("="*80)
    
    batch_size = 4
    subgoal_shape = [8, 8]
    decoded_features = 1280  # Example: dyn_deter (256) + dyn_stoch*dyn_discrete (32*32=1024)
    deter_dim = 256
    stoch_dim = 1024
    
    # Create mock tensors with correct shapes
    cached_subgoal = torch.randn(batch_size, *subgoal_shape)
    decoded_subgoal = torch.randn(batch_size, decoded_features)
    subgoal_with_time = decoded_subgoal.unsqueeze(1)
    
    subactor_state = [
        {
            "deter": torch.randn(batch_size, deter_dim),
            "stoch": torch.randn(batch_size, stoch_dim),
        }
    ]
    
    state_with_time = {
        k: v.unsqueeze(1) for k, v in subactor_state[0].items()
    }
    
    # Call debug function
    debug_info = debug_subgoal_visualization_shapes(
        cached_subgoal=cached_subgoal,
        subactor_state=subactor_state,
        decoded_subgoal=decoded_subgoal,
        subgoal_with_time=subgoal_with_time,
        state_with_time=state_with_time,
        subactor_idx=0,
        enable_logging=True,
    )
    
    # Check for errors
    if debug_info.get("errors"):
        print("\n❌ TEST FAILED: Unexpected errors detected")
        return False
    else:
        print("\n✅ TEST PASSED: All shapes are valid")
        return True


def test_incorrect_shapes_case1():
    """Test with incorrect cached_subgoal shape (simulating the original bug)."""
    print("\n" + "="*80)
    print("TEST 2: Testing with INCORRECT cached_subgoal shape")
    print("(Simulating original bug where batch dimension was wrong)")
    print("="*80)
    
    batch_size = 8
    wrong_batch = 64  # Wrong batch size
    subgoal_shape = [8, 8]
    decoded_features = 1280
    deter_dim = 256
    stoch_dim = 1024
    
    # Create mock tensors with INCORRECT shapes
    # This simulates what would happen if cached_subgoal had wrong batch size
    cached_subgoal = torch.randn(wrong_batch, *subgoal_shape)  # Wrong!
    decoded_subgoal = torch.randn(wrong_batch, decoded_features)  # Will be wrong too
    subgoal_with_time = decoded_subgoal.unsqueeze(1)
    
    subactor_state = [
        {
            "deter": torch.randn(batch_size, deter_dim),
            "stoch": torch.randn(batch_size, stoch_dim),
        }
    ]
    
    state_with_time = {
        k: v.unsqueeze(1) for k, v in subactor_state[0].items()
    }
    
    # Call debug function
    debug_info = debug_subgoal_visualization_shapes(
        cached_subgoal=cached_subgoal,
        subactor_state=subactor_state,
        decoded_subgoal=decoded_subgoal,
        subgoal_with_time=subgoal_with_time,
        state_with_time=state_with_time,
        subactor_idx=0,
        enable_logging=True,
    )
    
    # Check for errors (should have errors)
    if debug_info.get("errors"):
        print("\n✅ TEST PASSED: Errors correctly detected")
        return True
    else:
        print("\n❌ TEST FAILED: Should have detected shape mismatch")
        return False


def test_incorrect_shapes_case2():
    """Test with incorrect time dimension."""
    print("\n" + "="*80)
    print("TEST 3: Testing with INCORRECT time dimension")
    print("="*80)
    
    batch_size = 4
    subgoal_shape = [8, 8]
    decoded_features = 1280
    deter_dim = 256
    stoch_dim = 1024
    
    # Create mock tensors with INCORRECT time dimension
    cached_subgoal = torch.randn(batch_size, *subgoal_shape)
    decoded_subgoal = torch.randn(batch_size, decoded_features)
    # Wrong: add time dimension with size 2 instead of 1
    subgoal_with_time = decoded_subgoal.unsqueeze(1).repeat(1, 2, 1)
    
    subactor_state = [
        {
            "deter": torch.randn(batch_size, deter_dim),
            "stoch": torch.randn(batch_size, stoch_dim),
        }
    ]
    
    state_with_time = {
        k: v.unsqueeze(1) for k, v in subactor_state[0].items()
    }
    
    # Call debug function
    debug_info = debug_subgoal_visualization_shapes(
        cached_subgoal=cached_subgoal,
        subactor_state=subactor_state,
        decoded_subgoal=decoded_subgoal,
        subgoal_with_time=subgoal_with_time,
        state_with_time=state_with_time,
        subactor_idx=0,
        enable_logging=True,
    )
    
    # Check for errors (should have errors)
    if debug_info.get("errors"):
        print("\n✅ TEST PASSED: Errors correctly detected")
        return True
    else:
        print("\n❌ TEST FAILED: Should have detected wrong time dimension")
        return False


def test_disabled_logging():
    """Test with logging disabled."""
    print("\n" + "="*80)
    print("TEST 4: Testing with logging DISABLED")
    print("="*80)
    
    batch_size = 4
    subgoal_shape = [8, 8]
    decoded_features = 1280
    deter_dim = 256
    stoch_dim = 1024
    
    # Create mock tensors
    cached_subgoal = torch.randn(batch_size, *subgoal_shape)
    decoded_subgoal = torch.randn(batch_size, decoded_features)
    subgoal_with_time = decoded_subgoal.unsqueeze(1)
    
    subactor_state = [
        {
            "deter": torch.randn(batch_size, deter_dim),
            "stoch": torch.randn(batch_size, stoch_dim),
        }
    ]
    
    state_with_time = {
        k: v.unsqueeze(1) for k, v in subactor_state[0].items()
    }
    
    # Call debug function with logging disabled
    debug_info = debug_subgoal_visualization_shapes(
        cached_subgoal=cached_subgoal,
        subactor_state=subactor_state,
        decoded_subgoal=decoded_subgoal,
        subgoal_with_time=subgoal_with_time,
        state_with_time=state_with_time,
        subactor_idx=0,
        enable_logging=False,  # Disabled
    )
    
    # Should return empty dict
    if not debug_info:
        print("\n✅ TEST PASSED: Logging disabled, no output produced")
        return True
    else:
        print("\n❌ TEST FAILED: Should return empty dict when disabled")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING DEBUG FUNCTION TESTS")
    print("="*80)
    
    results = []
    
    # Run all tests
    results.append(("Correct shapes", test_correct_shapes()))
    results.append(("Incorrect batch size", test_incorrect_shapes_case1()))
    results.append(("Incorrect time dimension", test_incorrect_shapes_case2()))
    results.append(("Logging disabled", test_disabled_logging()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
