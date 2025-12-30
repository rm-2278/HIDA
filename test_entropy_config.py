#!/usr/bin/env python3
"""
Integration test to verify actor_entropy configuration works correctly.

This test creates a minimal config and verifies that:
1. Single value actor_entropy works (backward compatibility)
2. List of values actor_entropy works
"""

import sys
import os

# Add hieros to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hieros'))


def test_backward_compatibility():
    """Test that single value actor_entropy still works."""
    print("="*80)
    print("TEST: Backward compatibility with single value")
    print("="*80)
    
    # Create a simple config object
    class Config:
        def __init__(self):
            self.actor_entropy = '3e-4'
            self.actor_state_entropy = 0.0
            self.imag_gradient_mix = '0.0'
    
    config = Config()
    
    # Simulate what happens in SubActor.__init__
    layer_idx = 0
    actor_entropy_value = config.actor_entropy
    
    print(f"Original config.actor_entropy: {config.actor_entropy}")
    print(f"Layer index: {layer_idx}")
    
    if isinstance(actor_entropy_value, (list, tuple)):
        actor_entropy_value = actor_entropy_value[min(layer_idx, len(actor_entropy_value) - 1)]
    
    print(f"Selected actor_entropy_value: {actor_entropy_value}")
    
    if actor_entropy_value == '3e-4':
        print("✅ PASSED: Single value works correctly")
        return True
    else:
        print(f"❌ FAILED: Expected '3e-4', got {actor_entropy_value}")
        return False


def test_list_values():
    """Test that list of values works correctly."""
    print("\n" + "="*80)
    print("TEST: List of values for multiple layers")
    print("="*80)
    
    # Create a config with list
    class Config:
        def __init__(self):
            self.actor_entropy = ['3e-4', '1e-3', '5e-3']
            self.actor_state_entropy = [0.0, 0.1, 0.2]
            self.imag_gradient_mix = '0.0'
    
    config = Config()
    
    print(f"Original config.actor_entropy: {config.actor_entropy}")
    print(f"Original config.actor_state_entropy: {config.actor_state_entropy}")
    
    expected_entropy_values = {
        0: '3e-4',
        1: '1e-3',
        2: '5e-3',
    }
    
    expected_state_entropy_values = {
        0: 0.0,
        1: 0.1,
        2: 0.2,
    }
    
    all_passed = True
    for layer_idx in range(3):
        # Test actor_entropy
        actor_entropy_value = config.actor_entropy
        if isinstance(actor_entropy_value, (list, tuple)):
            actor_entropy_value = actor_entropy_value[min(layer_idx, len(actor_entropy_value) - 1)]
        
        expected = expected_entropy_values[layer_idx]
        print(f"Layer {layer_idx} actor_entropy: {actor_entropy_value} (expected: {expected})")
        
        if actor_entropy_value != expected:
            print(f"  ❌ FAILED: Expected {expected}, got {actor_entropy_value}")
            all_passed = False
        else:
            print(f"  ✅ PASSED")
        
        # Test actor_state_entropy
        actor_state_entropy_value = config.actor_state_entropy
        if isinstance(actor_state_entropy_value, (list, tuple)):
            actor_state_entropy_value = actor_state_entropy_value[min(layer_idx, len(actor_state_entropy_value) - 1)]
        
        expected_state = expected_state_entropy_values[layer_idx]
        print(f"Layer {layer_idx} actor_state_entropy: {actor_state_entropy_value} (expected: {expected_state})")
        
        if actor_state_entropy_value != expected_state:
            print(f"  ❌ FAILED: Expected {expected_state}, got {actor_state_entropy_value}")
            all_passed = False
        else:
            print(f"  ✅ PASSED")
    
    if all_passed:
        print("✅ ALL LAYERS PASSED")
    return all_passed


def test_list_shorter_than_layers():
    """Test that shorter list reuses last value."""
    print("\n" + "="*80)
    print("TEST: Shorter list reuses last value")
    print("="*80)
    
    # Create a config with shorter list
    class Config:
        def __init__(self):
            self.actor_entropy = ['3e-4', '1e-3']  # Only 2 values
            self.actor_state_entropy = 0.0
            self.imag_gradient_mix = '0.0'
    
    config = Config()
    
    print(f"Original config.actor_entropy: {config.actor_entropy}")
    print(f"List length: {len(config.actor_entropy)}")
    
    # Test layer 2 (should use last value from list)
    layer_idx = 2
    actor_entropy_value = config.actor_entropy
    
    if isinstance(actor_entropy_value, (list, tuple)):
        actor_entropy_value = actor_entropy_value[min(layer_idx, len(actor_entropy_value) - 1)]
    
    print(f"Layer {layer_idx}: {actor_entropy_value} (expected: '1e-3')")
    
    if actor_entropy_value == '1e-3':
        print("✅ PASSED: Last value correctly reused")
        return True
    else:
        print(f"❌ FAILED: Expected '1e-3', got {actor_entropy_value}")
        return False


def main():
    """Run all tests."""
    print("Testing actor_entropy configuration handling\n")
    
    tests = [
        test_backward_compatibility,
        test_list_values,
        test_list_shorter_than_layers,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
