#!/usr/bin/env python3
"""
Example of using the debug function for custom debugging.

This script shows how to use debug_subgoal_visualization_shapes()
in your own debugging scenarios.
"""

import torch
from hieros.hieros import debug_subgoal_visualization_shapes


def example_basic_usage():
    """
    Basic example of using the debug function.
    
    This shows how to call the function when you have the relevant tensors.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80 + "\n")
    
    # Create example tensors (these would come from your actual code)
    batch_size = 8
    subgoal_shape = [8, 8]
    decoded_features = 1280
    
    # Example tensors
    cached_subgoal = torch.randn(batch_size, *subgoal_shape)
    decoded_subgoal = torch.randn(batch_size, decoded_features)
    subgoal_with_time = decoded_subgoal.unsqueeze(1)
    
    subactor_state = [
        {
            "deter": torch.randn(batch_size, 256),
            "stoch": torch.randn(batch_size, 1024),
        }
    ]
    
    state_with_time = {
        k: v.unsqueeze(1) for k, v in subactor_state[0].items()
    }
    
    # Call the debug function
    debug_info = debug_subgoal_visualization_shapes(
        cached_subgoal=cached_subgoal,
        subactor_state=subactor_state,
        decoded_subgoal=decoded_subgoal,
        subgoal_with_time=subgoal_with_time,
        state_with_time=state_with_time,
        subactor_idx=0,
        enable_logging=True,
    )
    
    # Use the returned debug_info
    if debug_info.get("errors"):
        print("⚠️  Found errors, need to investigate!")
        for error in debug_info["errors"]:
            print(f"  - {error}")
    else:
        print("✅ No errors detected, shapes are valid!")
    
    return debug_info


def example_conditional_debugging():
    """
    Example of using the debug function conditionally.
    
    This shows how to enable debugging only in certain conditions.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Conditional Debugging")
    print("="*80 + "\n")
    
    # Simulate different scenarios
    scenarios = [
        ("Small batch", 4, True),
        ("Medium batch", 16, False),
        ("Large batch", 64, True),
    ]
    
    for name, batch_size, debug_enabled in scenarios:
        print(f"\nScenario: {name} (batch_size={batch_size}, debug={debug_enabled})")
        
        # Create tensors
        cached_subgoal = torch.randn(batch_size, 8, 8)
        decoded_subgoal = torch.randn(batch_size, 1280)
        subgoal_with_time = decoded_subgoal.unsqueeze(1)
        
        subactor_state = [
            {
                "deter": torch.randn(batch_size, 256),
                "stoch": torch.randn(batch_size, 1024),
            }
        ]
        
        state_with_time = {
            k: v.unsqueeze(1) for k, v in subactor_state[0].items()
        }
        
        # Only debug if enabled
        debug_info = debug_subgoal_visualization_shapes(
            cached_subgoal=cached_subgoal,
            subactor_state=subactor_state,
            decoded_subgoal=decoded_subgoal,
            subgoal_with_time=subgoal_with_time,
            state_with_time=state_with_time,
            subactor_idx=0,
            enable_logging=debug_enabled,
        )
        
        if not debug_enabled:
            print("  (Debug logging disabled)")


def example_error_detection():
    """
    Example of intentionally creating errors to see how they're detected.
    
    This demonstrates the validation capabilities of the debug function.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Error Detection")
    print("="*80 + "\n")
    
    # Create tensors with INTENTIONAL errors
    batch_size = 8
    wrong_batch_size = 16
    
    print("Creating tensors with intentional batch size mismatch...")
    
    # cached_subgoal has different batch size
    cached_subgoal = torch.randn(wrong_batch_size, 8, 8)  # Wrong!
    decoded_subgoal = torch.randn(wrong_batch_size, 1280)
    subgoal_with_time = decoded_subgoal.unsqueeze(1)
    
    # But state has correct batch size
    subactor_state = [
        {
            "deter": torch.randn(batch_size, 256),
            "stoch": torch.randn(batch_size, 1024),
        }
    ]
    
    state_with_time = {
        k: v.unsqueeze(1) for k, v in subactor_state[0].items()
    }
    
    # Call debug function - should detect the mismatch
    debug_info = debug_subgoal_visualization_shapes(
        cached_subgoal=cached_subgoal,
        subactor_state=subactor_state,
        decoded_subgoal=decoded_subgoal,
        subgoal_with_time=subgoal_with_time,
        state_with_time=state_with_time,
        subactor_idx=0,
        enable_logging=True,
    )
    
    # Verify errors were detected
    assert debug_info.get("errors"), "Expected errors to be detected!"
    print(f"\n✅ Successfully detected {len(debug_info['errors'])} error(s)!")


def example_integration_pattern():
    """
    Example of how to integrate debugging into your own code.
    
    This shows the pattern used in hieros.py.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Integration Pattern")
    print("="*80 + "\n")
    
    # Simulate a config object
    class Config:
        def __init__(self, debug=False):
            self.debug = debug
    
    config = Config(debug=True)
    
    print("Code pattern for integration:")
    print("""
    # In your computation loop:
    state_with_time = {k: v.unsqueeze(1) for k, v in subactor_state[0].items()}
    decoded_subgoal = subactor.decode_subgoal(cached_subgoal, isfirst=False)
    subgoal_with_time = decoded_subgoal.unsqueeze(1)
    
    # Add debug logging (only runs when config.debug=True)
    if config.debug:
        debug_subgoal_visualization_shapes(
            cached_subgoal=cached_subgoal,
            subactor_state=subactor_state,
            decoded_subgoal=decoded_subgoal,
            subgoal_with_time=subgoal_with_time,
            state_with_time=state_with_time,
            subactor_idx=i,
            enable_logging=True,
        )
    
    # Continue with computation
    reward = subactor._subgoal_reward(state_with_time, subgoal_with_time)
    """)
    
    print("\nThis pattern ensures:")
    print("  ✅ No performance impact when debug=False")
    print("  ✅ Detailed logging when debug=True")
    print("  ✅ Early detection of shape issues")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("DEBUG FUNCTION USAGE EXAMPLES")
    print("="*80)
    
    try:
        example_basic_usage()
        example_conditional_debugging()
        example_error_detection()
        example_integration_pattern()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey takeaways:")
        print("  1. Use enable_logging=True to see detailed output")
        print("  2. Use enable_logging=False for silent validation")
        print("  3. Check debug_info['errors'] to detect issues programmatically")
        print("  4. Integrate with config.debug flag for conditional debugging")
        print("")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have torch installed and hieros module available.")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
