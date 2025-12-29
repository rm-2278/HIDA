#!/usr/bin/env python3
"""
Simple syntax and structure test for the debug function.

This test validates that the debug function is correctly defined
and has the expected structure without requiring all dependencies.
"""

import ast
import sys


def test_debug_function_exists():
    """Check if the debug function exists in hieros.py."""
    print("="*80)
    print("TEST: Checking if debug_subgoal_visualization_shapes exists")
    print("="*80)
    
    with open("hieros/hieros.py", "r") as f:
        content = f.read()
    
    # Parse the file
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"❌ FAILED: Syntax error in hieros.py: {e}")
        return False
    
    # Find the function
    function_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == "debug_subgoal_visualization_shapes":
                function_found = True
                print(f"✅ Found function: {node.name}")
                
                # Check parameters
                expected_params = [
                    "cached_subgoal",
                    "subactor_state", 
                    "decoded_subgoal",
                    "subgoal_with_time",
                    "state_with_time",
                    "subactor_idx",
                    "enable_logging",
                ]
                
                actual_params = [arg.arg for arg in node.args.args]
                print(f"  Parameters: {actual_params}")
                
                missing_params = set(expected_params) - set(actual_params)
                if missing_params:
                    print(f"  ⚠️  Missing parameters: {missing_params}")
                    return False
                
                print(f"  ✅ All expected parameters present")
                
                # Check if it returns something
                has_return = False
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Return):
                        has_return = True
                        break
                
                if has_return:
                    print(f"  ✅ Function has return statement")
                else:
                    print(f"  ⚠️  Function has no return statement")
                
                break
    
    if not function_found:
        print(f"❌ FAILED: Function debug_subgoal_visualization_shapes not found")
        return False
    
    print(f"\n✅ TEST PASSED: Function structure is correct")
    return True


def test_debug_function_integration():
    """Check if the debug function is called in the right place."""
    print("\n" + "="*80)
    print("TEST: Checking if debug function is integrated")
    print("="*80)
    
    with open("hieros/hieros.py", "r") as f:
        content = f.read()
    
    # Check if the function is called with debug config check
    if "if self._config.debug:" in content and "debug_subgoal_visualization_shapes(" in content:
        print("✅ Debug function is called with config check")
        
        # Count occurrences
        call_count = content.count("debug_subgoal_visualization_shapes(")
        print(f"  Found {call_count} call(s) to the debug function")
        
        # Check if it's in the right section (near decode_subgoal)
        decode_section = content.find("decode_subgoal(cached_subgoal")
        debug_call = content.find("debug_subgoal_visualization_shapes(")
        
        if decode_section > 0 and debug_call > decode_section:
            print(f"  ✅ Debug call is positioned after decode_subgoal call")
        else:
            print(f"  ⚠️  Debug call position may be incorrect")
        
        print(f"\n✅ TEST PASSED: Integration looks correct")
        return True
    else:
        print(f"❌ FAILED: Debug function not properly integrated")
        return False


def test_enhanced_error_handling():
    """Check if enhanced error handling is present."""
    print("\n" + "="*80)
    print("TEST: Checking enhanced error handling")
    print("="*80)
    
    with open("hieros/hieros.py", "r") as f:
        content = f.read()
    
    # Check for enhanced error messages
    checks = [
        ("Tensor dimension mismatch detected", "Enhanced error message"),
        ("To debug, enable debug mode", "Debug suggestion"),
        ("Common causes:", "Error diagnosis help"),
    ]
    
    all_found = True
    for text, description in checks:
        if text in content:
            print(f"  ✅ Found: {description}")
        else:
            print(f"  ❌ Missing: {description}")
            all_found = False
    
    if all_found:
        print(f"\n✅ TEST PASSED: All enhanced error messages present")
        return True
    else:
        print(f"\n❌ TEST FAILED: Some error messages missing")
        return False


def test_file_syntax():
    """Test that the file has valid Python syntax."""
    print("\n" + "="*80)
    print("TEST: Checking Python syntax")
    print("="*80)
    
    try:
        with open("hieros/hieros.py", "r") as f:
            content = f.read()
        ast.parse(content)
        print("✅ TEST PASSED: Valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"❌ TEST FAILED: Syntax error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING STRUCTURE TESTS FOR DEBUG FUNCTIONALITY")
    print("="*80)
    print()
    
    results = []
    
    # Run all tests
    results.append(("File syntax", test_file_syntax()))
    results.append(("Debug function exists", test_debug_function_exists()))
    results.append(("Debug function integration", test_debug_function_integration()))
    results.append(("Enhanced error handling", test_enhanced_error_handling()))
    
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
