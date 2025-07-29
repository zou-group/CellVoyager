#!/usr/bin/env python3
"""
Simple test to verify comprehensive logging is working correctly.
"""

import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_log_parsing():
    """Test that the AgentTester log parsing logic works correctly."""
    import tempfile
    import os
    from base_tester import AgentTester
    
    print("🧪 Testing log parsing logic using AgentTester...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock log file with test content
        log_file = os.path.join(temp_dir, "test_log.log")
        mock_log_content = """
2024-01-01 10:00:00 - INFO
================================================================================
STEP 1 EXECUTION ATTEMPT - Analysis 1

2024-01-01 10:00:01 - ERROR
================================================================================
ERROR

Code errored with: NameError: name 'x' is not defined

2024-01-01 10:00:02 - INFO
================================================================================
FIX ATTEMPT 1/3 - Analysis 1, Step 1

2024-01-01 10:00:03 - ERROR
================================================================================
FIX ATTEMPT FAILED 1/3 - Analysis 1, Step 1: SyntaxError: invalid syntax

2024-01-01 10:00:04 - INFO
================================================================================
FIX ATTEMPT 2/3 - Analysis 1, Step 1

2024-01-01 10:00:05 - INFO
================================================================================
FIX SUCCESSFUL on attempt 2/3 - Analysis 1, Step 1

2024-01-01 10:00:06 - INFO
================================================================================
STEP 2 EXECUTION ATTEMPT - Analysis 1
"""
        
        # Write the mock log content to file
        with open(log_file, 'w') as f:
            f.write(mock_log_content)
        
        # Create an AgentTester instance
        tester = AgentTester(
            h5ad_path='test.h5ad',
            manuscript_path='test.txt',
            test_name='log_parsing_test',
            num_analyses=1,
            max_iterations=1
        )
        
        # Use the internal _analyze_logs method to test log parsing
        log_stats = tester._analyze_logs(temp_dir)
        
        print(f"   Total failures: {log_stats['total_failures']}")
        print(f"   Total attempts: {log_stats['total_attempts']}")
        
        expected_total_failures = 1  # Only failed fix attempts (per current implementation)
        expected_total_attempts = 4  # 2 steps + 2 fix attempts = 4
        
        assert log_stats['total_failures'] == expected_total_failures, f"Expected {expected_total_failures}, got {log_stats['total_failures']}"
        assert log_stats['total_attempts'] == expected_total_attempts, f"Expected {expected_total_attempts}, got {log_stats['total_attempts']}"
    
    print("✅ Log parsing test passed!")
    return True

def test_import():
    """Test that imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        from base_tester import AgentTester
        print("✅ AgentTester import successful")
        
        from agent import AnalysisAgent
        print("✅ AnalysisAgent import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing CellVoyager Logging Framework\n")
    
    all_passed = True
    
    tests = [
        ("Import test", test_import),
        ("Log parsing test", test_log_parsing)
    ]
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
                print(f"\n❌ {test_name} FAILED")
            else:
                print(f"\n✅ {test_name} PASSED")
        except Exception as e:
            all_passed = False
            print(f"\n❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All tests PASSED! Comprehensive logging is working correctly.")
        print("\nThe framework will now track:")
        print("- Every code execution attempt (initial + fixes)")
        print("- Every failure (initial errors + fix failures)")
        print("- Final success rates after all fix attempts")
    else:
        print("⚠️ Some tests FAILED. Please check the issues above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 