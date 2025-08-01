#!/usr/bin/env python3
"""
Test script to verify the log parsing functionality for failed fix attempts per step.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tester import AgentTester

def get_ground_truth():
    """
    Define expected ground truth values for test log files.
    TODO: Fill in actual expected values for test_log_1.log and test_log_2.log
    """
    ground_truth = {
        # Format: log_file_name: {
        #   'total_failures': expected_total_failures,
        #   'total_attempts': expected_total_attempts,
        #   'analyses': {
        #       analysis_num: {
        #           step_num: {
        #               'failed_fix_attempts': expected_failed_fix_attempts,
        #               'total_attempts': expected_total_attempts,
        #               'successful_executions': expected_successful_executions,
        #               'initial_success': expected_initial_success,
        #               'fix_succeeded': expected_fix_succeeded
        #           }
        #       }
        #   }
        # }
        
        'test_log_1.log': {
            'total_failures': 3,
            'total_attempts': 9,
            'analyses': {
                1: {  # Analysis 1
                    1: {  # Step 1
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    2: {  
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    3: {  
                        'failed_fix_attempts': 0,
                        'total_attempts': 2,
                        'successful_executions': 1,
                        'initial_success': False,
                        'fix_succeeded': True
                    },
                    4: {  
                        'failed_fix_attempts': 0,
                        'total_attempts': 2,
                        'successful_executions': 1,
                        'initial_success': False,
                        'fix_succeeded': True
                    },
                    5: {  
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    6: {  
                        'failed_fix_attempts': 0,
                        'total_attempts': 2,
                        'successful_executions': 1,
                        'initial_success': False,
                        'fix_succeeded': True
                    }
                    
                }
            }
        },
        
        'test_log_2.log': {
            'total_failures': 7,
            'total_attempts': 24,  # 7 + 11 + 6
            'analyses': {
                1: {  # Analysis 1
                    1: {  # Step 1
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    2: {  # Step 2
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    3: {  # Step 3
                        'failed_fix_attempts': 0,
                        'total_attempts': 2,
                        'successful_executions': 1,
                        'initial_success': False,
                        'fix_succeeded': True
                    },
                    4: {
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    5: {
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    6: {
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                },
                2: {  # Analysis 2
                    1: {  # Step 1
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    2: {  # Step 2
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    3: {  # Step 3
                        'failed_fix_attempts': 0,
                        'total_attempts': 2,
                        'successful_executions': 1,
                        'initial_success': False,
                        'fix_succeeded': True
                    },
                    4: {  # Step 4
                        'failed_fix_attempts': 3,
                        'total_attempts': 4,
                        'successful_executions': 0,
                        'initial_success': False,
                        'fix_succeeded': False
                    },
                    5: {  # Step 5
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    6: {  # Step 6
                        'failed_fix_attempts': 0,
                        'total_attempts': 2,
                        'successful_executions': 1,
                        'initial_success': False,
                        'fix_succeeded': True
                    }
                },
                3: {  # Analysis 3
                    1: {  # Step 1
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    2: {  # Step 2
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    3: {  # Step 3
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    4: {  # Step 4
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    },
                    5: {  # Step 5
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    }, 
                    6: {  # Step 6
                        'failed_fix_attempts': 0,
                        'total_attempts': 1,
                        'successful_executions': 1,
                        'initial_success': True,
                        'fix_succeeded': False
                    }
                }
            }
        }
    }
    
    return ground_truth

def compare_results(actual, expected, path=""):
    """
    Recursively compare actual vs expected results and report differences.
    
    Args:
        actual: Actual parsed results
        expected: Expected ground truth results
        path: Current path in the data structure (for error reporting)
    
    Returns:
        List of error messages
    """
    errors = []
    
    if isinstance(expected, dict) and isinstance(actual, dict):
        # Check for missing keys in actual
        for key in expected:
            if key not in actual:
                errors.append(f"Missing key at {path}.{key}")
            else:
                errors.extend(compare_results(actual[key], expected[key], f"{path}.{key}"))
        
        # Check for unexpected keys in actual
        for key in actual:
            if key not in expected:
                errors.append(f"Unexpected key at {path}.{key}")
    
    elif actual != expected:
        errors.append(f"Value mismatch at {path}: expected {expected}, got {actual}")
    
    return errors

def validate_against_ground_truth(log_stats):
    """Validate parsed results against ground truth values."""
    ground_truth = get_ground_truth()
    
    print(f"\n=== GROUND TRUTH VALIDATION ===")
    
    all_passed = True
    
    for log_file_path, actual_data in log_stats.items():
        log_file_name = os.path.basename(log_file_path)
        print(f"\n📄 Validating: {log_file_name}")
        
        if log_file_name not in ground_truth:
            print(f"  ⚠️  No ground truth defined for {log_file_name}")
            continue
        
        expected_data = ground_truth[log_file_name]
        
        # Compare top-level statistics
        print(f"  📊 Top-level statistics:")
        if actual_data['total_failures'] != expected_data['total_failures']:
            print(f"    ❌ total_failures: expected {expected_data['total_failures']}, got {actual_data['total_failures']}")
            all_passed = False
        else:
            print(f"    ✅ total_failures: {actual_data['total_failures']}")
        
        if actual_data['total_attempts'] != expected_data['total_attempts']:
            print(f"    ❌ total_attempts: expected {expected_data['total_attempts']}, got {actual_data['total_attempts']}")
            all_passed = False
        else:
            print(f"    ✅ total_attempts: {actual_data['total_attempts']}")
        
        # Compare detailed analysis data
        print(f"  📋 Detailed analysis data:")
        errors = compare_results(actual_data['analyses'], expected_data['analyses'], f"{log_file_name}.analyses")
        
        if errors:
            print(f"    ❌ Found {len(errors)} error(s):")
            for error in errors:
                print(f"      - {error}")
            all_passed = False
        else:
            print(f"    ✅ All detailed data matches")
    
    print(f"\n=== VALIDATION SUMMARY ===")
    if all_passed:
        print("🎉 All ground truth validations passed!")
    else:
        print("❌ Some validations failed. Please check the ground truth values.")
    
    return all_passed

def print_actual_results_for_ground_truth(log_stats):
    """
    Helper function to print actual results in a format that can be copied
    into the ground truth structure.
    """
    print("\n=== ACTUAL RESULTS FOR GROUND TRUTH ===")
    print("Copy these values into the ground_truth dictionary:")
    print()
    
    for log_file_path, data in log_stats.items():
        log_file_name = os.path.basename(log_file_path)
        print(f"'{log_file_name}': {{")
        print(f"    'total_failures': {data['total_failures']},")
        print(f"    'total_attempts': {data['total_attempts']},")
        print(f"    'analyses': {{")
        
        for analysis_num, steps in data['analyses'].items():
            print(f"        {analysis_num}: {{")
            for step_num, step_data in steps.items():
                print(f"            {step_num}: {{")
                print(f"                'failed_fix_attempts': {step_data['failed_fix_attempts']},")
                print(f"                'total_attempts': {step_data['total_attempts']},")
                print(f"                'successful_executions': {step_data['successful_executions']},")
                print(f"                'initial_success': {step_data['initial_success']},")
                print(f"                'fix_succeeded': {step_data['fix_succeeded']}")
                print(f"            }},")
            print(f"        }},")
        print(f"    }}")
        print(f"}},")
        print()

def test_log_parsing(validate_ground_truth=True, show_actual=False):
    """Test the log parsing functionality with a sample log file."""
    
    # Create a minimal tester instance
    tester = AgentTester("dummy.h5ad", "dummy.txt", "log_parsing_test")
    
    # Test with the actual log file
    log_dir = "test_logs"  # Adjust path as needed
    
    print("Testing log parsing with actual log files...")
    log_stats = tester._analyze_logs(log_dir)
    
    print("\n=== LOG PARSING RESULTS ===")
    total_failures = sum(s['total_failures'] for s in log_stats.values())
    total_attempts = sum(s['total_attempts'] for s in log_stats.values())
    print(f"Total failures (all files): {total_failures}")
    print(f"Total attempts (all files): {total_attempts}")
    print(f"Log files found: {list(log_stats.keys())}")
    
    for log_file, analysis_data in log_stats.items():
        print(f"\nLog File: {os.path.basename(log_file)}")
        print(f"  Total failures: {analysis_data['total_failures']}")
        print(f"  Total attempts: {analysis_data['total_attempts']}")
        print(f"  Analyses found: {list(analysis_data['analyses'].keys())}")
        
        for analysis_num, steps in analysis_data['analyses'].items():
            print(f"  Analysis {analysis_num}:")
            for step_num, step_data in steps.items():
                failed_attempts = step_data['failed_fix_attempts']
                total_attempts = step_data['total_attempts']
                successful_executions = step_data['successful_executions']
                print(f"    Step {step_num}: {failed_attempts} failed fix attempts, {successful_executions} successful executions, {total_attempts} total attempts")
    
    # Check for test logs directory for validation
    test_logs_dir = "./test_logs"
    if os.path.exists(test_logs_dir):
        print(f"\n=== TESTING WITH TEST LOGS ===")
        test_log_stats = tester._analyze_logs(test_logs_dir)
        
        print(f"Found {len(test_log_stats)} test log files")
        
        for log_file, analysis_data in test_log_stats.items():
            print(f"\nTest Log File: {os.path.basename(log_file)}")
            print(f"  Total failures: {analysis_data['total_failures']}")
            print(f"  Total attempts: {analysis_data['total_attempts']}")
            print(f"  Analyses found: {list(analysis_data['analyses'].keys())}")
            
            for analysis_num, steps in analysis_data['analyses'].items():
                print(f"  Analysis {analysis_num}:")
                for step_num, step_data in steps.items():
                    failed_attempts = step_data['failed_fix_attempts']
                    total_attempts = step_data['total_attempts']
                    successful_executions = step_data['successful_executions']
                    print(f"    Step {step_num}: {failed_attempts} failed fix attempts, {successful_executions} successful executions, {total_attempts} total attempts")
        
        if show_actual:
            print_actual_results_for_ground_truth(test_log_stats)
        
        if validate_ground_truth:
            validate_against_ground_truth(test_log_stats)
    else:
        print(f"\n⚠️  Test logs directory not found: {test_logs_dir}")
        print("Skipping ground truth validation.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test log parsing functionality")
    parser.add_argument("--no-validation", action="store_true", 
                       help="Skip ground truth validation")
    parser.add_argument("--show-actual", action="store_true", 
                       help="Print actual results to help fill in ground truth")
    
    args = parser.parse_args()
    
    test_log_parsing(
        validate_ground_truth=not args.no_validation,
        show_actual=args.show_actual
    ) 