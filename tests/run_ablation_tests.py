#!/usr/bin/env python3
"""
Main test runner for CellVoyager ablation studies.
Runs different agent configurations and compares code execution success rates.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add parent directory to Python path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_tester import AgentTester


def run_ablation_study(h5ad_path, manuscript_path, num_analyses=2, max_iterations=3):
    """
    Run all ablation studies and compare results.
    
    Args:
        h5ad_path (str): Path to .h5ad file
        manuscript_path (str): Path to manuscript .txt file
        num_analyses (int): Number of analyses per configuration
        max_iterations (int): Max iterations per analysis
    
    Returns:
        dict: Combined results from all configurations
    """
    
    # Define test configurations using built-in ablation flags
    configurations = [
        {"test_name": "baseline", "use_self_critique": True, "use_VLM": True},
        {"test_name": "no_vlm", "use_self_critique": True, "use_VLM": False},
        {"test_name": "no_critique", "use_self_critique": False, "use_VLM": True},
        {"test_name": "no_vlm_no_critique", "use_self_critique": False, "use_VLM": False},
    ]
    
    all_results = {}
    
    print(f"🚀 Starting CellVoyager Ablation Study")
    print(f"📁 Data: {h5ad_path}")
    print(f"📄 Manuscript: {manuscript_path}")
    print(f"🔢 {num_analyses} analyses × {max_iterations} iterations per configuration\n")
    
    for config in configurations:
        test_name = config["test_name"]
        use_self_critique = config["use_self_critique"]
        use_VLM = config["use_VLM"]
        
        print(f"\n{'='*60}")
        print(f"Testing: {test_name.upper()}")
        print(f"{'='*60}")
        
        # Create tester for this configuration
        tester = AgentTester(
            h5ad_path=h5ad_path,
            manuscript_path=manuscript_path,
            test_name=test_name,
            num_analyses=num_analyses,
            max_iterations=max_iterations
        )
        
        # Run test with specific ablation flags
        results = tester.test_agent(
            use_self_critique=use_self_critique,
            use_VLM=use_VLM
        )
        
        # Save individual results
        tester.save_results()
        
        # Store in combined results
        all_results[test_name] = results
        
        print(f"\n📊 {test_name.upper()} Results:")
        print(f"   Final Success Rate: {results['final_success_rate']:.2%}")
        print(f"   Total Failure Rate: {results['failure_rate']:.2%}")
        print(f"   Final Successful Cells: {results['final_successful_cells']}")
        print(f"   Total Failures: {results['total_failures']}")
        print(f"   Total Attempts: {results['total_code_cells_attempted']}")
    
    return all_results


def save_comparison_report(all_results, output_path="ablation_comparison.json"):
    """Save comparison report of all configurations."""
    
    # Create summary comparison
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'detailed_results': all_results
    }
    
    # Extract key metrics for easy comparison
    for config_name, results in all_results.items():
        comparison['summary'][config_name] = {
            'final_success_rate': results['final_success_rate'],
            'total_failure_rate': results['failure_rate'],
            'final_successful_cells': results['final_successful_cells'],
            'final_failed_cells': results['final_failed_cells'],
            'total_failures': results['total_failures'],
            'total_attempts': results['total_code_cells_attempted'],
            'num_analyses': len(results['analyses'])
        }
    
    # Save combined results
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Comparison report saved to: {output_path}")
    return output_path


def print_summary_table(all_results):
    """Print a formatted summary table of results."""
    
    print(f"\n{'='*90}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*90}")
    
    # Table header
    print(f"{'Configuration':<18} {'Final Success':<12} {'Total Failures':<13} {'Failure Rate':<12} {'Attempts':<9}")
    print("-" * 90)
    
    # Sort by final success rate (descending)
    sorted_results = sorted(all_results.items(), 
                          key=lambda x: x[1]['final_success_rate'], 
                          reverse=True)
    
    for config_name, results in sorted_results:
        final_success = f"{results['final_success_rate']:.1%}"
        total_failures = results['total_failures']
        failure_rate = f"{results['failure_rate']:.1%}"
        attempts = results['total_code_cells_attempted']
        
        print(f"{config_name:<18} {final_success:<12} {total_failures:<13} {failure_rate:<12} {attempts:<9}")
    
    print("-" * 90)
    
    # Show best and worst performing configurations
    best_config, best_results = sorted_results[0]
    worst_config, worst_results = sorted_results[-1]
    
    print(f"\n🏆 Best final success rate: {best_config.upper()} ({best_results['final_success_rate']:.1%})")
    print(f"💥 Lowest failure rate: {min(all_results.items(), key=lambda x: x[1]['failure_rate'])[0].upper()} ({min(r['failure_rate'] for r in all_results.values()):.1%})")
    print(f"⚠️ Highest failure rate: {max(all_results.items(), key=lambda x: x[1]['failure_rate'])[0].upper()} ({max(r['failure_rate'] for r in all_results.values()):.1%})")


def main():
    """Main entry point for ablation testing."""
    parser = argparse.ArgumentParser(description="Run CellVoyager ablation studies")
    parser.add_argument("h5ad_path", help="Path to .h5ad file")
    parser.add_argument("manuscript_path", help="Path to manuscript .txt file")
    parser.add_argument("--num_analyses", type=int, default=2, 
                       help="Number of analyses per configuration (default: 2)")
    parser.add_argument("--max_iterations", type=int, default=3,
                       help="Max iterations per analysis (default: 3)")
    parser.add_argument("--output", default="ablation_comparison.json",
                       help="Output path for comparison report")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.h5ad_path):
        print(f"❌ Error: .h5ad file not found: {args.h5ad_path}")
        return 1
    
    if not os.path.exists(args.manuscript_path):
        print(f"❌ Error: manuscript file not found: {args.manuscript_path}")
        return 1
    
    # Check for required environment variable
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return 1
    
    try:
        # Run ablation study
        all_results = run_ablation_study(
            h5ad_path=args.h5ad_path,
            manuscript_path=args.manuscript_path,
            num_analyses=args.num_analyses,
            max_iterations=args.max_iterations
        )
        
        # Save comparison report
        save_comparison_report(all_results, args.output)
        
        # Print summary
        print_summary_table(all_results)
        
        print(f"\n✅ Ablation study completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during ablation study: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 