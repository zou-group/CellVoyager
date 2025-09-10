#!/usr/bin/env python3
"""
Main test runner for CellVoyager ablation studies.
Runs different agent configurations as separate batch jobs and compares code execution success rates.
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime

# Add parent directory to Python path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_tester import AgentTester


def create_batch_script(config, h5ad_path, manuscript_path, num_analyses, max_iterations, output_dir):
    """
    Create a batch script for a specific ablation configuration.
    
    Args:
        config (dict): Configuration dictionary with test_name, use_self_critique, use_VLM, use_documentation
        h5ad_path (str): Path to .h5ad file
        manuscript_path (str): Path to manuscript file
        num_analyses (int): Number of analyses
        max_iterations (int): Max iterations per analysis
        output_dir (str): Output directory for results
    
    Returns:
        str: Path to the created batch script
    """
    
    test_name = config["test_name"]
    use_self_critique = config["use_self_critique"]
    use_VLM = config["use_VLM"]
    use_documentation = config["use_documentation"]
    
    # Create batch script content
    batch_script_content = f"""#!/bin/bash
#SBATCH --job-name=ablation_{test_name}
#SBATCH --output={output_dir}/ablation_{test_name}_%j.out
#SBATCH --error={output_dir}/ablation_{test_name}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --partition=jamesz

# Load any required modules
module load python/3.9
module load cuda/11.7

# Activate your conda environment if needed
source /scratch/users/salber/packages/miniconda3/etc/profile.d/conda.sh
conda activate CellVoyager

# Change to the tests directory
cd {os.path.dirname(os.path.abspath(__file__))}

# Run the single ablation test
python single_ablation_test.py \\
    --h5ad-path "{h5ad_path}" \\
    --manuscript-path "{manuscript_path}" \\
    --test-name "{test_name}" \\
    --num-analyses {num_analyses} \\
    --max-iterations {max_iterations} \\
    --output-dir "{output_dir}" \\
    {"--use-self-critique" if use_self_critique else "--no-self-critique"} \\
    {"--use-vlm" if use_VLM else "--no-vlm"} \\
    {"--use-documentation" if use_documentation else "--no-documentation"}

echo "✅ Ablation test {test_name} completed!"
"""
    
    # Write batch script to file
    script_path = os.path.join(output_dir, f"ablation_{test_name}.sh")
    with open(script_path, 'w') as f:
        f.write(batch_script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    return script_path


def create_single_test_runner():
    """Create the single_ablation_test.py script that runs one configuration."""
    
    single_test_content = '''#!/usr/bin/env python3
"""
Single ablation test runner - runs one specific configuration.
"""

import os
import sys
import json
import argparse

# Add parent directory to Python path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_tester import AgentTester


def main():
    """Run a single ablation test configuration."""
    parser = argparse.ArgumentParser(description="Run single CellVoyager ablation test")
    parser.add_argument("--h5ad-path", required=True, help="Path to .h5ad file")
    parser.add_argument("--manuscript-path", required=True, help="Path to manuscript .txt file")
    parser.add_argument("--test-name", required=True, help="Name of the test configuration")
    parser.add_argument("--num-analyses", type=int, default=2, help="Number of analyses")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max iterations per analysis")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--use-self-critique", action="store_true", help="Enable self-critique")
    parser.add_argument("--no-self-critique", action="store_true", help="Disable self-critique")
    parser.add_argument("--use-vlm", action="store_true", help="Enable VLM")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM")
    parser.add_argument("--use-documentation", action="store_true", help="Enable documentation lookup")
    parser.add_argument("--no-documentation", action="store_true", help="Disable documentation lookup")
    
    args = parser.parse_args()
    
    # Determine boolean flags
    use_self_critique = args.use_self_critique and not args.no_self_critique
    use_VLM = args.use_vlm and not args.no_vlm
    use_documentation = args.use_documentation and not args.no_documentation
    
    print(f"🚀 Running ablation test: {args.test_name}")
    print(f"   Self-critique: {'✅' if use_self_critique else '❌'}")
    print(f"   VLM: {'✅' if use_VLM else '❌'}")
    print(f"   Documentation: {'✅' if use_documentation else '❌'}")
    print(f"   Analyses: {args.num_analyses}")
    print(f"   Max iterations: {args.max_iterations}")
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return 1
    
    # Validate input files
    if not os.path.exists(args.h5ad_path):
        print(f"❌ Error: .h5ad file not found: {args.h5ad_path}")
        return 1
    
    if not os.path.exists(args.manuscript_path):
        print(f"❌ Error: manuscript file not found: {args.manuscript_path}")
        return 1
    
    try:
        # Create tester for this configuration
        tester = AgentTester(
            h5ad_path=args.h5ad_path,
            manuscript_path=args.manuscript_path,
            test_name=args.test_name,
            num_analyses=args.num_analyses,
            max_iterations=args.max_iterations,
            base_output_dir=args.output_dir
        )
        
        # Run test with specific ablation flags
        results = tester.test_agent(
            use_self_critique=use_self_critique,
            use_VLM=use_VLM,
            use_documentation=use_documentation
        )
        
        # Save individual results to output directory
        output_path = os.path.join(args.output_dir, f"results_{args.test_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        print(f"📊 {args.test_name.upper()} Results:")
        print(f"   Final Success Rate: {results['final_success_rate']:.2%}")
        print(f"   Total Failure Rate: {results['failure_rate']:.2%}")
        print(f"   Final Successful Cells: {results['final_successful_cells']}")
        print(f"   Total Failures: {results['total_failures']}")
        print(f"   Total Attempts: {results['total_code_cells_attempted']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during test {args.test_name}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''
    
    script_path = os.path.join(os.path.dirname(__file__), "single_ablation_test.py")
    with open(script_path, 'w') as f:
        f.write(single_test_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    return script_path


def submit_batch_jobs(h5ad_path, manuscript_path, num_analyses=2, max_iterations=3, output_dir=None):
    """
    Submit separate batch jobs for each ablation configuration.
    
    Returns:
        tuple: (job_ids, output_dir)
    """
    
    # Create output directory if not specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"ablation_results_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the single test runner script
    create_single_test_runner()
    
    # Define test configurations
    configurations = [
        {"test_name": "baseline", "use_self_critique": True, "use_VLM": True, "use_documentation": True},
        {"test_name": "no_vlm", "use_self_critique": True, "use_VLM": False, "use_documentation": True},
        {"test_name": "no_critique", "use_self_critique": False, "use_VLM": True, "use_documentation": True},
        #{"test_name": "no_vlm_no_critique", "use_self_critique": False, "use_VLM": False, "use_documentation": True},
        {"test_name": "no_documentation", "use_self_critique": True, "use_VLM": True, "use_documentation": False},
    ]
    
    job_ids = []
    
    print(f"🚀 Submitting CellVoyager Ablation Study Batch Jobs")
    print(f"📁 Data: {h5ad_path}")
    print(f"📄 Manuscript: {manuscript_path}")
    print(f"📂 Output directory: {output_dir}")
    print(f"🔢 {num_analyses} analyses × {max_iterations} iterations per configuration")
    print(f"🎯 {len(configurations)} configurations to test\n")
    
    for config in configurations:
        test_name = config["test_name"]
        
        # Create batch script for this configuration
        script_path = create_batch_script(
            config, h5ad_path, manuscript_path, 
            num_analyses, max_iterations, output_dir
        )
        
        # Submit the batch job
        try:
            result = subprocess.run(
                ["sbatch", script_path], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Extract job ID from sbatch output
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            
            print(f"✅ Submitted {test_name:<20} → Job ID: {job_id}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to submit {test_name}: {e}")
            print(f"   Error output: {e.stderr}")
            return None, None
    
    print(f"\n🎉 Successfully submitted {len(job_ids)} batch jobs!")
    print(f"📋 Job IDs: {', '.join(job_ids)}")
    print(f"\n💡 Monitor jobs with: squeue -u $USER")
    print(f"💡 Check results in: {output_dir}")
    print(f"💡 Collect results when complete: python {__file__} --collect-results {output_dir}")
    
    return job_ids, output_dir


def reanalyze_from_logs(output_dir):
    """Re-analyze results by processing log files with updated analysis methods."""
    
    # Find test result directories
    test_dirs = {}
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("test_results_"):
            # Extract test name from directory name: test_results_{test_name}_{timestamp}
            parts = item.split("_")
            if len(parts) >= 3:
                test_name = "_".join(parts[2:-1])  # Everything between "test_results_" and timestamp
                test_dirs[test_name] = item_path
    
    if not test_dirs:
        print("❌ No test result directories found.")
        return {}
    
    print(f"🔍 Found {len(test_dirs)} test result directories")
    
    all_results = {}
    for test_name, test_dir in test_dirs.items():
        print(f"🔄 Re-analyzing: {test_name}")
        
        try:
            # Create a tester instance for this configuration
            tester = AgentTester("dummy.h5ad", "dummy.txt", test_name)
            
            # Re-analyze with the fixed methods
            tester._analyze_notebooks_and_logs(test_dir)
            
            # Get the updated results
            results = tester.results
            all_results[test_name] = results
            
            # Save updated results to JSON file
            output_path = os.path.join(output_dir, f"results_{test_name}.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Re-analyzed {test_name}: {results['final_success_rate']:.1%} success, {results['failure_rate']:.1%} failure")
            
        except Exception as e:
            print(f"❌ Error re-analyzing {test_name}: {e}")
            # Keep old results if available
            old_result_path = os.path.join(output_dir, f"results_{test_name}.json")
            if os.path.exists(old_result_path):
                try:
                    with open(old_result_path, 'r') as f:
                        all_results[test_name] = json.load(f)
                except:
                    pass
    
    return all_results


def collect_results(output_dir, reanalyze=False):
    """Collect and analyze results from completed batch jobs."""
    
    print(f"📊 Collecting results from: {output_dir}")
    
    if reanalyze:
        print("🔄 Re-analyzing logs with updated analysis methods...")
        all_results = reanalyze_from_logs(output_dir)
    else:
        # Find all result files
        result_files = []
        for file in os.listdir(output_dir):
            if file.startswith("results_") and file.endswith(".json"):
                result_files.append(os.path.join(output_dir, file))
        
        if not result_files:
            print("❌ No result files found. Jobs may still be running.")
            return None
        
        print(f"📄 Found {len(result_files)} result files")
        
        # Load all results
        all_results = {}
        for result_file in result_files:
            test_name = os.path.basename(result_file).replace("results_", "").replace(".json", "")
            
            try:
                with open(result_file, 'r') as f:
                    results = json.load(f)
                all_results[test_name] = results
                print(f"✅ Loaded results for: {test_name}")
            except Exception as e:
                print(f"❌ Error loading {result_file}: {e}")
    
    if all_results:
        # Save comparison report
        comparison_path = save_comparison_report(all_results, os.path.join(output_dir, "ablation_comparison.json"))
        
        # Print summary
        print_summary_table(all_results)
        
        print(f"\n✅ Results collection completed!")
        print(f"📋 Comparison report: {comparison_path}")
    
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
    parser = argparse.ArgumentParser(description="Run CellVoyager ablation studies as batch jobs")
    
    # Mode selection
    parser.add_argument("--collect-results", metavar="OUTPUT_DIR", 
                       help="Collect results from completed batch jobs")
    parser.add_argument("--reanalyze", action="store_true",
                       help="Re-analyze logs with updated methods (use with --collect-results)")
    
    # Required for submission
    parser.add_argument("h5ad_path", nargs='?', help="Path to .h5ad file")
    parser.add_argument("manuscript_path", nargs='?', help="Path to manuscript .txt file")
    
    # Optional parameters
    parser.add_argument("--num-analyses", type=int, default=2, 
                       help="Number of analyses per configuration (default: 2)")
    parser.add_argument("--max-iterations", type=int, default=3,
                       help="Max iterations per analysis (default: 3)")
    parser.add_argument("--output-dir", 
                       help="Output directory for results (default: timestamped directory)")
    
    args = parser.parse_args()
    
    # Collect results mode
    if args.collect_results:
        if not os.path.exists(args.collect_results):
            print(f"❌ Error: Output directory not found: {args.collect_results}")
            return 1
        
        collect_results(args.collect_results, reanalyze=args.reanalyze)
        return 0
    
    # Submission mode - require h5ad_path and manuscript_path
    if not args.h5ad_path or not args.manuscript_path:
        print("❌ Error: h5ad_path and manuscript_path are required for job submission")
        parser.print_help()
        return 1
    
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
        # Submit batch jobs
        job_ids, output_dir = submit_batch_jobs(
            h5ad_path=args.h5ad_path,
            manuscript_path=args.manuscript_path,
            num_analyses=args.num_analyses,
            max_iterations=args.max_iterations,
            output_dir=args.output_dir
        )
        
        if job_ids:
            print(f"\n✅ Batch job submission completed successfully!")
            return 0
        else:
            print(f"\n❌ Batch job submission failed!")
            return 1
        
    except Exception as e:
        print(f"❌ Error during batch job submission: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
