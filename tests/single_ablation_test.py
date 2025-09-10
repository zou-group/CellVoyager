#!/usr/bin/env python3
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
