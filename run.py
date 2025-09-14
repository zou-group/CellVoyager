import os
import json
import argparse
import openai
from dotenv import load_dotenv
from agent import AnalysisAgent
from notebook_generator import generate_notebook

# Load environment variables from .env file
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run CellVoyager analysis agent")
    
    # REQUIRED arguments
    parser.add_argument("--h5ad-path", 
                       default=os.path.join(os.getcwd(), "example/covid19.h5ad"),
                       help="Path to the .h5ad file (default: example/covid19.h5ad)")
    
    parser.add_argument("--paper-path", 
                       default=os.path.join(os.getcwd(), "example/covid19_summary.txt"),
                       help="Path to the paper summary text file (default: example/covid19_summary.txt)")
    
    parser.add_argument("--analysis-name", 
                       default="covid19",
                       help="Name for the analysis (default: covid19)")
    
    # Optional arguments with defaults
    parser.add_argument("--model-name", 
                       default="o3-mini",
                       help="OpenAI model name to use (default: o3-mini)")
    
    parser.add_argument("--num-analyses", 
                       type=int, 
                       default=8,
                       help="Number of analyses to run (default: 8)")
    
    parser.add_argument("--max-iterations", 
                       type=int, 
                       default=6,
                       help="Maximum iterations per analysis (default: 6)")
    
    parser.add_argument("--max-fix-attempts", 
                       type=int, 
                       default=3,
                       help="Maximum fix attempts per step (default: 3)")
    
    parser.add_argument("--output-home", 
                       default=".",
                       help="Home directory for outputs (default: current directory)")
    
    parser.add_argument("--log-home", 
                       default=".",
                       help="Home directory for logs (default: current directory)")
    
    parser.add_argument("--prompt-dir", 
                       default="prompts",
                       help="Directory containing prompt templates (default: prompts)")
    
    # Boolean flags
    parser.add_argument("--no-self-critique", 
                       action="store_true",
                       help="Disable self-critique functionality")
    
    parser.add_argument("--no-vlm", 
                       action="store_true",
                       help="Disable Vision Language Model functionality")
    
    parser.add_argument("--no-documentation", 
                       action="store_true",
                       help="Disable documentation functionality")
    
    parser.add_argument("--log-prompts", 
                       action="store_true",
                       help="Enable prompt logging")
    
    args = parser.parse_args()
    
    # Check if OpenAI API key is available
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return 1
    
    # Check if required files exist
    if not os.path.exists(args.h5ad_path):
        print(f"❌ Error: H5AD file not found: {args.h5ad_path}")
        return 1
    
    if not os.path.exists(args.paper_path):
        print(f"❌ Error: Paper summary file not found: {args.paper_path}")
        return 1
    
    print("🚀 Starting CellVoyager Analysis Agent")
    print(f"   H5AD file: {args.h5ad_path}")
    print(f"   Paper summary: {args.paper_path}")
    print(f"   Analysis name: {args.analysis_name}")
    print(f"   Model: {args.model_name}")
    print(f"   Number of analyses: {args.num_analyses}")
    print(f"   Max iterations: {args.max_iterations}")
    print(f"   Self-critique: {'❌' if args.no_self_critique else '✅'}")
    print(f"   VLM: {'❌' if args.no_vlm else '✅'}")
    print(f"   Documentation: {'❌' if args.no_documentation else '✅'}")
    print()
    
    # Initialize the agent
    agent = AnalysisAgent(
        h5ad_path=args.h5ad_path,
        paper_summary_path=args.paper_path,
        openai_api_key=openai_api_key,
        model_name=args.model_name,
        analysis_name=args.analysis_name,
        num_analyses=args.num_analyses,
        max_iterations=args.max_iterations,
        prompt_dir=args.prompt_dir,
        output_home=args.output_home,
        log_home=args.log_home,
        use_self_critique=not args.no_self_critique,
        use_VLM=not args.no_vlm,
        use_documentation=not args.no_documentation,
        log_prompts=args.log_prompts,
        max_fix_attempts=args.max_fix_attempts
    )
    
    try:
        # Run the analysis
        print("🔬 Running analyses...")
        agent.run()
        print("✅ Analysis complete!")
            
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    finally:
        # Clean up agent resources
        if hasattr(agent, 'cleanup'):
            agent.cleanup()


if __name__ == "__main__":
    exit(main())
