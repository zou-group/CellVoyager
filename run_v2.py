"""
Run CellVoyager analysis using agent_v2 with configurable execution module.
"""
import os
import argparse
from agent_v2 import AnalysisAgentV2


def main():
    parser = argparse.ArgumentParser(description="Run CellVoyager analysis agent (v2 with modular execution)")

    # REQUIRED arguments
    parser.add_argument(
        "--h5ad-path",
        default=os.path.join(os.getcwd(), "example/covid19.h5ad"),
        help="Path to the .h5ad file (default: example/covid19.h5ad)",
    )
    parser.add_argument(
        "--paper-path",
        default=os.path.join(os.getcwd(), "example/covid19_summary.txt"),
        help="Path to the paper summary text file (default: example/covid19_summary.txt)",
    )
    parser.add_argument(
        "--analysis-name",
        default="covid19",
        help="Name for the analysis (default: covid19)",
    )

    # Execution module selection
    parser.add_argument(
        "--execution-mode",
        choices=["legacy", "claude"],
        default="claude",
        help="Execution module: 'legacy' = IdeaExecutor (programmatic kernel), "
        "'claude' = ClaudeJupyterExecutor (live Jupyter + Claude Agent SDK) (default: legacy)",
    )
    parser.add_argument(
        "--anthropic-api-key",
        default=None,
        help="Anthropic API key for Claude execution mode (or set ANTHROPIC_API_KEY env)",
    )

    # Jupyter options (for execution_mode=claude)
    parser.add_argument(
        "--jupyter-port",
        type=int,
        default=8899,
        help="Jupyter Lab port when using claude execution (default: 8888)",
    )
    parser.add_argument(
        "--jupyter-token",
        default=None,
        help="Jupyter token (default: CELLVOYAGER or JUPYTER_TOKEN env)",
    )
    parser.add_argument(
        "--no-auto-start-jupyter",
        action="store_true",
        help="Do not auto-start Jupyter; assume it is already running",
    )
    parser.add_argument(
        "--stop-jupyter-on-complete",
        action="store_true",
        help="Stop Jupyter when analysis completes (default: keep running to inspect results)",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--model-name",
        default="o3-mini",
        help="OpenAI model name to use (default: o3-mini)",
    )
    parser.add_argument(
        "--num-analyses",
        type=int,
        default=1,
        help="Number of analyses to run (default: 8)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Maximum iterations per analysis (default: 6)",
    )
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        default=3,
        help="Maximum fix attempts per step (default: 3)",
    )
    parser.add_argument(
        "--output-home",
        default=".",
        help="Home directory for outputs (default: current directory)",
    )
    parser.add_argument(
        "--log-home",
        default=".",
        help="Home directory for logs (default: current directory)",
    )
    parser.add_argument(
        "--prompt-dir",
        default="prompts",
        help="Directory containing prompt templates (default: prompts)",
    )

    # Boolean flags
    parser.add_argument(
        "--no-self-critique",
        action="store_true",
        help="Disable self-critique functionality",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable Vision Language Model functionality",
    )
    parser.add_argument(
        "--no-documentation",
        action="store_true",
        help="Disable documentation functionality",
    )
    parser.add_argument(
        "--log-prompts",
        action="store_true",
        help="Enable prompt logging",
    )
    parser.add_argument(
        "--no-deepresearch",
        action="store_true",
        help="Disable DeepResearch background generation",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: agent pauses after each interpretation step. Edit the notebook in Jupyter, then press Enter and optionally type feedback for the agent.",
    )

    args = parser.parse_args()

    # Check if OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return 1

    # Check Anthropic API key when using claude execution
    if args.execution_mode == "claude":
        anthropic_api_key = args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            print("❌ Error: ANTHROPIC_API_KEY required for --execution-mode claude")
            print("Set the env var or pass --anthropic-api-key")
            return 1

    # Check if required files exist
    if not os.path.exists(args.h5ad_path):
        print(f"❌ Error: H5AD file not found: {args.h5ad_path}")
        return 1

    if not os.path.exists(args.paper_path):
        print(f"❌ Error: Paper summary file not found: {args.paper_path}")
        return 1

    print("🚀 Starting CellVoyager Analysis Agent (v2)")
    print(f"   H5AD file: {args.h5ad_path}")
    print(f"   Paper summary: {args.paper_path}")
    print(f"   Analysis name: {args.analysis_name}")
    print(f"   Model: {args.model_name}")
    print(f"   Number of analyses: {args.num_analyses}")
    print(f"   Max iterations: {args.max_iterations}")
    print(f"   Execution mode: {args.execution_mode}")
    print(f"   Self-critique: {'❌' if args.no_self_critique else '✅'}")
    print(f"   VLM: {'❌' if args.no_vlm else '✅'}")
    print(f"   Documentation: {'❌' if args.no_documentation else '✅'}")
    if args.execution_mode == "claude":
        print(f"   Jupyter port: {args.jupyter_port}")
        print(f"   Auto-start Jupyter: {'❌' if args.no_auto_start_jupyter else '✅'}")
        print(f"   Interactive mode: {'✅' if args.interactive else '❌'}")
    print()

    # Execution kwargs for Claude mode
    execution_kwargs = {}
    if args.execution_mode == "claude":
        execution_kwargs = {
            "jupyter_port": args.jupyter_port,
            "jupyter_token": args.jupyter_token,
            "auto_start_jupyter": not args.no_auto_start_jupyter,
            "stop_jupyter_on_complete": args.stop_jupyter_on_complete,
            "interactive_mode": args.interactive,
        }

    agent = AnalysisAgentV2(
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
        max_fix_attempts=args.max_fix_attempts,
        use_deepresearch_background=not args.no_deepresearch,
        execution_mode=args.execution_mode,
        anthropic_api_key=args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
        **execution_kwargs,
    )

    try:
        print("🔬 Running analyses...")
        agent.run()
        print("\n✅ Analysis complete!")
        return 0
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
