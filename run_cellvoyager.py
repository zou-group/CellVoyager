"""
CLI entry point for CellVoyager analysis with configurable execution module.
"""
import json
import os
import argparse
from pathlib import Path
from cellvoyager.agent import AnalysisAgentV2


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
        help="Path to context summary text file (dataset summary, prior analyses, focus directions, bio background).",
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
        default="claude-sonnet-4-6",
        help="LLM model for hypothesis generation — any OpenAI or Anthropic model (e.g. o3-mini, gpt-4o, claude-sonnet-4-5). Default: o3-mini",
    )
    parser.add_argument(
        "--execution-model",
        default=None,
        help="Anthropic model for the Claude execution agent (e.g. claude-sonnet-4-6, claude-opus-4-6). Defaults to the Claude Code CLI default.",
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
        "--output-dir",
        default=None,
        help="Exact output directory (overrides output-home + timestamp). Used by GUI to track a specific run.",
    )
    parser.add_argument(
        "--log-home",
        default=".",
        help="Home directory for logs (default: current directory)",
    )
    parser.add_argument(
        "--prompt-dir",
        default=None,
        help="Directory containing prompt templates (default: built-in)",
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
        "--deepresearch",
        action="store_true",
        help="Enable DeepResearch background generation (off by default)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: agent pauses after each interpretation step. Edit the notebook in Jupyter, then press Enter and optionally type feedback for the agent.",
    )
    parser.add_argument(
        "--intervene-every",
        type=int,
        default=1,
        metavar="N",
        help="When interactive: show edit screen every N steps (default 1 = every step).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: re-run a completed analysis to restore kernel state, then enter interactive mode.",
    )
    parser.add_argument(
        "--resume-output-dir",
        default=None,
        help="Output dir for resume (required when --resume).",
    )
    parser.add_argument(
        "--resume-analysis-idx",
        type=int,
        default=1,
        help="Analysis index (1-based) to resume (default 1).",
    )
    parser.add_argument(
        "--resume-intervene-every",
        type=int,
        default=None,
        help="Override intervene_every for resume (e.g. 999 to run to completion without pausing).",
    )

    args = parser.parse_args()

    # Check if OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return 1

    # Resume mode: handle separately before normal validation
    if args.resume:
        if not args.resume_output_dir:
            print("❌ Error: --resume-output-dir required when using --resume")
            return 1
        out_dir = Path(args.resume_output_dir)
        config_path = out_dir / ".run_config.json"
        if not config_path.exists():
            print("❌ Error: No run config found. Run config is created when starting from the GUI.")
            return 1
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        notebook_path = out_dir / f"{cfg['analysis_name']}_analysis_{args.resume_analysis_idx}.ipynb"
        if not notebook_path.exists():
            print(f"❌ Error: Notebook not found: {notebook_path}")
            return 1
        anthropic_api_key = args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            print("❌ Error: ANTHROPIC_API_KEY required for resume (claude execution)")
            return 1
        exec_mode = cfg.get("execution_mode", "claude")
        if exec_mode != "claude":
            print("❌ Error: Resume only supports execution_mode=claude")
            return 1
        intervene = args.resume_intervene_every if args.resume_intervene_every is not None else cfg.get("intervene_every", 1)
        print("🔄 Resume mode: restoring kernel state, then entering interactive mode...")
        resume_exec_kwargs = {
            "jupyter_port": args.jupyter_port,
            "jupyter_token": args.jupyter_token,
            "auto_start_jupyter": not args.no_auto_start_jupyter,
            "stop_jupyter_on_complete": args.stop_jupyter_on_complete,
            "interactive_mode": True,
            "intervene_every": intervene,
            "execution_model": args.execution_model or cfg.get("execution_model"),
        }
        agent = AnalysisAgentV2(
            h5ad_path=cfg["h5ad_path"],
            paper_summary_path=cfg["paper_path"],
            openai_api_key=openai_api_key,
            model_name=cfg.get("model_name", args.model_name),
            analysis_name=cfg["analysis_name"],
            num_analyses=1,
            max_iterations=cfg.get("max_iterations", args.max_iterations),
            prompt_dir=args.prompt_dir,
            output_home=args.output_home,
            output_dir=str(out_dir),
            log_home=args.log_home,
            use_self_critique=not args.no_self_critique,
            use_VLM=not args.no_vlm,
            use_documentation=not args.no_documentation,
            log_prompts=args.log_prompts,
            max_fix_attempts=args.max_fix_attempts,
            use_deepresearch_background=cfg.get("use_deepresearch", False),
            execution_mode=exec_mode,
            anthropic_api_key=anthropic_api_key,
            **resume_exec_kwargs,
        )
        try:
            agent.run_resume(str(notebook_path), args.resume_analysis_idx - 1)
            print("\n✅ Resume session complete!")
            return 0
        except KeyboardInterrupt:
            print("\n⚠️ Resume interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Error during resume: {e}")
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
            "intervene_every": args.intervene_every,
            "execution_model": args.execution_model,
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
        output_dir=args.output_dir,
        log_home=args.log_home,
        use_self_critique=not args.no_self_critique,
        use_VLM=not args.no_vlm,
        use_documentation=not args.no_documentation,
        log_prompts=args.log_prompts,
        max_fix_attempts=args.max_fix_attempts,
        use_deepresearch_background=args.deepresearch,
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
