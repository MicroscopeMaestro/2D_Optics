#!/usr/bin/env python3
"""
NVIDIA Nemotron Autonomous Agent CLI for 2D Optics Engine.

Usage:
  python nemotron_agent.py "Investigate and fix A230-C aspheric lens prescription"
  python nemotron_agent.py --goal "Optimize NIR spectrometer layout for 600 lpm"
  python nemotron_agent.py chat
  python nemotron_agent.py verify C:/Users/Juan/Desktop/second_mirror_spectrometer_NIR.json
  python nemotron_agent.py --tier medium "Run all unit tests and summarize results"
"""

import os
import sys
import argparse

# Ensure UTF-8 output encoding across all Windows terminals
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optics_app.ai.nemotron_client import NemotronClient, ModelTier, TIER_MODEL_MAP
from optics_app.ai.tools import get_default_tool_registry
from optics_app.ai.agent import NemotronAgent


def print_banner():
    banner = """
========================================================================
    NVIDIA Nemotron Autonomous AI Agent for 2D Optics Engine
========================================================================
"""
    print(banner)


def handle_verify(json_path: str, agent: NemotronAgent):
    """Run direct automated optical diagnostics on a design file."""
    print(f"\n[Nemotron Agent] Verifying Optical Design File: {json_path}")
    if not os.path.exists(json_path):
        print(f"[Error] File '{json_path}' does not exist.")
        return 1

    tools = agent.tools
    trace_res = tools.execute("trace_optical_design", {"json_path": json_path})
    print("\n" + "=" * 50)
    print(trace_res)

    cad_res = tools.execute("verify_cad_export", {"json_path": json_path})
    print("\n" + "=" * 50)
    print(cad_res)
    print("=" * 50 + "\n")
    return 0


def handle_chat(agent: NemotronAgent):
    """Start an interactive terminal chat session with Nemotron."""
    print(f"\n[Interactive Chat Mode] Model: {agent.client.model} (Tier: {agent.client.tier.value})")
    print("Type your message, or 'exit' / 'quit' / 'reset' to manage the session.\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting Nemotron chat session. Goodbye!")
                break
            if user_input.lower() in ["reset", "clear"]:
                agent.reset_session()
                print("[Session reset to system prompt]\n")
                continue

            print("\nNemotron Thinking...")
            reply = agent.chat_turn(user_input, verbose=True)
            print(f"\nNemotron >\n{reply}\n" + "-" * 70)

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Nemotron Autonomous Coding and Verification Agent for 2D Optics."
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Task prompt, goal, or sub-command ('chat', 'verify')."
    )
    parser.add_argument(
        "target_file",
        nargs="?",
        default=None,
        help="Target file for 'verify' sub-command."
    )
    parser.add_argument(
        "--goal",
        dest="goal_task",
        default=None,
        help="Execute an autonomous goal resolution loop."
    )
    parser.add_argument(
        "--tier",
        choices=["high", "medium"],
        default="high",
        help="Model tier ('high': Nemotron 70B, 'medium': LLaMA 8B / Nemotron Mini)."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Explicit model identifier override on NVIDIA NIM API."
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NVIDIA API Key (defaults to NVIDIA_API_KEY env var or .env)."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Maximum agent tool loop iterations (default: 15)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress intermediate tool call logs."
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Optional path to save markdown execution summary."
    )

    args = parser.parse_args()

    # Build Agent
    agent = NemotronAgent(
        api_key=args.api_key,
        tier=args.tier,
        model=args.model,
        max_iterations=args.max_iterations,
    )

    # Sub-command: verify
    if args.task == "verify":
        if not args.target_file:
            print("[Error] Please specify the .json optical design file to verify. Example: python nemotron_agent.py verify path/to/design.json")
            return 1
        return handle_verify(args.target_file, agent)

    # Sub-command: chat
    if args.task == "chat":
        print_banner()
        handle_chat(agent)
        return 0

    # Task / Goal Execution
    prompt = args.goal_task or args.task
    if prompt:
        print_banner()
        result = agent.run(prompt, verbose=not args.quiet)

        print("\n" + "=" * 70)
        print("FINAL RESULT SUMMARY:")
        print("=" * 70)
        print(result["final_answer"])
        print("=" * 70)

        # Write markdown report if requested
        if args.output_report:
            report_lines = [
                f"# Nemotron Agent Execution Report",
                f"- **Model**: `{agent.client.model}` ({agent.client.tier.value} tier)",
                f"- **Task**: {prompt}",
                f"- **Iterations**: {result['iterations']}",
                f"- **Success**: {result['success']}",
                "",
                "## Final Output",
                result["final_answer"],
                "",
                "## Execution Steps",
            ]
            for step in result.get("steps", []):
                report_lines.append(f"### Iteration {step['iteration']} - Tool `{step['tool']}`")
                report_lines.append(f"**Arguments**: `{json.dumps(step['arguments'])}`")
                report_lines.append(f"```\n{step['output']}\n```\n")

            with open(args.output_report, "w", encoding="utf-8") as fp:
                fp.write("\n".join(report_lines))
            print(f"\n[Report saved to {args.output_report}]")

        return 0 if result["success"] else 1

    # No arguments provided
    print_banner()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
