"""
Autonomous ReAct AI Agent powered by NVIDIA Nemotron for 2D Optics Engine.

Executes iterative coding, bug-fixing, feature implementation, and optical
verification loops using tool calling and domain-specific knowledge.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Union

from optics_app.ai.nemotron_client import NemotronClient, ModelTier
from optics_app.ai.tools import ToolRegistry, get_default_tool_registry

SYSTEM_PROMPT = """You are an expert AI Optical Physics & Scientific Software Engineer assisting with the 2D Optics codebase.
Your goal is to investigate, write clean code, solve optical design issues, and design high-performance optical elements using physics-informed automated design sequences.

You have access to a rich set of tools to explore the repository, edit source code, execute tests, trace optical systems, run automated optical design sequences, verify 3D CAD STEP generation, and look up Thorlabs catalog specifications.

Automated Optical Design Sequence Methodology:
The 2D Optics application provides a standardized 5/6-stage physics-informed sequential design workflow:
1. Paraxial Synthesis Stage: Solves first-order optical power (Phi = 1 / f'), aperture stop diameter, conjugate factors, and initial thin lens symmetrical curvatures c1 = -c2 = Phi / (2*(n - 1)).
2. Coddington Bending Stage: Applies analytical Coddington shape factor bending to minimize primary spherical aberration S_I without altering net optical power:
   q_min = (2*(n^2 - 1) / (n + 2)) * |p|
   where p = (s' - s)/(s' + s) is the conjugate factor (p = -1 for infinite conjugate). Optimal curvatures:
   c1 = (Phi / (2*(n - 1))) * (1 + q_min)
   c2 = (Phi / (2*(n - 1))) * (q_min - 1)
3. Thick Lens Realization Stage: Converts paraxial thin lens to physical thick lens with realistic center thickness (tc >= 0.10 * D, min 1.5 mm) and edge thickness (te >= 1.5 mm), conserving Gullstrand power.
4. Glatzel Adaptive DLS Aberration Minimization Stage: Executes multi-variable Levenberg-Marquardt optimization with Glatzel (1968) parameter-specific diagonal damping matrix D_jj = p^2 * (J^T W^2 J)_jj * gamma_j, dynamically steering damping based on linear prediction discrepancy rho_j.
5. Aspherization Stage: Converts spherical lens to AsphericLens, initializing front conic constant to theoretical infinite-conjugate aplanatic null k = -n^2, and optimizes conic and 4th-order polynomial coefficient (alpha_4) to null residual spherical aberration and achieve diffraction-limited spot sizes (< 10 um).
6. Tolerancing & Manufacturing Yield Stage: Executes sensitivity analysis and Monte Carlo simulation (>= 90% yield) across standard commercial fabrication tolerances (radius, thickness, index, tilt, decenter).

Available AI Sequence Tools:
- `replace_lens_from_thorlabs_query(element, focal_length_mm, diameter_mm=None, coating_code=None)`: Resolves a named lens, selects the best matching Thorlabs achromat, and atomically replaces its complete surface group. Treat an unlabeled value such as "30 mm achromat" as focal length; only use diameter_mm when the user explicitly says diameter or Ø.
- `optimize_focus_at_surface(target_surface, adjust_surface=None, min_gap_mm=0.01, max_gap_mm=None)`: Focuses rays at a named/indexed surface by varying only one preceding AIR gap, reports before/after RMS spot size, and preserves the original model if no improvement is found.
- `inspect_optical_model_sanity()`: Audits the active model for uncommon or likely accidental geometry before diagnosis or modification.
- `run_optical_design_sequence(target_efl, target_diameter, target_f_number=None, glass="N-BK7")`: Executes complete 5/6-stage end-to-end design sequence.
- `run_lens_bending_optimization(efl, diameter, glass="N-BK7")`: Runs analytical Coddington shape factor bending.
- `run_aspheric_optimization(efl, diameter, glass="N-BK7")`: Runs progressive design through aspherization to achieve diffraction-limited spot size.
- `run_tolerancing_sequence(efl, diameter, trials=500)`: Runs Monte Carlo manufacturing yield simulation and sensitivity analysis.

Key Guidelines:
1. Always explore files and verify requirements before making modifications.
2. Maintain mathematical & physical accuracy (coordinate breaks, surface normals, Snell's law, grating diffraction equations m*G*lambda = sin(alpha) + sin(beta), aspheric sag, apodization, Coddington invariants).
3. Before diagnosing failed propagation or modifying a loaded optical model, run `inspect_optical_model_sanity` and explicitly address its warnings. Treat large tilts on refractive surfaces and inconsistent tilts within one lens group as likely accidental until verified.
4. Whenever modifying optical algorithms or designs, test them using `run_command` (e.g. running pytest or diagnostic scripts) or `trace_optical_design` to ensure all rays pass and spot sizes meet specifications.
5. Leverage the automated optical design sequence tools for synthesizing and optimizing optical elements.
6. For requests to focus at a particular image or detector surface, use `optimize_focus_at_surface`; do not substitute aperture tuning for axial focus optimization.
7. When finished, provide a concise summary of what was solved and verified.
"""


class NemotronAgent:
    """
    Autonomous ReAct AI Agent for 2D Optics development and verification.
    """

    def __init__(
        self,
        client: Optional[NemotronClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = 15,
        tier: Union[ModelTier, str] = ModelTier.HIGH,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if client is not None:
            self.client = client
        else:
            self.client = NemotronClient(
                api_key=api_key,
                tier=tier,
                model=model,
            )

        self.tools = tool_registry or get_default_tool_registry()
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.session_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset_session(self) -> None:
        """Clear conversation history and reset to system prompt."""
        self.session_messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def run(self, task: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute an autonomous multi-turn ReAct tool loop for a given task or goal.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]

        if verbose:
            print(f"\n[NemotronAgent] Starting Task (Model: {self.client.model}, Tier: {self.client.tier.value})")
            print(f"[NemotronAgent] Prompt: {task}\n" + "-" * 70)

        step_logs: List[Dict[str, Any]] = []
        final_answer = ""

        for iteration in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n>>> Iteration {iteration}/{self.max_iterations} (Thinking...)")

            try:
                response = self.client.chat_completion(
                    messages=messages,
                    tools=self.tools.schemas,
                    tool_choice="auto",
                )
            except Exception as e:
                err_msg = f"API Error in Agent loop: {e}"
                if verbose:
                    print(f"[Error] {err_msg}")
                return {
                    "success": False,
                    "final_answer": err_msg,
                    "iterations": iteration,
                    "steps": step_logs,
                }

            content = response.get("content") or ""
            tool_calls = response.get("tool_calls", [])

            # Print reasoning thought content if any
            if content and verbose:
                print(f"[Thought/Response]\n{content.strip()}")

            # Append assistant message to context
            asst_msg: Dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                asst_msg["tool_calls"] = tool_calls
            messages.append(asst_msg)

            # If no tool calls, task is complete
            if not tool_calls:
                final_answer = content
                if verbose:
                    print(f"\n[NemotronAgent] Goal completed in {iteration} iteration(s).")
                break

            # Execute all requested tool calls
            for tc in tool_calls:
                func_name = tc.get("function", {}).get("name", "")
                args_raw = tc.get("function", {}).get("arguments", "{}")
                call_id = tc.get("id", f"call_{int(time.time()*1000)}")

                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = {}

                if verbose:
                    print(f"\n[Tool Call] {func_name}({', '.join(f'{k}={repr(v)}' for k, v in args.items())})")

                # Execute tool
                tool_output = self.tools.execute(func_name, args)
                tool_str = str(tool_output)

                if verbose:
                    preview = tool_str[:300] + ("..." if len(tool_str) > 300 else "")
                    print(f"[Tool Output]\n{preview}")

                step_logs.append({
                    "iteration": iteration,
                    "tool": func_name,
                    "arguments": args,
                    "output": tool_str,
                })

                # Append tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": tool_str,
                })

        return {
            "success": bool(final_answer),
            "final_answer": final_answer or "Reached maximum iterations without final answer.",
            "iterations": iteration,
            "steps": step_logs,
        }

    def chat_turn(self, user_message: str, verbose: bool = True) -> str:
        """
        Execute one conversational turn in interactive session mode.
        """
        self.session_messages.append({"role": "user", "content": user_message})

        for _ in range(self.max_iterations):
            try:
                response = self.client.chat_completion(
                    messages=self.session_messages,
                    tools=self.tools.schemas,
                    tool_choice="auto",
                )
            except Exception as e:
                return f"Error communicating with NVIDIA NIM API: {e}"

            content = response.get("content") or ""
            tool_calls = response.get("tool_calls", [])

            asst_msg: Dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                asst_msg["tool_calls"] = tool_calls
            self.session_messages.append(asst_msg)

            if not tool_calls:
                return content

            # Execute tool calls
            for tc in tool_calls:
                func_name = tc.get("function", {}).get("name", "")
                args_raw = tc.get("function", {}).get("arguments", "{}")
                call_id = tc.get("id", f"call_{int(time.time()*1000)}")

                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = {}

                if verbose:
                    print(f"\n[Agent Tool] {func_name}({', '.join(f'{k}={repr(v)}' for k, v in args.items())})")

                tool_output = self.tools.execute(func_name, args)
                tool_str = str(tool_output)

                if verbose:
                    preview = tool_str[:250] + ("..." if len(tool_str) > 250 else "")
                    print(f"[Output] {preview}")

                self.session_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": tool_str,
                })

        return "Reached maximum iteration limit for this turn."
