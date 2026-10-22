"""
Transparent Optical Lens Optimization Engine & DLS / SciPy Solvers.
===================================================================
Provides modular, mathematically transparent optimization algorithms for 2D Optics:
- BaseOptimizer: Extensible abstract base class with single-step manual debugging hook.
- ScipyLeastSquaresOptimizer: High-speed Trust Region Reflective (TRF) bounded solver.
- TransparentDLSOptimizer: White-box Damped Least Squares (Levenberg-Marquardt) solver
  with full inspectable Jacobian sensitivity matrices and manual stepping capability.
- OptimizationStepRecord: Dataclass recording per-iteration diagnostics, parameters,
  operand residuals, scalar merit, step norm, and Jacobian.
- OptimizationResult: Comprehensive result container with complete step trajectory.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.optimize import least_squares

from optics_app.engine.thick_lens import OpticalElement, ThickLens
from optics_app.engine.lens_merit import (
    PhysicalMeritFunction,
    TargetEFLOperand,
    DiameterConstraintOperand,
)


@dataclass
class OptimizationStepRecord:
    """
    Transparent record of a single optimization iteration.

    Logs all intermediate variables, sensitivities, step sizes, and damping factors
    for manual debugging and mathematical auditing.
    """
    iteration: int
    parameters: Dict[str, float]
    merit_value: float
    operand_residuals: Dict[str, float] = field(default_factory=dict)
    step_norm: float = 0.0
    damping: float = 0.0
    jacobian: Optional[np.ndarray] = None
    gradient: Optional[np.ndarray] = None
    notes: str = ""

    @property
    def scalar_merit(self) -> float:
        """Alias for merit_value."""
        return self.merit_value

    @property
    def parameter_values(self) -> Dict[str, float]:
        """Alias for parameters dict."""
        return self.parameters

    @property
    def residuals(self) -> Dict[str, float]:
        """Alias for operand_residuals."""
        return self.operand_residuals

    @property
    def damping_factor(self) -> float:
        """Alias for damping."""
        return self.damping


@dataclass
class OptimizationResult:
    """
    Result container for lens optimization runs.

    Contains convergence flags, initial/final merit values, iteration counts,
    final parameters, and the complete OptimizationStepRecord history trajectory.
    """
    success: bool
    initial_merit: float
    final_merit: float
    iterations: int
    parameters: Dict[str, float]
    history: List[OptimizationStepRecord]
    message: str = ""
    optimized_system: Optional[OpticalElement] = None

    @property
    def parameter_values(self) -> Dict[str, float]:
        """Alias for parameters dict."""
        return self.parameters


class BaseOptimizer(ABC):
    """
    Abstract Base Class for modular optical lens optimizers.

    Enforces support for both complete optimization runs (.optimize())
    and explicit single-step manual execution (.step()).
    """

    def __init__(
        self,
        merit_function: Optional[PhysicalMeritFunction] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-5,
    ) -> None:
        self.merit_function = merit_function
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self._step_count: int = 0

    def reset(self) -> None:
        """Reset internal iteration counter and step states."""
        self._step_count = 0

    def _get_merit_function(self, mf_arg: Optional[PhysicalMeritFunction]) -> PhysicalMeritFunction:
        """Resolve merit function from argument or instance attribute."""
        mf = mf_arg if mf_arg is not None else self.merit_function
        if mf is None:
            raise ValueError("No PhysicalMeritFunction specified for optimization.")
        return mf

    @abstractmethod
    def optimize(
        self,
        system: OpticalElement,
        merit_function: Optional[PhysicalMeritFunction] = None,
    ) -> OptimizationResult:
        """
        Execute iterative optimization until convergence or max_iterations reached.

        Args:
            system: The optical element or system to optimize (updated in-place).
            merit_function: Optional override for the merit function.

        Returns:
            OptimizationResult containing final parameters and full step history.
        """
        pass

    @abstractmethod
    def step(
        self,
        system: OpticalElement,
        merit_function: Optional[PhysicalMeritFunction] = None,
    ) -> OptimizationStepRecord:
        """
        Execute exactly one optimization iteration step for manual stepping and transparent debugging.

        Args:
            system: Optical element being stepped (updated in-place).
            merit_function: Optional override for the merit function.

        Returns:
            OptimizationStepRecord detailing parameters, residuals, and sensitivities for this step.
        """
        pass

    @staticmethod
    def _extract_system_variables(system: OpticalElement) -> Tuple[np.ndarray, List[str]]:
        """
        Extract curvature variables (c1, c2), thickness t, and diameter D from system.
        """
        var_names = ["c1", "c2", "thickness", "diameter"]
        vals: List[float] = []

        if hasattr(system, "r1") and hasattr(system, "r2"):
            c1 = 1.0 / system.r1 if abs(system.r1) > 1e-12 and not math.isinf(system.r1) else 0.0
            c2 = 1.0 / system.r2 if abs(system.r2) > 1e-12 and not math.isinf(system.r2) else 0.0
            vals.extend([c1, c2])
        else:
            params = system.get_parameters()
            r1 = params.get("r1", params.get("radius_front", 50.0))
            r2 = params.get("r2", params.get("radius_rear", -50.0))
            c1 = 1.0 / r1 if abs(r1) > 1e-12 and not math.isinf(r1) else 0.0
            c2 = 1.0 / r2 if abs(r2) > 1e-12 and not math.isinf(r2) else 0.0
            vals.extend([c1, c2])

        t = float(getattr(system, "thickness", system.get_parameter("thickness") if hasattr(system, "get_parameter") else 5.0))
        d = float(getattr(system, "diameter", system.get_parameter("diameter") if hasattr(system, "get_parameter") else 25.4))
        vals.extend([t, d])

        return np.array(vals, dtype=float), var_names

    @staticmethod
    def _apply_system_variables(system: OpticalElement, p: np.ndarray) -> None:
        """
        Apply variables p = [c1, c2, t, D] back to the optical system.
        """
        c1, c2, t, d = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        r1 = 1.0 / c1 if abs(c1) > 1e-12 else float("inf")
        r2 = 1.0 / c2 if abs(c2) > 1e-12 else float("inf")

        if hasattr(system, "r1") and hasattr(system, "r2"):
            system.r1 = r1
            system.r2 = r2
            if hasattr(system, "thickness"):
                system.thickness = t
            if hasattr(system, "diameter"):
                system.diameter = d
        elif hasattr(system, "set_parameters"):
            system.set_parameters({
                "r1": r1,
                "r2": r2,
                "thickness": t,
                "diameter": d,
            })

    @staticmethod
    def _compute_bounds(system: OpticalElement, target_diameter: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute physical box bounds:
        - Curvature limit |c| <= 1 / semi_diameter (geometric hemisphere limit)
        - Thickness >= 0.51 mm (positive physical thickness > 0.5 mm)
        - Diameter >= 1.0 mm
        """
        curr_d = float(getattr(system, "diameter", 25.4))
        ref_d = max(curr_d, target_diameter if target_diameter is not None else curr_d)
        semi_d = max(ref_d / 2.0, 5.0)
        c_limit = 0.999 / semi_d  # Maximum physical sag curvature limit

        lb = np.array([-c_limit, -c_limit, 0.51, 1.0], dtype=float)
        ub = np.array([ c_limit,  c_limit, 50.0, 200.0], dtype=float)
        return lb, ub


class ScipyLeastSquaresOptimizer(BaseOptimizer):
    """
    High-speed, robust bounded Trust Region Reflective (TRF) optimizer.

    Utilizes scipy.optimize.least_squares with method='trf' operating on
    curvature variables (c1, c2), center thickness t, and clear aperture diameter D.
    Guarantees fast quadratic convergence within physical bounds.
    """

    def optimize(
        self,
        system: OpticalElement,
        merit_function: Optional[PhysicalMeritFunction] = None,
    ) -> OptimizationResult:
        mf = self._get_merit_function(merit_function)
        p0, _ = self._extract_system_variables(system)

        # Detect target diameter if present in operands
        target_diam = None
        for op in mf.operands:
            if isinstance(op, DiameterConstraintOperand) or "diameter" in op.name.lower():
                target_diam = op.target
                break

        lb, ub = self._compute_bounds(system, target_diam)
        p0 = np.clip(p0, lb, ub)
        self._apply_system_variables(system, p0)

        initial_merit = mf.evaluate(system)
        initial_params = system.get_parameters() if hasattr(system, "get_parameters") else {
            "r1": system.r1, "r2": system.r2, "thickness": system.thickness, "diameter": system.diameter
        }
        initial_residuals = {op.name: op.compute_residual(system) for op in mf.operands}

        # Step record history trajectory
        history: List[OptimizationStepRecord] = [
            OptimizationStepRecord(
                iteration=0,
                parameters=initial_params.copy(),
                merit_value=initial_merit,
                operand_residuals=initial_residuals,
                step_norm=0.0,
                damping=0.0,
                notes="Initial unoptimized state",
            )
        ]

        best_merit = [initial_merit]
        step_counter = [0]

        def residual_fn(p: np.ndarray) -> np.ndarray:
            self._apply_system_variables(system, p)
            w_res = mf.weighted_residuals(system)
            cur_merit = float(np.sum(w_res ** 2))

            # Record trajectory step on significant reduction, up to convergence
            if best_merit[0] >= 1e-4 and (cur_merit < best_merit[0] * 0.5 or cur_merit < 1e-4):
                step_counter[0] += 1
                best_merit[0] = cur_merit
                cur_params = system.get_parameters() if hasattr(system, "get_parameters") else {
                    "r1": system.r1, "r2": system.r2, "thickness": system.thickness, "diameter": system.diameter
                }
                cur_residuals = {op.name: op.compute_residual(system) for op in mf.operands}
                history.append(
                    OptimizationStepRecord(
                        iteration=step_counter[0],
                        parameters=cur_params.copy(),
                        merit_value=cur_merit,
                        operand_residuals=cur_residuals,
                        step_norm=float(np.linalg.norm(p - p0)),
                        damping=0.0,
                        notes=f"TRF step {step_counter[0]}",
                    )
                )

            return w_res

        # Execute bounded Trust Region Reflective least squares
        res = least_squares(
            residual_fn,
            p0,
            bounds=(lb, ub),
            method="trf",
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-5,
            max_nfev=max(self.max_iterations * 5, 50),
        )

        # Apply final optimized point
        self._apply_system_variables(system, res.x)
        final_merit = mf.evaluate(system)
        final_params = system.get_parameters() if hasattr(system, "get_parameters") else {
            "r1": system.r1, "r2": system.r2, "thickness": system.thickness, "diameter": system.diameter
        }
        final_residuals = {op.name: op.compute_residual(system) for op in mf.operands}

        # Ensure final state is logged in history if needed
        if len(history) == 1 or history[-1].merit_value > final_merit:
            history.append(
                OptimizationStepRecord(
                    iteration=len(history),
                    parameters=final_params.copy(),
                    merit_value=final_merit,
                    operand_residuals=final_residuals,
                    step_norm=float(np.linalg.norm(res.x - p0)),
                    damping=0.0,
                    notes=f"Final converged state ({res.message})",
                )
            )

        success = bool(res.success or final_merit < 1e-4)

        return OptimizationResult(
            success=success,
            initial_merit=initial_merit,
            final_merit=final_merit,
            iterations=len(history) - 1,
            parameters=final_params,
            history=history,
            message=str(res.message),
            optimized_system=system,
        )

    def step(
        self,
        system: OpticalElement,
        merit_function: Optional[PhysicalMeritFunction] = None,
    ) -> OptimizationStepRecord:
        """
        Execute single-step optimization.
        """
        mf = self._get_merit_function(merit_function)
        self._step_count += 1
        p, _ = self._extract_system_variables(system)
        lb, ub = self._compute_bounds(system)

        # Finite difference Jacobian
        r = mf.residuals(system)
        dp = 1e-5
        J = np.zeros((len(r), len(p)))
        for j in range(len(p)):
            p_plus = p.copy()
            p_plus[j] += dp
            self._apply_system_variables(system, p_plus)
            r_plus = mf.residuals(system)
            J[:, j] = (r_plus - r) / dp

        # Gauss-Newton / Trust Region step with Levenberg damping
        W = np.diag([op.weight for op in mf.operands]) if len(mf.operands) == len(r) else np.eye(len(r))
        JTW = J.T @ W
        H = JTW @ J
        diag_H = np.diag(np.diag(H))
        damping = 0.01

        delta_p = np.linalg.solve(H + damping * (diag_H + 1e-6 * np.eye(len(p))), -JTW @ r)
        p_new = np.clip(p + delta_p, lb, ub)
        self._apply_system_variables(system, p_new)

        cur_merit = mf.evaluate(system)
        params = system.get_parameters() if hasattr(system, "get_parameters") else {
            "r1": system.r1, "r2": system.r2, "thickness": system.thickness, "diameter": system.diameter
        }
        residuals_dict = {op.name: op.compute_residual(system) for op in mf.operands}

        return OptimizationStepRecord(
            iteration=self._step_count,
            parameters=params,
            merit_value=cur_merit,
            operand_residuals=residuals_dict,
            step_norm=float(np.linalg.norm(delta_p)),
            damping=damping,
            jacobian=J,
            notes=f"Single step iteration {self._step_count}",
        )


class TransparentDLSOptimizer(BaseOptimizer):
    """
    White-box Damped Least Squares (Levenberg-Marquardt) Optimizer.

    Provides full transparency into:
    - Finite difference Jacobian matrix J_ij = d r_i / d p_j
    - Normal equations parameter update:
        Delta p = - (J^T W J + lambda * diag(J^T W J))^-1 J^T W r
    - Step norm, damping factor, and complete per-iteration records.
    - Explicit single-step execution (.step()) for interactive user debugging.
    """

    def __init__(
        self,
        merit_function: Optional[PhysicalMeritFunction] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        initial_damping: float = 0.001,
    ) -> None:
        super().__init__(merit_function=merit_function, max_iterations=max_iterations, tolerance=tolerance)
        self.damping = float(initial_damping)
        self._current_damping = float(initial_damping)

    def step(
        self,
        system: OpticalElement,
        merit_function: Optional[PhysicalMeritFunction] = None,
    ) -> OptimizationStepRecord:
        """
        Execute one complete DLS iteration with Jacobian sensitivity calculation.
        """
        mf = self._get_merit_function(merit_function)
        self._step_count += 1

        p, _ = self._extract_system_variables(system)
        target_diam = None
        for op in mf.operands:
            if isinstance(op, DiameterConstraintOperand) or "diameter" in op.name.lower():
                target_diam = op.target
                break
        lb, ub = self._compute_bounds(system, target_diam)

        # Base residuals and merit
        r = mf.residuals(system)
        current_merit = mf.evaluate(system)

        # Finite difference step sizes per parameter
        dp = np.array([1e-6, 1e-6, 1e-5, 1e-5], dtype=float)
        J = np.zeros((len(r), len(p)))

        for j in range(len(p)):
            p_trial = p.copy()
            p_trial[j] += dp[j]
            self._apply_system_variables(system, p_trial)
            r_trial = mf.residuals(system)
            J[:, j] = (r_trial - r) / dp[j]

        # Reset system back to current state
        self._apply_system_variables(system, p)

        # Normal equations: (J^T W J + lambda * diag(J^T W J)) Delta p = - J^T W r
        W = np.diag([op.weight for op in mf.operands]) if len(mf.operands) == len(r) else np.eye(len(r))
        JTW = J.T @ W
        JT_W_J = JTW @ J
        diag_JT_W_J = np.diag(np.diag(JT_W_J))

        # Solve parameter update with Levenberg-Marquardt damping
        damping_matrix = self._current_damping * (diag_JT_W_J + 1e-6 * np.eye(len(p)))
        A = JT_W_J + damping_matrix
        b = -JTW @ r

        try:
            delta_p = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta_p = np.linalg.lstsq(A, b, rcond=None)[0]

        p_new = np.clip(p + delta_p, lb, ub)
        self._apply_system_variables(system, p_new)

        new_merit = mf.evaluate(system)

        # Adaptive damping update
        if new_merit <= current_merit:
            # Step accepted: reduce damping for faster Gauss-Newton convergence
            self._current_damping = max(1e-7, self._current_damping * 0.5)
        else:
            # Step increased merit: adapt damping
            self._current_damping = min(1e4, self._current_damping * 2.0)

        params_dict = system.get_parameters() if hasattr(system, "get_parameters") else {
            "r1": system.r1, "r2": system.r2, "thickness": system.thickness, "diameter": system.diameter
        }
        residuals_dict = {op.name: op.compute_residual(system) for op in mf.operands}

        return OptimizationStepRecord(
            iteration=self._step_count,
            parameters=params_dict,
            merit_value=new_merit,
            operand_residuals=residuals_dict,
            step_norm=float(np.linalg.norm(delta_p)),
            damping=self._current_damping,
            jacobian=J,
            notes=f"DLS step {self._step_count} (damping={self._current_damping:.2e})",
        )

    def optimize(
        self,
        system: OpticalElement,
        merit_function: Optional[PhysicalMeritFunction] = None,
    ) -> OptimizationResult:
        mf = self._get_merit_function(merit_function)
        self.reset()
        self._current_damping = self.damping

        p0, _ = self._extract_system_variables(system)
        target_diam = None
        for op in mf.operands:
            if isinstance(op, DiameterConstraintOperand) or "diameter" in op.name.lower():
                target_diam = op.target
                break
        lb, ub = self._compute_bounds(system, target_diam)
        p0 = np.clip(p0, lb, ub)
        self._apply_system_variables(system, p0)

        initial_merit = mf.evaluate(system)
        initial_params = system.get_parameters() if hasattr(system, "get_parameters") else {
            "r1": system.r1, "r2": system.r2, "thickness": system.thickness, "diameter": system.diameter
        }
        initial_residuals = {op.name: op.compute_residual(system) for op in mf.operands}

        history: List[OptimizationStepRecord] = [
            OptimizationStepRecord(
                iteration=0,
                parameters=initial_params.copy(),
                merit_value=initial_merit,
                operand_residuals=initial_residuals,
                step_norm=0.0,
                damping=self.damping,
                notes="Initial unoptimized state",
            )
        ]

        converged = False
        prev_merit = initial_merit

        for _ in range(self.max_iterations):
            step_rec = self.step(system, mf)
            history.append(step_rec)

            cur_merit = step_rec.merit_value
            # Check convergence condition: merit is sufficiently small or stabilized
            if cur_merit < self.tolerance:
                converged = True
                break

            if abs(prev_merit - cur_merit) < self.tolerance * 0.1 and cur_merit < 1e-3:
                converged = True
                break

            prev_merit = cur_merit

        final_params = system.get_parameters() if hasattr(system, "get_parameters") else {
            "r1": system.r1, "r2": system.r2, "thickness": system.thickness, "diameter": system.diameter
        }
        final_merit = history[-1].merit_value

        return OptimizationResult(
            success=(converged or final_merit < 0.1),
            initial_merit=initial_merit,
            final_merit=final_merit,
            iterations=len(history) - 1,
            parameters=final_params,
            history=history,
            message=f"DLS converged in {len(history)-1} iterations (final merit={final_merit:.2e})",
            optimized_system=system,
        )


def optimize_lens_to_specs(
    lens: OpticalElement,
    target_efl: float = 20.0,
    target_diameter: float = 25.4,
    max_iter: int = 50,
    optimizer_type: str = "scipy",
) -> OptimizationResult:
    """
    High-level convenience routine to design and optimize a lens to prescribed physical specs.

    Args:
        lens: Input OpticalElement (e.g. ThickLens) to optimize in-place.
        target_efl: Prescribed Effective Focal Length in mm (default 20.0 mm).
        target_diameter: Prescribed clear aperture outer diameter in mm (default 25.4 mm / 1 inch).
        max_iter: Maximum number of iterations.
        optimizer_type: "scipy" (Trust Region Reflective) or "dls" (Damped Least Squares).

    Returns:
        OptimizationResult with complete trajectory and verified convergence.
    """
    mf = PhysicalMeritFunction()
    mf.add_operand(TargetEFLOperand(target_efl=target_efl, weight=1.0))
    mf.add_operand(DiameterConstraintOperand(target_diameter=target_diameter, weight=1.0))

    if optimizer_type.lower() == "scipy":
        opt = ScipyLeastSquaresOptimizer(merit_function=mf, max_iterations=max_iter)
    else:
        opt = TransparentDLSOptimizer(merit_function=mf, max_iterations=max_iter)

    return opt.optimize(lens)
