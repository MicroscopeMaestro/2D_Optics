"""
Transparent Merit Function Architecture & Physical Lens Design Operands.
========================================================================
Implements an extensible, modular optimization merit function framework for 2D Optics.
Decouples operand evaluation from monolithic routines, enabling easy subclassing
and manual debugging as required by textbook optical engineering and modern lens design.

Operands:
- TargetEFLOperand: Targets Effective Focal Length (EFL).
- TargetBFLOperand: Targets Back Focal Length (BFL).
- DiameterConstraintOperand: Constrains lens clear aperture diameter.
- ThicknessConstraintOperand: Constrains center thickness within manufacturable bounds.
- CurvatureBoundsOperand: Penalizes extreme, unmanufacturable surface curvatures.

Composite Merit Function:
- PhysicalMeritFunction: Calculates scalar sum-of-squares merit ||r||_W^2 = sum(w_i * r_i^2),
  vector residuals, weighted residuals, and detailed diagnostic operand reports.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np



class Operand(ABC):
    """
    Abstract Base Class for optimization merit operands in 2D Optics.

    Every operand evaluates a physical observable from an optical system, compares it
    to a prescribed target value, and computes unweighted and weighted residuals.

    Attributes:
        name: Human-readable identifier for the operand.
        target: Target physical value (e.g. 20.0 mm for EFL).
        weight: Statistical weighting factor w_i >= 0.
    """

    def __init__(self, name: str = "", target: float = 0.0, weight: float = 1.0) -> None:
        self.name = name or self.__class__.__name__
        self.target = float(target)
        self.weight = float(weight)

    @abstractmethod
    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate and return the actual physical value from the optical system.

        Args:
            system: The optical element, assembly, or surface stack to inspect.
            context: Optional dictionary containing additional simulation context (e.g. wavelength).

        Returns:
            Calculated physical scalar value (e.g. focal length in mm).
        """
        pass

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute unweighted residual r = actual - target.

        Args:
            system: The optical system being evaluated.
            context: Optional context dictionary.

        Returns:
            Unweighted residual r = evaluate(system) - target.
        """
        actual = self.evaluate(system, context)
        return float(actual - self.target)

    def compute_weighted_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute weighted residual r_w = weight * (actual - target).

        Args:
            system: The optical system being evaluated.
            context: Optional context dictionary.

        Returns:
            Weighted residual value.
        """
        residual = self.compute_residual(system, context)
        return float(self.weight * residual)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', target={self.target}, weight={self.weight})"


class TargetEFLOperand(Operand):
    """
    Operand targeting paraxial Effective Focal Length (EFL).

    Evaluates EFL from system.cardinal_points().efl, system.efl, or system.focal_length.
    """

    def __init__(
        self,
        target_efl: float = 20.0,
        weight: float = 1.0,
        name: str = "TargetEFL",
        target: Optional[float] = None,
    ) -> None:
        tgt = float(target if target is not None else target_efl)
        super().__init__(name=name, target=tgt, weight=weight)
        self.target_efl = tgt

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """Extract paraxial EFL in mm."""
        if hasattr(system, "efl"):
            return float(system.efl)
        elif hasattr(system, "cardinal_points") and callable(system.cardinal_points):
            cp = system.cardinal_points()
            return float(cp.efl)
        elif hasattr(system, "focal_length"):
            return float(system.focal_length)
        elif hasattr(system, "transfer_matrix") and callable(system.transfer_matrix):
            mat = system.transfer_matrix()
            return float(mat.cardinal_points().efl)
        raise AttributeError(f"System {system} does not expose an EFL or cardinal_points() interface.")


class TargetBFLOperand(Operand):
    """
    Operand targeting Back Focal Length (BFL).

    Evaluates distance from rear optical surface vertex to paraxial focal point.
    """

    def __init__(
        self,
        target_bfl: float = 20.0,
        weight: float = 1.0,
        name: str = "TargetBFL",
        target: Optional[float] = None,
    ) -> None:
        tgt = float(target if target is not None else target_bfl)
        super().__init__(name=name, target=tgt, weight=weight)
        self.target_bfl = tgt

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """Extract paraxial BFL in mm."""
        if hasattr(system, "bfl"):
            return float(system.bfl)
        elif hasattr(system, "cardinal_points") and callable(system.cardinal_points):
            cp = system.cardinal_points()
            return float(cp.bfl)
        elif hasattr(system, "transfer_matrix") and callable(system.transfer_matrix):
            mat = system.transfer_matrix()
            return float(mat.cardinal_points().bfl)
        raise AttributeError(f"System {system} does not expose a BFL or cardinal_points() interface.")


class DiameterConstraintOperand(Operand):
    """
    Operand constraining clear aperture outer diameter (e.g. 25.4 mm / 1 inch).
    """

    def __init__(
        self,
        target_diameter: float = 25.4,
        weight: float = 1.0,
        name: str = "DiameterConstraint",
        target: Optional[float] = None,
    ) -> None:
        tgt = float(target if target is not None else target_diameter)
        super().__init__(name=name, target=tgt, weight=weight)
        self.target_diameter = tgt

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """Extract clear aperture outer diameter in mm."""
        if hasattr(system, "diameter"):
            return float(system.diameter)
        elif hasattr(system, "get_parameter") and callable(system.get_parameter):
            try:
                return float(system.get_parameter("diameter"))
            except KeyError:
                pass
        elif hasattr(system, "get_parameters") and callable(system.get_parameters):
            params = system.get_parameters()
            if "diameter" in params:
                return float(params["diameter"])
        raise AttributeError(f"System {system} does not expose a diameter attribute or parameter.")


class ThicknessConstraintOperand(Operand):
    """
    Operand constraining lens center thickness within physical bounds [min_thickness, max_thickness].
    Supports optional target thickness and soft penalties for exceeding bounds.
    """

    def __init__(
        self,
        target_thickness: Optional[float] = None,
        min_thickness: float = 1.0,
        max_thickness: float = 20.0,
        weight: float = 1.0,
        min_center: Optional[float] = None,
        max_center: Optional[float] = None,
        name: str = "ThicknessConstraint",
        target: Optional[float] = None,
    ) -> None:
        if min_center is not None:
            min_thickness = min_center
        if max_center is not None:
            max_thickness = max_center
        if target is not None:
            target_thickness = target

        self.min_thickness = float(min_thickness)
        self.max_thickness = float(max_thickness)
        self.min_center = self.min_thickness
        self.max_center = self.max_thickness

        if target_thickness is not None:
            tgt = float(target_thickness)
            self.target_thickness = tgt
        else:
            tgt = (self.min_thickness + self.max_thickness) / 2.0
            self.target_thickness = None

        super().__init__(name=name, target=tgt, weight=weight)

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """Extract center thickness in mm."""
        if hasattr(system, "thickness"):
            return float(system.thickness)
        elif hasattr(system, "get_parameter") and callable(system.get_parameter):
            try:
                return float(system.get_parameter("thickness"))
            except KeyError:
                pass
        elif hasattr(system, "get_parameters") and callable(system.get_parameters):
            params = system.get_parameters()
            if "thickness" in params:
                return float(params["thickness"])
        return 0.0

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute thickness residual with bound violations.
        If target_thickness was given: residual is (actual - target) plus bound penalty.
        If only min/max bounds were given: residual is 0 inside bounds, or distance to boundary outside.
        """
        val = self.evaluate(system, context)
        if self.target_thickness is not None:
            res = val - self.target_thickness
            if val < self.min_thickness:
                res += (val - self.min_thickness)
            elif val > self.max_thickness:
                res += (val - self.max_thickness)
            return float(res)
        else:
            if val < self.min_thickness:
                return float(val - self.min_thickness)
            elif val > self.max_thickness:
                return float(val - self.max_thickness)
            return 0.0


class CurvatureBoundsOperand(Operand):
    """
    Operand penalizing extreme, unmanufacturable surface curvatures (|c| > max_curvature).
    """

    def __init__(
        self,
        max_curvature: float = 0.1,
        weight: float = 1.0,
        name: str = "CurvatureBounds",
        target: Optional[float] = None,
    ) -> None:
        self.max_curvature = float(max_curvature)
        tgt = float(target if target is not None else max_curvature)
        super().__init__(name=name, target=tgt, weight=weight)

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Extract maximum absolute curvature (mm^-1) across all surfaces in the system.
        """
        curvatures: List[float] = []
        if hasattr(system, "r1") and hasattr(system, "r2"):
            if abs(system.r1) > 1e-12 and not math.isinf(system.r1):
                curvatures.append(abs(1.0 / system.r1))
            else:
                curvatures.append(0.0)
            if abs(system.r2) > 1e-12 and not math.isinf(system.r2):
                curvatures.append(abs(1.0 / system.r2))
            else:
                curvatures.append(0.0)

        if hasattr(system, "to_surfaces") and callable(system.to_surfaces):
            try:
                surfs = system.to_surfaces()
                for s in surfs:
                    if hasattr(s, "radius") and abs(s.radius) > 1e-12 and not math.isinf(s.radius):
                        curvatures.append(abs(1.0 / s.radius))
            except Exception:
                pass

        return float(max(curvatures)) if curvatures else 0.0

    def compute_residual(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """Penalize excess curvature above max_curvature."""
        max_c = self.evaluate(system, context)
        if max_c > self.max_curvature:
            return float(max_c - self.max_curvature)
        return 0.0


class PhysicalMeritFunction:
    """
    Composite Physical Merit Function for 2D Optics Lens Optimization.

    Aggregates a sequence of physical Operands, evaluating:
    - Composite scalar merit ||r||_W^2 = sum_i (w_i * r_i^2)
    - Unweighted residuals vector r
    - Weighted residuals vector r_w
    - Detailed, inspectable operand breakdown reports for transparency and manual debugging.
    """

    def __init__(self, operands: Optional[Sequence[Operand]] = None) -> None:
        self.operands: List[Operand] = list(operands) if operands is not None else []

    def add_operand(self, operand: Operand) -> None:
        """Add an operand to the merit function."""
        if not isinstance(operand, Operand):
            raise TypeError(f"Expected Operand instance, got {type(operand)}")
        self.operands.append(operand)

    def remove_operand(self, index_or_name: Union[int, str]) -> None:
        """Remove an operand by integer index or string name."""
        if isinstance(index_or_name, int):
            self.operands.pop(index_or_name)
        elif isinstance(index_or_name, str):
            for i, op in enumerate(self.operands):
                if op.name == index_or_name:
                    self.operands.pop(i)
                    return
            raise KeyError(f"Operand '{index_or_name}' not found in merit function.")
        else:
            raise TypeError("Index must be int or str name.")

    def clear(self) -> None:
        """Clear all operands."""
        self.operands.clear()

    def evaluate(self, system: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate composite scalar merit value:
            ||r||_W^2 = sum_i [ w_i * (v_i - T_i)^2 ]
        """
        total = 0.0
        for op in self.operands:
            r = op.compute_residual(system, context)
            total += op.weight * (r ** 2)
        return float(total)

    def residuals(self, system: Any, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Return 1D numpy array of unweighted residuals r_i = actual_i - target_i.
        """
        return np.array([op.compute_residual(system, context) for op in self.operands], dtype=float)

    def compute_residuals(self, system: Any, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Alias for residuals()."""
        return self.residuals(system, context)

    def weighted_residuals(self, system: Any, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Return 1D numpy array of weighted residuals r_w,i = weight_i * r_i.
        """
        return np.array([op.compute_weighted_residual(system, context) for op in self.operands], dtype=float)

    def compute_weighted_residuals(self, system: Any, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Alias for weighted_residuals()."""
        return self.weighted_residuals(system, context)

    def operand_report(self, system: Any, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate a detailed inspectable report for all operands.

        Returns:
            List of dictionaries containing name, target, actual, residual, weight,
            and squared weighted contribution.
        """
        report: List[Dict[str, Any]] = []
        for op in self.operands:
            actual = op.evaluate(system, context)
            residual = op.compute_residual(system, context)
            weighted_res = op.compute_weighted_residual(system, context)
            contribution = op.weight * (residual ** 2)
            report.append({
                "name": op.name,
                "target": op.target,
                "actual": actual,
                "residual": residual,
                "weight": op.weight,
                "weighted_residual": weighted_res,
                "contribution": contribution,
            })
        return report

    def __len__(self) -> int:
        return len(self.operands)

    def __iter__(self):
        return iter(self.operands)

    def __getitem__(self, index: int) -> Operand:
        return self.operands[index]

    def __repr__(self) -> str:
        return f"PhysicalMeritFunction(operands={self.operands})"
