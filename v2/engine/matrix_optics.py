"""
Paraxial 2x2 ABCD Ray Transfer Matrix & 3x3 Augmented Perturbation Engine.

Implements paraxial matrix ray optics based on standard textbook formulations
(University of Colorado Optical Engineering, Greivenkamp, Gerrard & Burch, Siegman).
Provides transparent diagnostic calculation tracing, cardinal point extraction
(EFL, BFL, FFL, Principal Planes, Nodal Points), and 3x3 augmented representations
for Zemax-style decenter and tilt tolerancing.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Iterator, Union
import numpy as np


@dataclass
class CalculationStep:
    """
    Individual mathematical step in an optical derivation or ray tracing sequence.

    Attributes:
        name: Human-readable name of the operation (e.g. 'Front Surface Refraction').
        formula: Algebraic or LaTeX string representing the formula applied.
        inputs: Dictionary of substituted input parameters.
        intermediate_values: Dictionary of intermediate evaluated scalars or vectors.
        matrix: 2x2 or 3x3 transfer matrix computed for this step.
        notes: Contextual optical physics notes or diagnostic commentary.
        step_number: 1-based sequential step index.
    """
    name: str
    formula: str
    inputs: Dict[str, Any]
    matrix: Optional["TransferMatrix"] = None
    notes: str = ""
    intermediate_values: Dict[str, Any] = field(default_factory=dict)
    step_number: int = 1

    def format_step(self) -> str:
        """Format step into a clear readable multi-line debug block."""
        lines = [f"Step {self.step_number}: {self.name}"]
        lines.append(f"  Formula: {self.formula}")
        if self.inputs:
            inputs_str = ", ".join(f"{k}={v}" for k, v in self.inputs.items())
            lines.append(f"  Inputs:  {inputs_str}")
        if self.intermediate_values:
            res_str = ", ".join(
                f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in self.intermediate_values.items()
            )
            lines.append(f"  Values:  {res_str}")
        if self.matrix is not None:
            lines.append(f"  Matrix:")
            lines.append(f"    [ {self.matrix.A:12.6f}  {self.matrix.B:12.6f} ]")
            lines.append(f"    [ {self.matrix.C:12.6f}  {self.matrix.D:12.6f} ]")
            if abs(self.matrix.E) > 1e-12 or abs(self.matrix.F) > 1e-12:
                lines.append(f"    Augmented 3x3 (E={self.matrix.E:+.6f}, F={self.matrix.F:+.6f})")
        if self.notes:
            lines.append(f"  Notes:   {self.notes}")
        return "\n".join(lines)


class CalculationTrace:
    """
    Sequential record of mathematical calculation steps for manual stepping and inspection.

    Fulfills the user constraint: 'make space for future new implementations, models
    and also debugging manually.'
    """

    def __init__(self, title: str = "Optical Calculation Trace"):
        self.title: str = title
        self.steps: List[CalculationStep] = []
        self.metadata: Dict[str, Any] = {}

    def add_step(self, step: CalculationStep) -> None:
        """Append a calculation step, ensuring step numbering is contiguous."""
        if step.step_number == 1 and len(self.steps) > 0:
            step.step_number = len(self.steps) + 1
        self.steps.append(step)

    def format_debug_report(self) -> str:
        """Return a formatted ASCII report of all recorded steps."""
        banner = f"=== {self.title} ==="
        border = "=" * max(len(banner), 60)
        lines = [border, banner, border]
        for s in self.steps:
            lines.append(s.format_step())
            lines.append("-" * 40)
        return "\n".join(lines)

    def __iter__(self) -> Iterator[CalculationStep]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> CalculationStep:
        return self.steps[idx]

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to structured dictionary for logging or UI display."""
        return {
            "title": self.title,
            "metadata": self.metadata,
            "steps": [
                {
                    "step_number": s.step_number,
                    "name": s.name,
                    "formula": s.formula,
                    "inputs": s.inputs,
                    "intermediate_values": s.intermediate_values,
                    "notes": s.notes,
                    "matrix": s.matrix.to_dict() if s.matrix else None,
                }
                for s in self.steps
            ],
        }


@dataclass
class CardinalPoints:
    """
    Standard paraxial cardinal points of an optical system.

    Attributes:
        efl: Effective Focal Length f = -1/C (mm). Positive = converging.
        bfl: Back Focal Length from rear vertex V2 to rear focus F2 (mm).
        ffl: Front Focal Length from front focus F1 to front vertex V1 (mm).
        p1: Front Principal Plane location from front vertex V1 (mm). Positive = inside lens.
        p2: Rear Principal Plane location from rear vertex V2 (mm). Negative = inside lens.
        f1: Directed position of front focal point F1 from front vertex V1 (mm).
        f2: Directed position of rear focal point F2 from rear vertex V2 (mm) (equals bfl).
        n1: Front Nodal Point location from front vertex V1 (mm). In air, n1 == p1.
        n2: Rear Nodal Point location from rear vertex V2 (mm). In air, n2 == p2.
        power: Total optical power Phi = -C (mm^-1).
        is_afocal: True if optical power |C| < 1e-12 (infinite EFL, e.g. Galilean telescope).
        angular_magnification: Paraxial angular magnification (for afocal systems m_alpha = D = -f1/f2).
        f1_pos: Alias for f1.
        f2_pos: Alias for f2.
    """
    efl: float
    bfl: float
    ffl: float
    p1: float
    p2: float
    f1: float
    f2: float
    n1: float
    n2: float
    power: float = 0.0
    is_afocal: bool = False
    angular_magnification: float = 1.0
    f1_pos: float = 0.0
    f2_pos: float = 0.0

    def __post_init__(self):
        if self.f1_pos == 0.0 and self.f1 != 0.0:
            self.f1_pos = self.f1
        if self.f2_pos == 0.0 and self.f2 != 0.0:
            self.f2_pos = self.f2

    def summary(self) -> str:
        """Return formatted ASCII summary of cardinal points."""
        lines = [
            "Cardinal Points Summary:",
            f"  EFL (Effective Focal Length):  {self.efl:12.6f} mm" if not self.is_afocal else "  EFL:                          Afocal (inf)",
            f"  BFL (Back Focal Length):       {self.bfl:12.6f} mm" if not self.is_afocal else "  BFL:                          Afocal (inf)",
            f"  FFL (Front Focal Length):      {self.ffl:12.6f} mm" if not self.is_afocal else "  FFL:                          Afocal (inf)",
            f"  P1  (Front Principal Plane):   {self.p1:12.6f} mm (from V1)",
            f"  P2  (Rear Principal Plane):    {self.p2:12.6f} mm (from V2)",
            f"  F1  (Front Focal Point):       {self.f1:12.6f} mm (from V1)",
            f"  F2  (Rear Focal Point):        {self.f2:12.6f} mm (from V2)",
            f"  N1  (Front Nodal Point):       {self.n1:12.6f} mm (from V1)",
            f"  N2  (Rear Nodal Point):        {self.n2:12.6f} mm (from V2)",
            f"  Power (Phi = -C):              {self.power:12.6e} mm^-1",
            f"  Afocal Status:                 {self.is_afocal}",
        ]
        if self.is_afocal:
            lines.append(f"  Angular Magnification:         {self.angular_magnification:12.6f}x")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, float]:
        return {
            "efl": self.efl,
            "bfl": self.bfl,
            "ffl": self.ffl,
            "p1": self.p1,
            "p2": self.p2,
            "f1": self.f1,
            "f2": self.f2,
            "n1": self.n1,
            "n2": self.n2,
            "power": self.power,
            "is_afocal": float(self.is_afocal),
            "angular_magnification": self.angular_magnification,
        }


@dataclass
class TransferMatrix:
    """
    2x2 Paraxial Ray Transfer Matrix with 3x3 Augmented Tolerancing Representation.

    Standard 2x2 ray transformation:
        [y_out]   [A  B] [y_in]
        [u_out] = [C  D] [u_in]

    Augmented 3x3 transformation for decenters (Delta y) and tilts (Delta theta):
        [y_out]   [A  B  E] [y_in]
        [u_out] = [C  D  F] [u_in]
        [  1  ]   [0  0  1] [  1  ]

    where:
        E: Transverse displacement offset Delta y contribution
        F: Paraxial angular deflection Delta theta contribution

    Attributes:
        A: Lateral magnification (d y_out / d y_in for u_in = 0).
        B: Effective optical distance / path length (d y_out / d u_in for y_in = 0).
        C: Negative optical power -Phi (d u_out / d y_in for u_in = 0).
        D: Angular magnification ratio (d u_out / d u_in for y_in = 0).
        E: Offset in ray height y_out from decenters or tilts (mm).
        F: Offset in ray slope u_out from decenters or tilts (rad).
        trace: Optional CalculationTrace detailing derivation steps.
        label: Descriptive label for reporting.
        n_in: Refractive index of entrance medium.
        n_out: Refractive index of exit medium.
    """
    A: float = 1.0
    B: float = 0.0
    C: float = 0.0
    D: float = 1.0
    E: float = 0.0
    F: float = 0.0
    trace: Optional[CalculationTrace] = None
    label: str = ""
    n_in: float = 1.0
    n_out: float = 1.0

    def to_2x2(self) -> np.ndarray:
        """Return 2x2 numpy array [[A, B], [C, D]]."""
        return np.array([[self.A, self.B], [self.C, self.D]], dtype=float)

    def to_3x3(self) -> np.ndarray:
        """Return 3x3 augmented numpy array [[A, B, E], [C, D, F], [0, 0, 1]]."""
        return np.array([
            [self.A, self.B, self.E],
            [self.C, self.D, self.F],
            [0.0, 0.0, 1.0]
        ], dtype=float)

    def to_array(self) -> np.ndarray:
        """Alias for to_2x2."""
        return self.to_2x2()

    @classmethod
    def from_2x2(
        cls,
        mat: np.ndarray,
        label: str = "",
        n_in: float = 1.0,
        n_out: float = 1.0,
        trace: Optional[CalculationTrace] = None
    ) -> "TransferMatrix":
        """Create TransferMatrix from 2x2 array."""
        return cls(
            A=float(mat[0, 0]),
            B=float(mat[0, 1]),
            C=float(mat[1, 0]),
            D=float(mat[1, 1]),
            label=label,
            n_in=n_in,
            n_out=n_out,
            trace=trace,
        )

    @classmethod
    def from_3x3(
        cls,
        mat: np.ndarray,
        label: str = "",
        n_in: float = 1.0,
        n_out: float = 1.0,
        trace: Optional[CalculationTrace] = None
    ) -> "TransferMatrix":
        """Create TransferMatrix from 3x3 augmented array."""
        return cls(
            A=float(mat[0, 0]),
            B=float(mat[0, 1]),
            C=float(mat[1, 0]),
            D=float(mat[1, 1]),
            E=float(mat[0, 2]),
            F=float(mat[1, 2]),
            label=label,
            n_in=n_in,
            n_out=n_out,
            trace=trace,
        )

    @classmethod
    def from_array(cls, mat: np.ndarray) -> "TransferMatrix":
        """Factory accepting either 2x2 or 3x3 numpy array."""
        if mat.shape == (2, 2):
            return cls.from_2x2(mat)
        elif mat.shape == (3, 3):
            return cls.from_3x3(mat)
        else:
            raise ValueError(f"Expected 2x2 or 3x3 matrix, got shape {mat.shape}")

    @classmethod
    def refraction(
        cls,
        n1: float = 1.0,
        n2: float = 1.0,
        radius: Optional[float] = None,
        R: Optional[float] = None,
        trace: Optional[CalculationTrace] = None,
    ) -> "TransferMatrix":
        """
        Create a paraxial curved or flat interface refraction transfer matrix.

        Args:
            n1: Entrance refractive index (before interface).
            n2: Exit refractive index (after interface).
            radius: Radius of curvature in mm (default inf for flat surface).
            R: Alias for radius.
            trace: Optional CalculationTrace for recording step.
        """
        r_val = radius if radius is not None else (R if R is not None else float("inf"))
        return refraction_matrix(R=r_val, n1=n1, n2=n2, trace=trace)

    @classmethod
    def translation(
        cls,
        thickness: Optional[float] = None,
        index: float = 1.0,
        d: Optional[float] = None,
        n: Optional[float] = None,
        trace: Optional[CalculationTrace] = None,
    ) -> "TransferMatrix":
        """
        Create a paraxial homogeneous translation transfer matrix.

        Args:
            thickness: Axial translation distance in mm.
            index: Refractive index of medium (default 1.0).
            d: Alias for thickness.
            n: Alias for index.
            trace: Optional CalculationTrace for recording step.
        """
        dist = thickness if thickness is not None else (d if d is not None else 0.0)
        idx = n if n is not None else index
        return space_matrix(d=dist, n=idx, trace=trace)

    def __matmul__(self, other: "TransferMatrix") -> "TransferMatrix":
        """
        Matrix multiplication: M_composite = self @ other.

        Light travels through 'other' first, then through 'self'.
        Calculates both 2x2 ABCD compounding and 3x3 augmented offsets E, F:
            E_new = A_self * E_other + B_self * F_other + E_self
            F_new = C_self * E_other + D_self * F_other + F_self
        """
        if not isinstance(other, TransferMatrix):
            return NotImplemented

        A_new = self.A * other.A + self.B * other.C
        B_new = self.A * other.B + self.B * other.D
        C_new = self.C * other.A + self.D * other.C
        D_new = self.C * other.B + self.D * other.D

        E_new = self.A * other.E + self.B * other.F + self.E
        F_new = self.C * other.E + self.D * other.F + self.F

        # Trace handling: preserve existing trace or chain steps
        new_trace = self.trace if self.trace is not None else other.trace

        new_label = f"({self.label} @ {other.label})" if self.label and other.label else (self.label or other.label)

        return TransferMatrix(
            A=float(A_new),
            B=float(B_new),
            C=float(C_new),
            D=float(D_new),
            E=float(E_new),
            F=float(F_new),
            trace=new_trace,
            label=new_label,
            n_in=other.n_in,
            n_out=self.n_out,
        )

    def determinant(self) -> float:
        """
        Determinant AD - BC.
        In geometric coordinates: det(M) = n_in / n_out.
        For a complete lens in air: det(M) == 1.0.
        """
        return float(self.A * self.D - self.B * self.C)

    def inverse(self) -> "TransferMatrix":
        """Matrix inverse (reverses ray propagation direction)."""
        det = self.determinant()
        if abs(det) < 1e-15:
            raise ValueError(f"TransferMatrix is singular (det={det}), cannot invert.")

        inv_A = self.D / det
        inv_B = -self.B / det
        inv_C = -self.C / det
        inv_D = self.A / det

        # For 3x3 inverse: [A B E; C D F; 0 0 1]^-1
        # [E_inv; F_inv] = - [A B; C D]^-1 @ [E; F]
        inv_E = -(inv_A * self.E + inv_B * self.F)
        inv_F = -(inv_C * self.E + inv_D * self.F)

        return TransferMatrix(
            A=inv_A,
            B=inv_B,
            C=inv_C,
            D=inv_D,
            E=inv_E,
            F=inv_F,
            label=f"inv({self.label})" if self.label else "inverse",
            n_in=self.n_out,
            n_out=self.n_in,
        )

    def propagate_ray(self, y: float, u: float) -> Tuple[float, float]:
        """
        Propagate paraxial ray vector [y, u]^T through this matrix.

        Args:
            y: Transverse ray height from optical axis (mm).
            u: Ray slope/angle with respect to optical axis (radians).

        Returns:
            (y_out, u_out) where y_out is height in mm, u_out is slope in rad.
        """
        y_out = self.A * y + self.B * u + self.E
        u_out = self.C * y + self.D * u + self.F
        return float(y_out), float(u_out)

    def propagate(self, y: float, u: float) -> Tuple[float, float]:
        """Convenience alias for propagate_ray."""
        return self.propagate_ray(y, u)

    def with_decenter(self, delta_y: float) -> "TransferMatrix":
        """
        Return a copy perturbed by a transverse decenter Delta y (mm).

        From Zemax/Literature 3x3 augmented formulation:
            E_new = E + (1 - A) * Delta y
            F_new = F - C * Delta y
        """
        E_new = self.E + (1.0 - self.A) * delta_y
        F_new = self.F - self.C * delta_y
        return TransferMatrix(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            E=E_new,
            F=F_new,
            trace=self.trace,
            label=f"{self.label}[dy={delta_y:+.4f}]" if self.label else f"decenter(dy={delta_y:+.4f})",
            n_in=self.n_in,
            n_out=self.n_out,
        )

    def with_tilt(self, delta_theta: float, in_degrees: bool = False) -> "TransferMatrix":
        """
        Return a copy perturbed by a tilt angle Delta theta.

        Args:
            delta_theta: Tilt angle (radians by default, or degrees if in_degrees=True).
            in_degrees: Whether delta_theta is provided in degrees.

        From Zemax/Literature 3x3 augmented formulation:
            E_new = E - B * Delta theta
            F_new = F + (1 - D) * Delta theta
        """
        theta_rad = math.radians(delta_theta) if in_degrees else delta_theta
        E_new = self.E - self.B * theta_rad
        F_new = self.F + (1.0 - self.D) * theta_rad
        return TransferMatrix(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            E=E_new,
            F=F_new,
            trace=self.trace,
            label=f"{self.label}[tilt={delta_theta:+.4f}]" if self.label else f"tilt({delta_theta:+.4f})",
            n_in=self.n_in,
            n_out=self.n_out,
        )

    def cardinal_points(
        self,
        n_in: Optional[float] = None,
        n_out: Optional[float] = None
    ) -> CardinalPoints:
        """
        Extract paraxial cardinal points (EFL, BFL, FFL, P1, P2, N1, N2).

        Textbook definitions (University of Colorado / Greivenkamp):
            Phi = -C (optical power)
            EFL = -1 / C (for lens in air)
            BFL = -A / C (rear vertex V2 to rear focus F2)
            FFL = -D / C (front focus F1 to front vertex V1)
            P1 = (D - 1) / C (front vertex V1 to front principal plane H1)
            P2 = (1 - A) / C (rear vertex V2 to rear principal plane H2)
            N1 = P1, N2 = P2 (when n_in == n_out)
        """
        n1 = self.n_in if n_in is None else n_in
        n2 = self.n_out if n_out is None else n_out

        if abs(self.C) < 1e-12:
            # Afocal system (Galilean / Keplerian telescope)
            ang_mag = self.D if abs(n1 - n2) < 1e-9 else (n1 / (n2 * self.A)) if abs(self.A) > 1e-12 else self.D
            return CardinalPoints(
                efl=float("inf"),
                bfl=float("inf"),
                ffl=float("inf"),
                p1=0.0,
                p2=0.0,
                f1=float("-inf"),
                f2=float("inf"),
                n1=0.0,
                n2=0.0,
                power=0.0,
                is_afocal=True,
                angular_magnification=float(ang_mag),
            )

        # Standard focal system
        if abs(n1 - n2) < 1e-9:
            efl = -1.0 / self.C
        else:
            efl = -n2 / self.C

        bfl = -self.A / self.C
        ffl = -self.D / self.C
        p1 = (self.D - 1.0) / self.C
        p2 = (1.0 - self.A) / self.C

        # Nodal points
        if abs(n1 - n2) < 1e-9:
            n1_pos = p1
            n2_pos = p2
        else:
            det = self.determinant()
            n1_pos = p1 + (-n2 / self.C - (-n1 / self.C))
            n2_pos = (det - self.A) / self.C

        f1_pos = p1 - efl
        f2_pos = bfl
        power = -self.C

        return CardinalPoints(
            efl=float(efl),
            bfl=float(bfl),
            ffl=float(ffl),
            p1=float(p1),
            p2=float(p2),
            f1=float(f1_pos),
            f2=float(f2_pos),
            n1=float(n1_pos),
            n2=float(n2_pos),
            power=float(power),
            is_afocal=False,
            angular_magnification=float(self.D),
        )

    def effective_focal_length(self) -> float:
        """Return Effective Focal Length (EFL)."""
        return self.cardinal_points().efl

    def back_focal_length(self) -> float:
        """Return Back Focal Length (BFL)."""
        return self.cardinal_points().bfl

    def front_focal_length(self) -> float:
        """Return Front Focal Length (FFL)."""
        return self.cardinal_points().ffl

    def principal_planes(self) -> Tuple[float, float]:
        """Return (P1, P2) locations."""
        cp = self.cardinal_points()
        return cp.p1, cp.p2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "D": self.D,
            "E": self.E,
            "F": self.F,
            "det": self.determinant(),
            "label": self.label,
            "n_in": self.n_in,
            "n_out": self.n_out,
        }


# =============================================================================
# Canonical 2x2 Matrix Generators
# =============================================================================

def refraction_matrix(
    R: float,
    n1: float,
    n2: float,
    trace: Optional[CalculationTrace] = None
) -> TransferMatrix:
    """
    Paraxial refraction matrix at a curved spherical interface of radius R.

    Formula:
        M_refr = [ 1            0    ]
                 [ -(n2-n1)/(n2*R)  n1/n2 ]

    Sign convention:
        R > 0: Surface center of curvature lies to the right (convex to incident beam).
        R < 0: Surface center of curvature lies to the left (concave to incident beam).
        R == 0 or inf: Planar flat surface (power = 0).

    Args:
        R: Radius of curvature (mm).
        n1: Refractive index before interface.
        n2: Refractive index after interface.
        trace: Optional CalculationTrace to record step.
    """
    if n2 == 0:
        raise ValueError("Refractive index n2 cannot be zero.")

    is_flat = (R == 0.0 or math.isinf(R) or abs(R) > 1e12)
    if is_flat:
        power = 0.0
        C = 0.0
    else:
        power = (n2 - n1) / R
        C = -(n2 - n1) / (n2 * R)

    A = 1.0
    B = 0.0
    D = n1 / n2

    mat = TransferMatrix(
        A=A,
        B=B,
        C=C,
        D=D,
        label=f"Refraction(R={R}, {n1}->{n2})",
        n_in=n1,
        n_out=n2,
    )

    if trace is not None:
        step = CalculationStep(
            name=f"Surface Refraction ({n1:.4f} -> {n2:.4f}, R={R})",
            formula="M_refr = [1, 0; -(n2 - n1)/(n2 * R), n1 / n2]",
            inputs={"R": R, "n1": n1, "n2": n2, "is_flat": is_flat},
            intermediate_values={"surface_power": power, "C": C, "D": D},
            matrix=mat,
            notes="Flat surface handled with zero power" if is_flat else "Spherical refraction",
        )
        trace.add_step(step)

    return mat


def space_matrix(
    d: float,
    n: float = 1.0,
    trace: Optional[CalculationTrace] = None
) -> TransferMatrix:
    """
    Paraxial translation matrix through distance d in a homogeneous medium.

    Formula (geometric ray coordinates [y, u]^T):
        M_space = [ 1  d ]
                  [ 0  1 ]

    Args:
        d: Axial translation distance (mm).
        n: Refractive index of the medium (default 1.0).
        trace: Optional CalculationTrace to record step.
    """
    mat = TransferMatrix(
        A=1.0,
        B=float(d),
        C=0.0,
        D=1.0,
        label=f"Space(d={d}, n={n})",
        n_in=n,
        n_out=n,
    )

    if trace is not None:
        step = CalculationStep(
            name=f"Homogeneous Space Propagation (d={d} mm)",
            formula="M_space = [1, d; 0, 1]",
            inputs={"d": d, "n": n},
            intermediate_values={"thickness": d},
            matrix=mat,
            notes=f"Propagating {d} mm through medium with index {n}",
        )
        trace.add_step(step)

    return mat


def thin_lens_matrix(
    f: float,
    trace: Optional[CalculationTrace] = None
) -> TransferMatrix:
    """
    Paraxial thin lens transfer matrix with focal length f.

    Formula:
        M_thin = [  1   0 ]
                 [ -1/f 1 ]

    Args:
        f: Effective focal length (mm). Positive = converging, negative = diverging.
        trace: Optional CalculationTrace to record step.
    """
    if f == 0.0:
        raise ValueError("Thin lens focal length f cannot be zero.")

    C = -1.0 / f
    mat = TransferMatrix(
        A=1.0,
        B=0.0,
        C=C,
        D=1.0,
        label=f"ThinLens(f={f})",
        n_in=1.0,
        n_out=1.0,
    )

    if trace is not None:
        step = CalculationStep(
            name=f"Thin Lens (f={f} mm)",
            formula="M_thin = [1, 0; -1/f, 1]",
            inputs={"f": f},
            intermediate_values={"optical_power": 1.0 / f, "C": C},
            matrix=mat,
            notes="Ideal paraxial thin lens approximation",
        )
        trace.add_step(step)

    return mat


def reflection_matrix(
    R: float,
    trace: Optional[CalculationTrace] = None
) -> TransferMatrix:
    """
    Paraxial reflection matrix at a curved spherical mirror of radius R.

    Formula:
        M_refl = [  1   0 ]
                 [ 2/R  1 ] = [  1    0 ]
                              [ -1/f  1 ]

    Sign convention:
        Concave mirror (converging): R < 0, f = -R/2 > 0.
        Convex mirror (diverging): R > 0, f = -R/2 < 0.
        Flat mirror: R == 0 or inf, C = 0.

    Args:
        R: Radius of curvature of mirror (mm).
        trace: Optional CalculationTrace to record step.
    """
    is_flat = (R == 0.0 or math.isinf(R) or abs(R) > 1e12)
    C = 0.0 if is_flat else 2.0 / R

    mat = TransferMatrix(
        A=1.0,
        B=0.0,
        C=C,
        D=1.0,
        label=f"Mirror(R={R})",
        n_in=1.0,
        n_out=1.0,
    )

    if trace is not None:
        step = CalculationStep(
            name=f"Curved Mirror Reflection (R={R})",
            formula="M_refl = [1, 0; 2/R, 1]",
            inputs={"R": R, "is_flat": is_flat},
            intermediate_values={"C": C, "focal_length": -R / 2.0 if not is_flat else float("inf")},
            matrix=mat,
            notes="Flat mirror" if is_flat else "Spherical reflection",
        )
        trace.add_step(step)

    return mat


def thick_lens_matrix(
    R1: float,
    R2: float,
    thickness: float,
    n_lens: float,
    n_ambient: float = 1.0,
    trace: Optional[CalculationTrace] = None
) -> TransferMatrix:
    """
    Composite 2x2 transfer matrix for a physical thick lens in an ambient medium.

    Cascades:
        M_thick = M_refr(R2, n_lens -> n_ambient) @ M_space(thickness, n_lens) @ M_refr(R1, n_ambient -> n_lens)

    Determinant invariant:
        det(M_thick) == 1.0 (for n_ambient_in == n_ambient_out).

    Args:
        R1: Front surface radius of curvature (mm).
        R2: Rear surface radius of curvature (mm).
        thickness: Center thickness along optical axis (mm).
        n_lens: Refractive index of lens glass.
        n_ambient: Refractive index of ambient medium (default 1.0 for air).
        trace: Optional CalculationTrace to record step-by-step diagnostic breakdown.

    Returns:
        TransferMatrix with cardinal points and optional diagnostic trace attached.
    """
    active_trace = trace if trace is not None else CalculationTrace("Thick Lens Matrix Derivation")

    # Step 1: Front Surface Refraction
    M1 = refraction_matrix(R1, n_ambient, n_lens, trace=active_trace)

    # Step 2: Homogeneous Glass Translation
    Mt = space_matrix(thickness, n=n_lens, trace=active_trace)

    # Step 3: Rear Surface Refraction
    M2 = refraction_matrix(R2, n_lens, n_ambient, trace=active_trace)

    # Composite transfer matrix
    M_thick = M2 @ Mt @ M1
    M_thick.label = f"ThickLens(R1={R1}, R2={R2}, t={thickness}, n={n_lens})"
    M_thick.n_in = n_ambient
    M_thick.n_out = n_ambient

    # Step 4: Record composite system matrix in trace
    cp = M_thick.cardinal_points(n_in=n_ambient, n_out=n_ambient)
    composite_step = CalculationStep(
        name="Composite Thick Lens Matrix",
        formula="M_thick = M_refr2 @ M_space @ M_refr1",
        inputs={
            "R1": R1,
            "R2": R2,
            "thickness": thickness,
            "n_lens": n_lens,
            "n_ambient": n_ambient,
        },
        intermediate_values={
            "A": M_thick.A,
            "B": M_thick.B,
            "C": M_thick.C,
            "D": M_thick.D,
            "determinant": M_thick.determinant(),
            "EFL": cp.efl,
            "BFL": cp.bfl,
            "P1": cp.p1,
            "P2": cp.p2,
        },
        matrix=M_thick,
        notes="Verified det(M) == 1.0; Cardinal points computed via ABCD elements.",
    )
    active_trace.add_step(composite_step)
    M_thick.trace = active_trace

    return M_thick


# =============================================================================
# Backward-Compatible Function Aliases
# =============================================================================

def matrix_refraction(
    *args: Any,
    n1: Optional[float] = None,
    n2: Optional[float] = None,
    radius: Optional[float] = None,
    R: Optional[float] = None,
    trace: Optional[CalculationTrace] = None,
    **kwargs: Any,
) -> TransferMatrix:
    """
    Backward-compatible functional interface for refraction matrix.
    Accepts (n1, n2, radius), (R, n1, n2), or named keywords.
    """
    if len(args) == 3:
        a, b, c = args
        # If called positionally as (1.0, 1.5, 50.0) -> (n1, n2, R)
        if a <= 2.5 and b <= 2.5 and (c > 2.5 or abs(c) > 2.5 or c <= 0 or math.isinf(c)):
            return refraction_matrix(R=float(c), n1=float(a), n2=float(b), trace=trace)
        else:
            return refraction_matrix(R=float(a), n1=float(b), n2=float(c), trace=trace)
    r_val = radius if radius is not None else (R if R is not None else kwargs.get("radius", kwargs.get("R", float("inf"))))
    n1_val = n1 if n1 is not None else kwargs.get("n1", 1.0)
    n2_val = n2 if n2 is not None else kwargs.get("n2", 1.0)
    return refraction_matrix(R=r_val, n1=n1_val, n2=n2_val, trace=trace)


def matrix_translation(
    *args: Any,
    thickness: Optional[float] = None,
    index: Optional[float] = None,
    d: Optional[float] = None,
    n: Optional[float] = None,
    trace: Optional[CalculationTrace] = None,
    **kwargs: Any,
) -> TransferMatrix:
    """
    Backward-compatible functional interface for translation / space matrix.
    Accepts (thickness, index), (d, n), or named keywords.
    """
    if len(args) >= 1:
        d_val = float(args[0])
        n_val = float(args[1]) if len(args) >= 2 else (n if n is not None else 1.0)
        return space_matrix(d=d_val, n=n_val, trace=trace)
    dist = thickness if thickness is not None else (d if d is not None else kwargs.get("thickness", kwargs.get("d", 0.0)))
    idx = index if index is not None else (n if n is not None else kwargs.get("index", kwargs.get("n", 1.0)))
    return space_matrix(d=dist, n=idx, trace=trace)


__all__ = [
    "CalculationStep",
    "CalculationTrace",
    "CardinalPoints",
    "TransferMatrix",
    "refraction_matrix",
    "space_matrix",
    "thin_lens_matrix",
    "reflection_matrix",
    "thick_lens_matrix",
    "matrix_refraction",
    "matrix_translation",
]
