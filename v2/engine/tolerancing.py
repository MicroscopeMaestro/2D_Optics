"""
Zemax-Style Optical Tolerancing & Simulation Engine.
===================================================
Implements comprehensive optical tolerancing based on Zemax OpticStudio
standards and classical optical engineering formulations (Greivenkamp,
Smith, Fischer, Shannon, ISO 10110-12).

Supports both:
1. Sequential 2D/3D Ray-Tracing SurfaceStack systems (with exact 3D Snell
   refraction, aspheres, coordinate breaks, mirrors, gratings, detectors).
2. Paraxial Matrix-Optics systems (ThickLens, ThinLens, augmented 3x3 matrices).

Key Features:
- Zemax Tolerance Operands (TDE):
  * TRAD: Tolerance on Radius of curvature (mm or %)
  * TCUR: Tolerance on Curvature (1/mm)
  * TTHI: Tolerance on Thickness or Airspace (mm)
  * TSDX, TSDY: Surface Decenter X / Y (mm)
  * TSTX, TSTY: Surface Tilt X / Y (degrees or mrad)
  * TEDX, TEDY: Element Decenter X / Y (mm)
  * TETX, TETY: Element Tilt X / Y (degrees or mrad)
  * TIND: Refractive Index shift (delta n)
  * TABB: Abbe Number dispersion shift (% delta V_d)
  * TCON: Conic Constant (delta k)
  * TIRR: Surface Irregularity (fringes/waves)
  * COMP: Compensator definition (e.g. refocusing image plane / BFL)
- Tolerance Probability Distributions:
  * Uniform [-Delta, +Delta]
  * Gaussian / Normal (3-sigma or 2-sigma limit, truncated at Delta)
  * Parabolic (Zemax OpticStudio P(x) ~ 1 - (x/Delta)^2)
- Tolerancing Criteria:
  * RMS_SPOT_RADIUS (Zemax TRAC / RSCE centroid RMS spot size via exact ray grid)
  * GEO_SPOT_RADIUS (Zemax GEOR 100% geometric encircled spot size)
  * EFFL / FOCAL_LENGTH (Zemax EFFL effective focal length error)
  * BFL (Back focal length)
  * MERIT_FUNCTION (User-defined merit function)
- Automatic Compensator Optimization:
  * Optimizes compensators (e.g. refocusing image/detector surface distance)
    per perturbation step and trial to find the true compensated performance.
- Zemax Sensitivity Analysis (RSS & Worst-Case):
  * Evaluates +/- Delta extreme perturbations with compensator optimization.
  * Calculates Root-Sum-Square (RSS) and Worst-Case predicted degradation.
  * Generates ranked worst-offenders table.
- Zemax Monte Carlo Simulation:
  * Multivariate random sampling across all defined distributions.
  * Re-optimizes compensators per trial.
  * Statistical metrics: Mean, Std Dev, Min, Max.
  * Cumulative percentiles: 50% (median), 80%, 90%, 95%, 98%, 99%.
  * Manufacturing yield rate calculation for any performance threshold.
- Standard Quality Presets:
  * Commercial (Thorlabs / Edmund standard optics)
  * Precision
  * High-Precision (Laser / lithography grade)
- Zemax File & Report Integration:
  * Export formatted Zemax OpticStudio ASCII tolerance reports (.txt)
  * Export and parse Zemax .zmx prescriptions with embedded TOL operands
"""

from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
import numpy as np
from scipy.optimize import minimize_scalar

# Matrix Optics Imports
from optics_app.engine.matrix_optics import (
    TransferMatrix,
    CardinalPoints,
    refraction_matrix,
    space_matrix,
)
from optics_app.engine.thick_lens import OpticalElement, ThickLens, ThinLens

# Surface Stack & Ray Tracing Imports
try:
    from optics_app.engine.surface import Surface, SurfaceType
    from optics_app.engine.lde import SurfaceStack
    from optics_app.engine.tracer import OpticalSystem
    from optics_app.engine.ray import Ray
    from optics_app.engine.matrix_tracer import MatrixTracer
except ImportError:
    try:
        from v2.engine.surface import Surface, SurfaceType
        from v2.engine.lde import SurfaceStack
        from v2.engine.tracer import OpticalSystem
        from v2.engine.ray import Ray
        from v2.engine.matrix_tracer import MatrixTracer
    except ImportError:
        pass


# =============================================================================
# 1. ZEMAX TOLERANCE OPERAND ENUMS & DATACLASSES
# =============================================================================

class ToleranceOperandType(str, Enum):
    """Zemax OpticStudio Standard Tolerance Operands."""
    TRAD = "TRAD"  # Tolerance on radius of curvature (mm)
    TCUR = "TCUR"  # Tolerance on curvature (1/mm)
    TTHI = "TTHI"  # Tolerance on thickness or airspace (mm)
    TSDX = "TSDX"  # Tolerance on surface decenter X (mm)
    TSDY = "TSDY"  # Tolerance on surface decenter Y (mm)
    TSTX = "TSTX"  # Tolerance on surface tilt X (deg)
    TSTY = "TSTY"  # Tolerance on surface tilt Y (deg)
    TEDX = "TEDX"  # Tolerance on element decenter X (mm)
    TEDY = "TEDY"  # Tolerance on element decenter Y (mm)
    TETX = "TETX"  # Tolerance on element tilt X (deg)
    TETY = "TETY"  # Tolerance on element tilt Y (deg)
    TIND = "TIND"  # Tolerance on refractive index (delta n)
    TABB = "TABB"  # Tolerance on Abbe number (% delta V_d)
    TCON = "TCON"  # Tolerance on conic constant (delta k)
    TIRR = "TIRR"  # Tolerance on surface irregularity (fringes)
    COMP = "COMP"  # Compensator definition


class ToleranceDistribution(str, Enum):
    """Tolerance Probability Distributions."""
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    NORMAL = "normal"
    PARABOLIC = "parabolic"


class TolerancingCriterion(str, Enum):
    """Optical Performance Criteria for Tolerancing."""
    RMS_SPOT_RADIUS = "rms_spot_radius"  # Centroid RMS spot radius (mm)
    GEO_SPOT_RADIUS = "geo_spot_radius"  # 100% geometric spot circle (mm)
    EFFL = "effl"                        # Effective focal length (mm)
    BFL = "bfl"                          # Back focal length (mm)
    MERIT_FUNCTION = "merit_function"    # Generic merit function value


@dataclass
class PerturbationSpec:
    """
    Specification for perturbing an optical parameter in sensitivity or Monte Carlo tolerancing.
    Fully backward-compatible with original 2D Optics API and expandable to Zemax operands.
    """
    parameter_name: str
    element_index: int = 0
    delta: float = 0.0
    distribution: str = "uniform"  # 'uniform', 'normal', 'gaussian', 'parabolic'
    notes: str = ""
    min_delta: Optional[float] = None
    max_delta: Optional[float] = None
    surface_index: Optional[int] = None
    is_compensator: bool = False
    comp_min: float = -2.0
    comp_max: float = 2.0

    def __post_init__(self) -> None:
        self.parameter_name = str(self.parameter_name).strip()
        self.element_index = int(self.element_index)
        self.delta = float(self.delta)
        self.distribution = str(self.distribution).strip().lower()
        if self.min_delta is None:
            self.min_delta = -abs(self.delta)
        if self.max_delta is None:
            self.max_delta = abs(self.delta)
        if self.surface_index is None:
            self.surface_index = self.element_index


# Alias for Zemax OpticStudio operand
ZemaxToleranceOperand = PerturbationSpec


@dataclass
class SensitivityEntry:
    """
    Univariate sensitivity result for a single perturbed optical parameter.
    """
    parameter_name: str
    element_index: int
    nominal_value: float
    delta: float
    nominal_merit: float
    perturbed_merit: float
    delta_merit: float
    sensitivity: float
    operand_type: str = "CUSTOM"
    surface_index: int = 0
    delta_plus_merit: float = 0.0
    delta_minus_merit: float = 0.0
    compensator_adj_plus: float = 0.0
    compensator_adj_minus: float = 0.0

    @property
    def abs_delta_merit(self) -> float:
        """Absolute magnitude of merit function degradation."""
        return abs(self.delta_merit)

    @property
    def abs_sensitivity(self) -> float:
        """Absolute parameter sensitivity."""
        return abs(self.sensitivity)

    @property
    def max_delta_merit(self) -> float:
        """Maximum absolute merit degradation between +/- perturbations."""
        return max(abs(self.delta_plus_merit), abs(self.delta_minus_merit), abs(self.delta_merit))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "operand_type": self.operand_type,
            "element_index": self.element_index,
            "surface_index": self.surface_index,
            "nominal_value": self.nominal_value,
            "delta": self.delta,
            "nominal_merit": self.nominal_merit,
            "perturbed_merit": self.perturbed_merit,
            "delta_merit": self.delta_merit,
            "sensitivity": self.sensitivity,
            "abs_delta_merit": self.abs_delta_merit,
            "max_delta_merit": self.max_delta_merit,
            "compensator_adj_plus": self.compensator_adj_plus,
            "compensator_adj_minus": self.compensator_adj_minus,
        }


# Alias for Zemax Sensitivity Entry
ZemaxSensitivityEntry = SensitivityEntry


@dataclass
class SensitivityReport:
    """
    Composite report of univariate sensitivity analysis across all specified tolerances.
    Includes Zemax OpticStudio Root-Sum-Square (RSS) and Worst-Case estimates.
    """
    entries: List[SensitivityEntry] = field(default_factory=list)
    nominal_merit: float = 0.0
    criterion_name: str = "Merit Function"
    estimated_worst_case: float = 0.0
    estimated_rss: float = 0.0
    estimated_worst_case_delta: float = 0.0
    estimated_rss_delta: float = 0.0

    @property
    def is_spot_criterion(self) -> bool:
        c_low = self.criterion_name.lower()
        return "spot" in c_low or "trac" in c_low or "rsce" in c_low or "geor" in c_low

    @property
    def nominal_merit_um(self) -> float:
        return self.nominal_merit * 1000.0 if self.is_spot_criterion else self.nominal_merit

    @property
    def estimated_rss_um(self) -> float:
        return self.estimated_rss * 1000.0 if self.is_spot_criterion else self.estimated_rss

    @property
    def estimated_worst_case_um(self) -> float:
        return self.estimated_worst_case * 1000.0 if self.is_spot_criterion else self.estimated_worst_case

    def ranked_worst_offenders(self) -> List[SensitivityEntry]:
        """Return sensitivity entries ranked descending by maximum absolute merit degradation."""
        return sorted(self.entries, key=lambda e: e.max_delta_merit, reverse=True)

    @property
    def rankings(self) -> List[SensitivityEntry]:
        return self.ranked_worst_offenders()

    @property
    def items(self) -> List[SensitivityEntry]:
        return self.ranked_worst_offenders()

    def format_table(self) -> str:
        """Format an ASCII sensitivity analysis table matching Zemax OpticStudio output."""
        ranked = self.ranked_worst_offenders()
        is_spot = self.is_spot_criterion
        nom_um_str = f" ({self.nominal_merit * 1000.0:.2f} \u03bcm)" if is_spot else ""
        lines = [
            "=" * 112,
            f"ZEMAX OPTICSTUDIO TOLERANCE SENSITIVITY REPORT",
            f"Criterion: {self.criterion_name} | Baseline Nominal Value: {self.nominal_merit:.6f} mm{nom_um_str}",
            "=" * 112,
            f"{'Rank':<5} {'Type':<6} {'Surf':<5} {'Parameter':<12} {'Nominal':<10} {'Delta':<9} "
            f"{'Delta M+':<12} {'Delta M-':<12} {'Max Delta':<12} {'Max (\u03bcm)':<10} {'Comp Adj':<12}",
            "-" * 112,
        ]
        for idx, e in enumerate(ranked, start=1):
            comp_str = f"{e.compensator_adj_plus:+.3f}/{e.compensator_adj_minus:+.3f}" if abs(e.compensator_adj_plus) > 1e-5 or abs(e.compensator_adj_minus) > 1e-5 else "---"
            max_um = f"{e.max_delta_merit * 1000.0:.2f}" if is_spot else f"{e.max_delta_merit:.4g}"
            lines.append(
                f"{idx:<5} {e.operand_type:<6} {e.surface_index:<5} {e.parameter_name:<12} {e.nominal_value:<10.4g} "
                f"{e.delta:<+9.3g} {e.delta_plus_merit:<+12.6f} {e.delta_minus_merit:<+12.6f} {e.max_delta_merit:<12.6f} {max_um:<10} {comp_str:<12}"
            )
        lines.append("=" * 112)
        lines.append(f"STATISTICAL ESTIMATED PERFORMANCE DEGRADATION:")
        if is_spot:
            lines.append(f"  Nominal Baseline Criterion   : {self.nominal_merit:12.6f} mm  ({self.nominal_merit * 1000.0:8.2f} \u03bcm)")
            lines.append(f"  Estimated Worst-Case Change  : {self.estimated_worst_case_delta:+12.6f} mm -> Worst-Case: {self.estimated_worst_case:12.6f} mm ({self.estimated_worst_case * 1000.0:8.2f} \u03bcm)")
            lines.append(f"  Estimated RSS Change (RMS)   : {self.estimated_rss_delta:+12.6f} mm -> RSS Total  : {self.estimated_rss:12.6f} mm ({self.estimated_rss * 1000.0:8.2f} \u03bcm)")
        else:
            lines.append(f"  Nominal Baseline Criterion   : {self.nominal_merit:12.6f}")
            lines.append(f"  Estimated Worst-Case Change  : {self.estimated_worst_case_delta:+12.6f} -> Worst-Case: {self.estimated_worst_case:12.6f}")
            lines.append(f"  Estimated RSS Change (RMS)   : {self.estimated_rss_delta:+12.6f} -> RSS Total  : {self.estimated_rss:12.6f}")
        lines.append("=" * 112)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, idx: int) -> SensitivityEntry:
        return self.entries[idx]


# Alias for Zemax Sensitivity Report
ZemaxSensitivityReport = SensitivityReport


@dataclass
class InverseSensitivityEntry:
    """
    Zemax Inverse Sensitivity result for a single tolerance operand.
    Computes the maximum allowable +/- parameter tolerance that keeps criterion
    degradation within max_degradation.
    """
    parameter_name: str
    operand_type: str
    surface_index: int
    nominal_value: float
    initial_delta: float
    allowable_delta: float = 0.0
    allowable_min_delta: float = 0.0
    allowable_max_delta: float = 0.0
    achieved_degradation: float = 0.0
    distribution: str = "uniform"
    notes: str = ""
    element_index: Optional[int] = None
    solved_plus_delta: Optional[float] = None
    solved_minus_delta: Optional[float] = None
    solved_symmetric_delta: Optional[float] = None
    target_degradation: float = 0.0
    achieved_plus_degradation: float = 0.0
    achieved_minus_degradation: float = 0.0

    def __post_init__(self) -> None:
        if self.element_index is None:
            self.element_index = self.surface_index
        if self.solved_plus_delta is not None and self.allowable_max_delta == 0.0:
            self.allowable_max_delta = float(self.solved_plus_delta)
        elif self.solved_plus_delta is None:
            self.solved_plus_delta = float(self.allowable_max_delta)

        if self.solved_minus_delta is not None and self.allowable_min_delta == 0.0:
            self.allowable_min_delta = float(self.solved_minus_delta)
        elif self.solved_minus_delta is None:
            self.solved_minus_delta = float(self.allowable_min_delta)

        if self.solved_symmetric_delta is not None and self.allowable_delta == 0.0:
            self.allowable_delta = float(self.solved_symmetric_delta)
        elif self.solved_symmetric_delta is None:
            self.solved_symmetric_delta = float(self.allowable_delta)

        if self.achieved_degradation == 0.0 and (self.achieved_plus_degradation != 0.0 or self.achieved_minus_degradation != 0.0):
            self.achieved_degradation = max(abs(self.achieved_plus_degradation), abs(self.achieved_minus_degradation))

    def to_perturbation_spec(self) -> PerturbationSpec:
        return PerturbationSpec(
            parameter_name=self.parameter_name,
            element_index=self.surface_index,
            surface_index=self.surface_index,
            delta=self.allowable_delta,
            min_delta=self.allowable_min_delta,
            max_delta=self.allowable_max_delta,
            distribution=self.distribution,
            notes=self.notes or f"InvSens {self.operand_type} S{self.surface_index}",
        )


ZemaxInverseSensitivityEntry = InverseSensitivityEntry


@dataclass
class InverseSensitivityReport:
    """
    Report for Zemax Inverse Sensitivity analysis where allowable tolerances are solved
    for a target performance degradation limit.
    """
    entries: List[InverseSensitivityEntry] = field(default_factory=list)
    nominal_merit: float = 0.0
    max_degradation_limit: float = 0.0
    target_degradation: Optional[float] = None
    criterion_name: str = "RMS Spot Radius"

    def __post_init__(self) -> None:
        if self.target_degradation is not None and self.max_degradation_limit == 0.0:
            self.max_degradation_limit = float(self.target_degradation)
        elif self.target_degradation is None:
            self.target_degradation = float(self.max_degradation_limit)

    @property
    def is_spot_criterion(self) -> bool:
        c_low = self.criterion_name.lower()
        return "spot" in c_low or "trac" in c_low or "rsce" in c_low or "geor" in c_low

    def to_tolerance_list(self) -> List[PerturbationSpec]:
        """Convert solved inverse sensitivity limits back into PerturbationSpec list."""
        return [e.to_perturbation_spec() for e in self.entries]

    def format_table(self) -> str:
        is_spot = self.is_spot_criterion
        deg_str = f"{self.max_degradation_limit:.6f} mm ({self.max_degradation_limit * 1000.0:.2f} \u03bcm)" if is_spot else f"{self.max_degradation_limit:.6f}"
        nom_str = f"{self.nominal_merit:.6f} mm ({self.nominal_merit * 1000.0:.2f} \u03bcm)" if is_spot else f"{self.nominal_merit:.6f}"
        lines = [
            "=" * 105,
            "ZEMAX OPTICSTUDIO INVERSE SENSITIVITY REPORT",
            f"Criterion: {self.criterion_name} | Nominal: {nom_str} | Max Allowed Degradation: {deg_str}",
            "=" * 105,
            f"{'Surf':<5} {'Operand':<8} {'Parameter':<12} {'Nominal':<12} {'Init Tol':<12} {'Solved Limit (+/-)':<18} {'Degradation (\u03bcm)':<16}",
            "-" * 105,
        ]
        for e in self.entries:
            deg_um = f"{e.achieved_degradation * 1000.0:.2f} \u03bcm" if is_spot else f"{e.achieved_degradation:.5f}"
            lines.append(
                f"{e.surface_index:<5} {e.operand_type:<8} {e.parameter_name:<12} {e.nominal_value:<12.4g} "
                f"{e.initial_delta:<+12.4g} {e.allowable_delta:<+18.4g} {deg_um:<16}"
            )
        lines.append("=" * 105)
        return "\n".join(lines)


ZemaxInverseSensitivityReport = InverseSensitivityReport


# =============================================================================
# 2. MONTE CARLO TRIAL RECORD & REPORT
# =============================================================================

@dataclass
class MonteCarloTrialRecord:
    """
    Result of an individual randomized multi-parameter trial in Monte Carlo tolerancing.
    """
    trial_index: int
    perturbations: Dict[str, float]
    merit: float
    efl: float
    bfl: float
    compensator_adjustment: float = 0.0
    spot_radius_rms: float = 0.0
    spot_radius_geo: float = 0.0

    @property
    def focal_length(self) -> float:
        return self.efl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "perturbations": self.perturbations,
            "merit": self.merit,
            "efl": self.efl,
            "bfl": self.bfl,
            "compensator_adjustment": self.compensator_adjustment,
            "spot_radius_rms": self.spot_radius_rms,
            "spot_radius_geo": self.spot_radius_geo,
        }


# Alias for Zemax Monte Carlo Trial
ZemaxMonteCarloTrial = MonteCarloTrialRecord


@dataclass
class MonteCarloReport:
    """
    Statistical compilation of multivariate Monte Carlo tolerancing simulation trials.
    Provides cumulative percentiles (P50, P80, P90, P95, P98, P99) and yield rates.
    """
    trials: List[MonteCarloTrialRecord] = field(default_factory=list)
    num_trials: int = 0
    nominal_merit: float = 0.0
    mean_merit: float = 0.0
    std_merit: float = 0.0
    min_merit: float = 0.0
    max_merit: float = 0.0
    percentiles: Dict[int, float] = field(default_factory=dict)
    mean_efl: float = 0.0
    std_efl: float = 0.0
    mean_bfl: float = 0.0
    std_bfl: float = 0.0
    mean_compensator: float = 0.0
    std_compensator: float = 0.0
    criterion_name: str = "Merit Function"
    best_trial_index: int = 0
    worst_trial_index: int = 0

    @property
    def is_spot_criterion(self) -> bool:
        c_low = self.criterion_name.lower()
        return "spot" in c_low or "trac" in c_low or "rsce" in c_low or "geor" in c_low

    @property
    def percentile_50(self) -> float:
        return self.percentiles.get(50, self.mean_merit)

    @property
    def percentile_80(self) -> float:
        return self.percentiles.get(80, self.mean_merit)

    @property
    def percentile_90(self) -> float:
        return self.percentiles.get(90, self.mean_merit)

    @property
    def percentile_95(self) -> float:
        return self.percentiles.get(95, self.mean_merit)

    @property
    def percentile_98(self) -> float:
        return self.percentiles.get(98, self.mean_merit)

    @property
    def percentile_99(self) -> float:
        return self.percentiles.get(99, self.mean_merit)

    @property
    def p50(self) -> float: return self.percentile_50
    @property
    def p80(self) -> float: return self.percentile_80
    @property
    def p90(self) -> float: return self.percentile_90
    @property
    def p95(self) -> float: return self.percentile_95
    @property
    def p98(self) -> float: return self.percentile_98
    @property
    def p99(self) -> float: return self.percentile_99

    @property
    def p50_um(self) -> float: return self.p50 * 1000.0 if self.is_spot_criterion else self.p50
    @property
    def p90_um(self) -> float: return self.p90 * 1000.0 if self.is_spot_criterion else self.p90
    @property
    def p95_um(self) -> float: return self.p95 * 1000.0 if self.is_spot_criterion else self.p95
    @property
    def p99_um(self) -> float: return self.p99 * 1000.0 if self.is_spot_criterion else self.p99

    def yield_rate(self, threshold: float) -> float:
        """Compute fraction of manufactured systems whose criterion satisfies criterion <= threshold."""
        if not self.trials:
            return 0.0
        passed = sum(1 for t in self.trials if t.merit <= threshold)
        return float(passed / len(self.trials))

    def yield_percentage(self, threshold: float) -> float:
        """Yield rate as percentage [0% to 100%]."""
        return self.yield_rate(threshold) * 100.0

    def format_report(self) -> str:
        """Format an ASCII report of Monte Carlo simulation distributions in Zemax style."""
        is_spot = self.is_spot_criterion
        def _fmt(val: float) -> str:
            return f"{val:14.6f} mm ({val * 1000.0:8.2f} \u03bcm)" if is_spot else f"{val:14.6f}"

        lines = [
            "=" * 88,
            f"ZEMAX OPTICSTUDIO MONTE CARLO TOLERANCING REPORT",
            f"Trials Executed: {self.num_trials} | Criterion: {self.criterion_name}",
            "=" * 88,
            f"Nominal Baseline Criterion:  {_fmt(self.nominal_merit)}",
            f"Mean Simulated Criterion:    {_fmt(self.mean_merit)} +/- {self.std_merit:10.6f}",
            f"Minimum Criterion (Best):    {_fmt(self.min_merit)} (Trial #{self.best_trial_index})",
            f"Maximum Criterion (Worst):   {_fmt(self.max_merit)} (Trial #{self.worst_trial_index})",
            "-" * 88,
            "CUMULATIVE PROBABILITY DISTRIBUTION (PERCENTILES):",
            f"  50% of systems <= {_fmt(self.p50)} (Median Performance)",
            f"  80% of systems <= {_fmt(self.p80)}",
            f"  90% of systems <= {_fmt(self.p90)} (90% Confidence Limit)",
            f"  95% of systems <= {_fmt(self.p95)} (95% Confidence Limit)",
            f"  98% of systems <= {_fmt(self.p98)}",
            f"  99% of systems <= {_fmt(self.p99)} (3-Sigma Fabrication Limit)",
            "-" * 88,
            "OPTICAL PARAMETERS & COMPENSATOR ADJUSTMENTS:",
            f"  EFL (Effective Focal Length) : {self.mean_efl:12.4f} +/- {self.std_efl:10.4f} mm",
            f"  BFL (Back Focal Length)      : {self.mean_bfl:12.4f} +/- {self.std_bfl:10.4f} mm",
        ]
        if abs(self.mean_compensator) > 1e-6 or abs(self.std_compensator) > 1e-6:
            lines.append(f"  Compensator Shift (Refocus)  : {self.mean_compensator:12.4f} +/- {self.std_compensator:10.4f} mm ({self.mean_compensator * 1000.0:+.1f} \u03bcm)")
        lines.append("=" * 88)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.trials)

    def __iter__(self):
        return iter(self.trials)

    def __getitem__(self, idx: int) -> MonteCarloTrialRecord:
        return self.trials[idx]


# Alias for Zemax Monte Carlo Report
ZemaxMonteCarloReport = MonteCarloReport


# =============================================================================
# 3. 3x3 AUGMENTED TRANSFER MATRIX PERTURBATION HELPERS
# =============================================================================

def decenter_matrix_3x3(delta_y: float) -> np.ndarray:
    """Generate 3x3 lateral decenter deflection matrix."""
    return np.array([
        [1.0, 0.0, float(delta_y)],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def tilt_matrix_3x3(delta_theta: float) -> np.ndarray:
    """Generate 3x3 angular tilt deflection matrix."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, float(delta_theta)],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def augmented_transfer_matrix_3x3(
    matrix_or_element: Any,
    decenter_y: float = 0.0,
    tilt_theta: float = 0.0,
) -> np.ndarray:
    """
    Construct 3x3 augmented transfer matrix incorporating decenter and tilt offsets.
    Standard formulation (Greivenkamp / Zemax OpticStudio):
        [y_out]   [ A  B  E ] [y_in]
        [u_out] = [ C  D  F ] [u_in]
        [  1  ]   [ 0  0  1 ] [  1 ]
    where:
        E = (1 - A) * Delta_y - B * Delta_theta
        F = -C * Delta_y + (1 - D) * Delta_theta
    """
    if hasattr(matrix_or_element, "transfer_matrix") and callable(matrix_or_element.transfer_matrix):
        mat = matrix_or_element.transfer_matrix()
    elif isinstance(matrix_or_element, TransferMatrix):
        mat = matrix_or_element
    elif isinstance(matrix_or_element, np.ndarray) and matrix_or_element.shape == (2, 2):
        mat = TransferMatrix.from_2x2(matrix_or_element)
    elif isinstance(matrix_or_element, np.ndarray) and matrix_or_element.shape == (3, 3):
        return matrix_or_element
    else:
        raise TypeError(f"Cannot extract transfer matrix from {type(matrix_or_element)}")

    A = float(mat.A)
    B = float(mat.B)
    C = float(mat.C)
    D = float(mat.D)

    E = (1.0 - A) * float(decenter_y) - B * float(tilt_theta)
    F = -C * float(decenter_y) + (1.0 - D) * float(tilt_theta)

    return np.array([
        [A, B, E],
        [C, D, F],
        [0.0, 0.0, 1.0],
    ], dtype=float)


# =============================================================================
# 4. RANDOM DISTRIBUTION SAMPLING (UNIFORM, GAUSSIAN, PARABOLIC)
# =============================================================================

def sample_tolerance_delta(
    delta: float,
    distribution: str = "uniform",
    rng: Optional[np.random.Generator] = None,
    min_delta: Optional[float] = None,
    max_delta: Optional[float] = None,
) -> float:
    """
    Sample a random perturbation increment according to specified Zemax probability distribution.

    Args:
        delta: Half-width magnitude (or 3-sigma limit for gaussian).
        distribution: 'uniform', 'normal', 'gaussian', or 'parabolic'.
        rng: NumPy random generator.
        min_delta: Optional asymmetric lower limit (defaults to -delta).
        max_delta: Optional asymmetric upper limit (defaults to +delta).
    """
    if rng is None:
        rng = np.random.default_rng()

    dist = str(distribution).lower().strip()
    low = min_delta if min_delta is not None else -abs(delta)
    high = max_delta if max_delta is not None else abs(delta)

    if abs(high - low) < 1e-15:
        return 0.0

    if dist in ("normal", "gaussian"):
        # By Zemax convention, delta represents 3-sigma limit
        sigma = abs(delta) / 3.0 if abs(delta) > 0.0 else 1.0
        val = float(rng.normal(loc=0.0, scale=sigma))
        return float(np.clip(val, low, high))

    elif dist == "parabolic":
        # Zemax parabolic density: P(u) ~ 1 - u^2 on [-1, +1]
        half_w = max(abs(low), abs(high))
        while True:
            u = float(rng.uniform(-1.0, 1.0))
            v = float(rng.uniform(0.0, 1.0))
            if v <= (1.0 - u * u):
                val = u * half_w
                return float(np.clip(val, low, high))

    else:
        # Uniform distribution
        return float(rng.uniform(low=low, high=high))


# =============================================================================
# 5. OPTICAL PERFORMANCE & RAY TRACE CRITERION EVALUATION
# =============================================================================

def compute_system_spot_radius(
    system: Any,
    num_rings: int = 4,
    rays_per_ring: int = 8,
    wavelength_um: float = 0.5876,
    geometric: bool = False,
) -> float:
    """
    Compute centroid RMS spot radius (TRAC) or 100% geometric spot circle (GEOR)
    by tracing a 2D polar pupil grid through the optical system to the detector.
    """
    # 1. SurfaceStack ray tracing
    stack = None
    if isinstance(system, SurfaceStack):
        stack = system
    elif hasattr(system, "stack") and isinstance(system.stack, SurfaceStack):
        stack = system.stack
    elif hasattr(system, "surfaces") and isinstance(system.surfaces, list):
        if hasattr(system, "_update_accumulated_positions"):
            stack = system
        else:
            stack = SurfaceStack()
            for s in system.surfaces:
                stack.add_surface(s)

    if stack is not None:
        stack._update_accumulated_positions()
        opt = OpticalSystem(stack)

        first_optical = None
        for s in stack.surfaces:
            if s.surface_type not in (SurfaceType.OBJECT, SurfaceType.COORDINATE_BREAK):
                first_optical = s
                break
        semi_diam = float(getattr(first_optical, "semi_diameter", 12.5)) if first_optical else 12.5
        if semi_diam <= 0.0:
            semi_diam = 12.5

        x_hits: List[float] = []
        y_hits: List[float] = []

        # Chief ray on axis
        r0 = Ray(pos=np.array([0.0, 0.0, -1.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wavelength_um)
        tr0 = opt.trace_ray(r0, track_polarization=False)
        if not tr0.vignetted and tr0.history:
            x_hits.append(float(tr0.history[-1]["x"]))
            y_hits.append(float(tr0.history[-1]["y"]))

        # Concentric polar pupil sampling
        ring_fractions = np.linspace(0.2, 0.95, num_rings)
        for r_frac in ring_fractions:
            r = r_frac * semi_diam
            for th in np.linspace(0, 2 * np.pi, rays_per_ring, endpoint=False):
                px = float(r * np.cos(th))
                py = float(r * np.sin(th))
                ray = Ray(pos=np.array([px, py, -1.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wavelength_um)
                tr = opt.trace_ray(ray, track_polarization=False)
                if not tr.vignetted and tr.history:
                    x_hits.append(float(tr.history[-1]["x"]))
                    y_hits.append(float(tr.history[-1]["y"]))

        if len(x_hits) < 4:
            return 999.0

        x_arr = np.array(x_hits, dtype=float)
        y_arr = np.array(y_hits, dtype=float)
        x_mean = np.mean(x_arr)
        y_mean = np.mean(y_arr)
        devs_sq = (x_arr - x_mean) ** 2 + (y_arr - y_mean) ** 2

        if geometric:
            return float(np.sqrt(np.max(devs_sq)))
        return float(np.sqrt(np.mean(devs_sq)))

    # 2. ThickLens / OpticalElement fallback (via paraxial aberrations)
    if hasattr(system, "transfer_matrix"):
        mat = system.transfer_matrix()
        c = abs(mat.C)
        efl = 1.0 / c if c > 1e-12 else 50.0
        diam = float(getattr(system, "diameter", 25.4))
        f_num = efl / diam if diam > 0 else 2.0
        # Spherical aberration spot approximation: delta_y ~ 0.067 * y^3 / (f^2)
        return float(0.067 * (diam / 2.0) ** 3 / (efl ** 2))

    return 0.1


def evaluate_tolerancing_criterion(
    system: Any,
    criterion: Union[TolerancingCriterion, str] = TolerancingCriterion.RMS_SPOT_RADIUS,
    merit_function: Optional[Any] = None,
    wavelength_um: float = 0.5876,
) -> float:
    """Evaluate performance criterion against optical system."""
    crit_str = str(criterion.value if hasattr(criterion, "value") else criterion).lower().strip()

    if crit_str in ("rms_spot_radius", "rms_spot", "trac", "rsce"):
        return compute_system_spot_radius(system, wavelength_um=wavelength_um, geometric=False)

    elif crit_str in ("geo_spot_radius", "geo_spot", "geor", "spot"):
        return compute_system_spot_radius(system, wavelength_um=wavelength_um, geometric=True)

    elif crit_str in ("effl", "focal_length", "efl"):
        efl, _ = extract_system_cardinal_metrics(system)
        return float(abs(efl))

    elif crit_str in ("bfl", "back_focal_length"):
        _, bfl = extract_system_cardinal_metrics(system)
        return float(abs(bfl))

    elif crit_str in ("merit_function", "merit") or merit_function is not None:
        if merit_function is not None:
            return evaluate_system_merit(merit_function, system)

    # Default fallback
    return compute_system_spot_radius(system, wavelength_um=wavelength_um)


# =============================================================================
# 6. SYSTEM INSPECTION, PERTURBATION & CLONING UTILITIES
# =============================================================================

def clone_optical_system(system: Any) -> Any:
    """Deep-clone an optical system to guarantee pre-simulation nominal immutability."""
    if hasattr(system, "clone") and callable(system.clone):
        try:
            return system.clone()
        except Exception:
            pass

    try:
        return copy.deepcopy(system)
    except Exception:
        pass

    if isinstance(system, ThickLens):
        return ThickLens(
            r1=system.r1,
            r2=system.r2,
            thickness=system.thickness,
            glass_or_index=system.glass_or_index,
            diameter=system.diameter,
            decenter=system.decenter,
            tilt=system.tilt,
            name=system.name,
        )

    return copy.copy(system)


def _resolve_target_element(system: Any, element_index: int = 0) -> Any:
    """Retrieve element or surface at element_index from system."""
    if hasattr(system, "surfaces") and isinstance(system.surfaces, list):
        if 0 <= element_index < len(system.surfaces):
            return system.surfaces[element_index]
    if hasattr(system, "components") and isinstance(system.components, list):
        if 0 <= element_index < len(system.components):
            return system.components[element_index]
    if hasattr(system, "_elements") and isinstance(system._elements, list):
        if 0 <= element_index < len(system._elements):
            return system._elements[element_index]
    return system


def get_element_parameter(elem: Any, param_name: str) -> float:
    """Read physical parameter value from an optical element or Surface."""
    key = param_name.lower().strip()

    # Surface object lookups
    if hasattr(elem, "radius") and key in ("radius", "r", "r1", "curv", "c", "trad", "tcur"):
        r = float(getattr(elem, "radius", 0.0))
        if key in ("curv", "c", "tcur"):
            return 1.0 / r if abs(r) > 1e-12 else 0.0
        return r

    if hasattr(elem, "thickness") and key in ("thickness", "t", "tthi", "air_gap"):
        return float(getattr(elem, "thickness", 0.0))

    if hasattr(elem, "glass") and key in ("index", "n", "tind"):
        g = str(getattr(elem, "glass", "AIR")).upper()
        if "INDEX_" in g:
            try: return float(g.split("INDEX_")[1])
            except Exception: pass
        if g in ("AIR", ""): return 1.0
        if g == "MIRROR": return 1.0
        try:
            from optics_app.utils.materials import get_refractive_index
            return float(get_refractive_index(g, 0.5876))
        except Exception:
            return 1.5168

    if hasattr(elem, "glass") and key in ("abbe", "vd", "tabb"):
        if hasattr(elem, "_abbe_override") and getattr(elem, "_abbe_override") is not None:
            return float(getattr(elem, "_abbe_override"))
        g = str(getattr(elem, "glass", "AIR")).upper()
        if g in ("AIR", "MIRROR", "") or "INDEX_" in g:
            return float(getattr(elem, "_abbe_override", 64.17))
        try:
            from optics_app.utils.materials import GlassCatalog
            vd = float(GlassCatalog.calculate_abbe_number(g))
            return vd if math.isfinite(vd) else 64.17
        except Exception:
            return 64.17

    if key in ("tirr", "irregularity", "fringes"):
        return float(getattr(elem, "irregularity_fringes", 0.0))

    if hasattr(elem, "conic") and key in ("conic", "k", "tcon"):
        return float(getattr(elem, "conic", 0.0))

    if key in ("decenter", "decenter_y", "dy", "tsdy", "tedy"):
        return float(getattr(elem, "decenter_y", getattr(elem, "decenter", 0.0)))

    if key in ("decenter_x", "dx", "tsdx", "tedx"):
        return float(getattr(elem, "decenter_x", 0.0))

    if key in ("tilt", "tilt_x", "tstx", "tetx", "theta"):
        return float(getattr(elem, "tilt_x", getattr(elem, "tilt", 0.0)))

    if key in ("tilt_y", "tsty", "tety"):
        return float(getattr(elem, "tilt_y", 0.0))

    # ThickLens / OpticalElement methods
    if hasattr(elem, "get_parameter") and callable(elem.get_parameter):
        try: return float(elem.get_parameter(param_name))
        except Exception: pass

    if hasattr(elem, "get_parameters") and callable(elem.get_parameters):
        params = elem.get_parameters()
        for k, v in params.items():
            if k.lower().strip() == key:
                return float(v)

    # Standard attribute lookups
    alias_map = {
        "r1": ["r1", "radius_front", "radius1", "c1", "radius"],
        "r2": ["r2", "radius_rear", "radius2", "c2"],
        "thickness": ["thickness", "t", "center_thickness"],
        "index": ["index", "glass_or_index", "n", "refractive_index"],
        "diameter": ["diameter", "diam", "d", "aperture"],
        "decenter": ["decenter", "decenter_y", "dy"],
        "tilt": ["tilt", "tilt_x", "theta", "tilt_theta"],
    }
    for cand in alias_map.get(key, [key]):
        if hasattr(elem, cand):
            val = getattr(elem, cand)
            if callable(val):
                try: return float(val())
                except Exception: continue
            if isinstance(val, (int, float)):
                return float(val)

    return 0.0


def perturb_element_parameter(elem: Any, param_name: str, delta: float, system: Optional[Any] = None) -> None:
    """Apply parameter perturbation delta to an element or Surface."""
    key = param_name.lower().strip()

    # Element-level rigid decenters and tilts (TEDX, TEDY, TETX, TETY) MUST be handled before surface-level fallbacks
    if system is not None and hasattr(system, "displace_element") and key in ("tedx", "tedy", "element_decenter"):
        surfs = getattr(system, "surfaces", [])
        if elem in surfs:
            elem_idx: Any = surfs.index(elem)
        else:
            elem_idx = getattr(elem, "element_id", None)
            if elem_idx is None:
                elem_idx = 1
        curr_dx = float(getattr(elem, "decenter_x", 0.0))
        curr_dy = float(getattr(elem, "decenter_y", 0.0))
        dx = curr_dx + (delta if "x" in key else 0.0)
        dy = curr_dy + (delta if "y" in key or key == "element_decenter" else 0.0)
        system.displace_element(elem_idx, decenter_x=dx, decenter_y=dy)
        return

    if system is not None and hasattr(system, "rotate_element") and key in ("tetx", "tety", "element_tilt"):
        surfs = getattr(system, "surfaces", [])
        if elem in surfs:
            elem_idx = surfs.index(elem)
        else:
            elem_idx = getattr(elem, "element_id", None)
            if elem_idx is None:
                elem_idx = 1
        curr_tx = float(getattr(elem, "tilt_x", 0.0))
        curr_ty = float(getattr(elem, "tilt_y", 0.0))
        tx = curr_tx + (delta if "x" in key or key == "element_tilt" else 0.0)
        ty = curr_ty + (delta if "y" in key else 0.0)
        system.rotate_element(elem_idx, tilt_x=tx, tilt_y=ty)
        return

    # Surface object direct perturbations
    if hasattr(elem, "radius") and key in ("radius", "r", "r1", "trad"):
        r = float(getattr(elem, "radius", 0.0))
        if abs(r) > 1e-12:
            setattr(elem, "radius", r + delta)
        return

    if hasattr(elem, "radius") and key in ("curv", "c", "tcur"):
        r = float(getattr(elem, "radius", 0.0))
        c0 = 1.0 / r if abs(r) > 1e-12 else 0.0
        c_new = c0 + delta
        setattr(elem, "radius", 1.0 / c_new if abs(c_new) > 1e-12 else 0.0)
        return

    if hasattr(elem, "thickness") and key in ("thickness", "t", "tthi", "air_gap"):
        t = float(getattr(elem, "thickness", 0.0))
        setattr(elem, "thickness", max(0.001, t + delta))
        return

    if hasattr(elem, "conic") and key in ("conic", "k", "tcon"):
        k = float(getattr(elem, "conic", 0.0))
        setattr(elem, "conic", k + delta)
        return

    if key in ("tirr", "irregularity", "fringes"):
        curr_irr = float(getattr(elem, "irregularity_fringes", 0.0))
        setattr(elem, "irregularity_fringes", curr_irr + delta)
        if hasattr(elem, "radius"):
            sd = float(getattr(elem, "semi_diameter", 12.5))
            if sd <= 0.0:
                sd = 12.5
            # 1 fringe = lambda/2 sag over semi_diameter -> delta_c = 2 * delta_z / sd^2 = N * lambda / sd^2
            wl_mm = 0.0005876
            delta_c = (delta * wl_mm) / (sd ** 2)
            r = float(getattr(elem, "radius", 0.0))
            c0 = 1.0 / r if abs(r) > 1e-12 else 0.0
            c_new = c0 + delta_c
            setattr(elem, "radius", 1.0 / c_new if abs(c_new) > 1e-12 else 0.0)
        return

    if key in ("abbe", "vd", "tabb") and hasattr(elem, "glass"):
        curr_vd = get_element_parameter(elem, "abbe")
        curr_n = get_element_parameter(elem, "index")
        # In Zemax TABB, delta is % change in Vd if |delta| <= 5.0 else absolute Vd shift
        rel_shift = (delta / 100.0) if abs(delta) <= 5.0 else (delta / max(10.0, curr_vd))
        new_vd = max(5.0, curr_vd * (1.0 + rel_shift))
        setattr(elem, "_abbe_override", new_vd)
        # Chromatic dispersion coupling into effective refractive index spread across visible band
        if curr_n > 1.01 and curr_vd > 1.0:
            delta_n_disp = -((curr_n - 1.0) / curr_vd) * rel_shift * 0.5
            new_n = max(1.0, curr_n + delta_n_disp)
            setattr(elem, "glass", f"INDEX_{new_n:.6f}")
        return

    if key in ("decenter", "decenter_y", "dy", "tsdy", "tedy"):
        if hasattr(elem, "decenter_y"):
            elem.decenter_y += delta
            return
        elif hasattr(elem, "decenter"):
            elem.decenter += delta
            return

    if key in ("decenter_x", "dx", "tsdx", "tedx") and hasattr(elem, "decenter_x"):
        elem.decenter_x += delta
        return

    if key in ("tilt", "tilt_x", "tstx", "tetx", "theta"):
        if hasattr(elem, "tilt_x"):
            elem.tilt_x += delta
            return
        elif hasattr(elem, "tilt"):
            elem.tilt += delta
            return

    if key in ("tilt_y", "tsty", "tety") and hasattr(elem, "tilt_y"):
        elem.tilt_y += delta
        return

    if key in ("index", "n", "tind") and hasattr(elem, "glass"):
        curr_n = get_element_parameter(elem, "index")
        new_n = max(1.0, curr_n + delta)
        setattr(elem, "glass", f"INDEX_{new_n:.5f}")
        return

    # ThickLens / OpticalElement methods
    if hasattr(elem, "perturb") and callable(elem.perturb):
        try:
            elem.perturb(param_name, delta)
            return
        except Exception:
            pass

    if hasattr(elem, "set_parameter") and callable(elem.set_parameter):
        try:
            curr = get_element_parameter(elem, param_name)
            elem.set_parameter(param_name, curr + delta)
            return
        except Exception:
            pass

    # Direct attribute fallback
    curr = get_element_parameter(elem, param_name)
    alias_map = {
        "r1": "r1",
        "r2": "r2",
        "thickness": "thickness",
        "index": "glass_or_index",
        "diameter": "diameter",
        "decenter": "decenter",
        "tilt": "tilt",
    }
    target_attr = alias_map.get(key, key)
    if hasattr(elem, target_attr):
        setattr(elem, target_attr, curr + delta)
        return


def evaluate_system_merit(merit_function: Any, system: Any) -> float:
    """Evaluate merit function against optical system."""
    if hasattr(merit_function, "evaluate") and callable(merit_function.evaluate):
        return float(merit_function.evaluate(system))
    elif callable(merit_function):
        return float(merit_function(system))
    raise TypeError(f"Merit function {merit_function} does not implement evaluate() or __call__().")


def extract_system_cardinal_metrics(system: Any) -> Tuple[float, float]:
    """Extract (efl, bfl) in millimeters from system."""
    efl = 0.0
    bfl = 0.0

    # 1. SurfaceStack via MatrixTracer
    if isinstance(system, SurfaceStack) or (hasattr(system, "surfaces") and isinstance(system.surfaces, list)):
        try:
            tracer = MatrixTracer()
            mat = tracer.build_from_system(system)
            if abs(mat.C) > 1e-12:
                efl = -1.0 / mat.C
                bfl = -mat.A / mat.C
                return float(efl), float(bfl)
        except Exception:
            pass

    # 2. ThickLens / OpticalElement
    if hasattr(system, "cardinal_points") and callable(system.cardinal_points):
        try:
            cp = system.cardinal_points()
            return float(cp.efl), float(cp.bfl)
        except Exception:
            pass

    if hasattr(system, "efl"):
        try: efl = float(system.efl)
        except Exception: pass
    elif hasattr(system, "focal_length"):
        try: efl = float(system.focal_length)
        except Exception: pass

    if hasattr(system, "bfl"):
        try: bfl = float(system.bfl)
        except Exception: pass

    return efl, bfl


# =============================================================================
# 7. AUTOMATIC COMPENSATOR OPTIMIZATION (REFOCUSING)
# =============================================================================

def solve_system_compensator(
    system: Any,
    compensator_surface_idx: int = -2,
    criterion: Union[TolerancingCriterion, str] = TolerancingCriterion.RMS_SPOT_RADIUS,
    merit_function: Optional[Any] = None,
    delta_range: float = 2.0,
    wavelength_um: float = 0.5876,
) -> float:
    """
    Auto-refocus optical system by optimizing compensator surface thickness to minimize criterion.
    Returns the compensator adjustment (shift from nominal thickness in mm).
    """
    # Verify SurfaceStack
    stack = system if isinstance(system, SurfaceStack) else getattr(system, "stack", None)
    if stack is None or not hasattr(stack, "surfaces") or len(stack.surfaces) < 2:
        return 0.0

    # Adjust index for negative index indexing
    idx = compensator_surface_idx if compensator_surface_idx >= 0 else len(stack.surfaces) + compensator_surface_idx
    if not (0 <= idx < len(stack.surfaces)):
        idx = max(0, len(stack.surfaces) - 2)

    surf = stack.surfaces[idx]
    t0 = float(surf.thickness)

    def objective(t_val: float) -> float:
        surf.thickness = float(t_val)
        return evaluate_tolerancing_criterion(stack, criterion=criterion, merit_function=merit_function, wavelength_um=wavelength_um)

    low_b = max(0.001, t0 - abs(delta_range))
    high_b = t0 + abs(delta_range)

    try:
        res = minimize_scalar(objective, bounds=(low_b, high_b), method="bounded", options={"xatol": 5e-4, "maxiter": 10})
        best_t = float(res.x)
        surf.thickness = best_t
        return best_t - t0
    except Exception:
        surf.thickness = t0
        return 0.0


# =============================================================================
# 8. ZEMAX TOLERANCING ENGINE (SENSITIVITY & MONTE CARLO)
# =============================================================================

class TolerancingEngine:
    """
    Zemax-Style Optical Tolerancing Engine for 2D Optics.

    Provides:
    - Zemax Tolerance Operands (TRAD, TCUR, TTHI, TSDX, TSDY, TSTX, TSTY, TEDX, TEDY, TIND, TABB, TCON).
    - Univariate Sensitivity Analysis with Root-Sum-Square (RSS) and Worst-Case estimates.
    - Multivariate Monte Carlo simulation with Uniform, Gaussian/Normal, and Parabolic distributions.
    - Automatic compensator optimization (auto-refocusing image/detector plane).
    - Full compatibility with SurfaceStack (2D/3D ray tracing) and paraxial OpticalElements.
    """

    def __init__(
        self,
        system: Optional[Any] = None,
        merit_function: Optional[Any] = None,
        random_seed: Optional[int] = None,
        seed: Optional[int] = None,
        criterion: Union[TolerancingCriterion, str] = TolerancingCriterion.RMS_SPOT_RADIUS,
        compensator_enabled: bool = True,
        compensator_surface_idx: int = -2,
        compensator_range_mm: float = 2.0,
    ) -> None:
        self.system = system
        self.merit_function = merit_function
        self.random_seed: Optional[int] = seed if seed is not None else random_seed
        self.criterion = criterion
        self.compensator_enabled = compensator_enabled
        self.compensator_surface_idx = compensator_surface_idx
        self.compensator_range_mm = float(compensator_range_mm)

    def sensitivity_analysis(
        self,
        system: Optional[Any] = None,
        merit_function: Optional[Any] = None,
        tolerances: Optional[Sequence[PerturbationSpec]] = None,
        criterion: Optional[Union[TolerancingCriterion, str]] = None,
        compensator_surface_idx: Optional[int] = None,
        compensator_range_mm: Optional[float] = None,
    ) -> SensitivityReport:
        """
        Execute univariate sensitivity analysis across all tolerance specifications.
        Evaluates both +Delta and -Delta perturbations with optional compensator refocusing.
        Computes Zemax RSS and Worst-Case performance estimates.
        """
        active_system = system if system is not None else self.system
        active_merit = merit_function if merit_function is not None else self.merit_function
        active_tols = list(tolerances) if tolerances is not None else []
        active_crit = criterion if criterion is not None else self.criterion
        comp_idx = compensator_surface_idx if compensator_surface_idx is not None else self.compensator_surface_idx
        comp_range = compensator_range_mm if compensator_range_mm is not None else self.compensator_range_mm

        if active_system is None:
            raise ValueError("No optical system provided for sensitivity analysis.")

        # Baseline evaluation
        nominal_merit = evaluate_tolerancing_criterion(active_system, criterion=active_crit, merit_function=active_merit)

        entries: List[SensitivityEntry] = []
        delta_wc_sum = 0.0
        delta_rss_sq_sum = 0.0

        for spec in active_tols:
            elem_idx = spec.surface_index if spec.surface_index is not None else spec.element_index
            target_nominal = _resolve_target_element(active_system, elem_idx)
            nom_val = get_element_parameter(target_nominal, spec.parameter_name)

            # Determine operand type
            p_upper = spec.parameter_name.upper().strip()
            op_type = p_upper if p_upper in ToleranceOperandType.__members__ else "TOL"

            # 1. Plus Delta Evaluation
            delta_plus = abs(spec.delta)
            sys_plus = clone_optical_system(active_system)
            target_plus = _resolve_target_element(sys_plus, elem_idx)
            perturb_element_parameter(target_plus, spec.parameter_name, delta_plus, system=sys_plus)
            if hasattr(sys_plus, "_update_accumulated_positions"):
                sys_plus._update_accumulated_positions()

            comp_adj_plus = 0.0
            if self.compensator_enabled:
                comp_adj_plus = solve_system_compensator(sys_plus, comp_idx, criterion=active_crit, merit_function=active_merit, delta_range=comp_range)

            crit_plus = evaluate_tolerancing_criterion(sys_plus, criterion=active_crit, merit_function=active_merit)
            delta_m_plus = crit_plus - nominal_merit

            # 2. Minus Delta Evaluation
            delta_minus = -abs(spec.delta)
            sys_minus = clone_optical_system(active_system)
            target_minus = _resolve_target_element(sys_minus, elem_idx)
            perturb_element_parameter(target_minus, spec.parameter_name, delta_minus, system=sys_minus)
            if hasattr(sys_minus, "_update_accumulated_positions"):
                sys_minus._update_accumulated_positions()

            comp_adj_minus = 0.0
            if self.compensator_enabled:
                comp_adj_minus = solve_system_compensator(sys_minus, comp_idx, criterion=active_crit, merit_function=active_merit, delta_range=comp_range)

            crit_minus = evaluate_tolerancing_criterion(sys_minus, criterion=active_crit, merit_function=active_merit)
            delta_m_minus = crit_minus - nominal_merit

            # Primary sensitivity metrics
            max_delta_deg = max(abs(delta_m_plus), abs(delta_m_minus))
            primary_delta = delta_m_plus if abs(delta_m_plus) >= abs(delta_m_minus) else delta_m_minus
            sensitivity = primary_delta / delta_plus if delta_plus > 1e-15 else 0.0

            delta_wc_sum += max_delta_deg
            delta_rss_sq_sum += max_delta_deg ** 2

            entries.append(
                SensitivityEntry(
                    parameter_name=spec.parameter_name,
                    element_index=elem_idx,
                    surface_index=elem_idx,
                    nominal_value=nom_val,
                    delta=delta_plus,
                    nominal_merit=nominal_merit,
                    perturbed_merit=nominal_merit + primary_delta,
                    delta_merit=primary_delta,
                    sensitivity=sensitivity,
                    operand_type=op_type,
                    delta_plus_merit=delta_m_plus,
                    delta_minus_merit=delta_m_minus,
                    compensator_adj_plus=comp_adj_plus,
                    compensator_adj_minus=comp_adj_minus,
                )
            )

        rss_delta = math.sqrt(delta_rss_sq_sum)

        crit_label = str(active_crit.value if hasattr(active_crit, "value") else active_crit).replace("_", " ").title()

        return SensitivityReport(
            entries=entries,
            nominal_merit=nominal_merit,
            criterion_name=crit_label,
            estimated_worst_case=nominal_merit + delta_wc_sum,
            estimated_rss=nominal_merit + rss_delta,
            estimated_worst_case_delta=delta_wc_sum,
            estimated_rss_delta=rss_delta,
        )

    def run_sensitivity_analysis(self, tolerances: Optional[Sequence[PerturbationSpec]] = None, **kwargs) -> SensitivityReport:
        """Alias for sensitivity_analysis using bound system and parameters."""
        return self.sensitivity_analysis(self.system, self.merit_function, tolerances=tolerances, **kwargs)

    def inverse_sensitivity_analysis(
        self,
        system: Optional[Any] = None,
        merit_function: Optional[Any] = None,
        tolerances: Optional[Sequence[PerturbationSpec]] = None,
        allowable_degradation: float = 0.005,
        allowable_degradation_um: Optional[float] = None,
        criterion: Optional[Union[TolerancingCriterion, str]] = None,
        compensator_surface_idx: Optional[int] = None,
        compensator_range_mm: Optional[float] = None,
        max_multiplier: float = 5.0,
        min_multiplier: float = 0.02,
    ) -> InverseSensitivityReport:
        """
        Execute Zemax-style Inverse Sensitivity Analysis:
        Given a target maximum allowable criterion degradation (e.g. +5.0 um RMS spot radius
        growth per operand), solves for the maximum allowable +Delta and -Delta tolerance
        limits on each parameter.
        """
        active_system = system if system is not None else self.system
        active_merit = merit_function if merit_function is not None else self.merit_function
        active_tols = list(tolerances) if tolerances is not None else []
        active_crit = criterion if criterion is not None else self.criterion
        comp_idx = compensator_surface_idx if compensator_surface_idx is not None else self.compensator_surface_idx
        comp_range = compensator_range_mm if compensator_range_mm is not None else self.compensator_range_mm

        if active_system is None:
            raise ValueError("No optical system provided for inverse sensitivity analysis.")

        crit_label = str(active_crit.value if hasattr(active_crit, "value") else active_crit).replace("_", " ").title()
        is_spot = any(k in crit_label.lower() for k in ("spot", "rms", "geo", "trac"))

        if allowable_degradation_um is not None:
            target_deg = float(allowable_degradation_um) / 1000.0 if is_spot else float(allowable_degradation_um)
        else:
            target_deg = max(1e-7, abs(float(allowable_degradation)))

        nominal_merit = evaluate_tolerancing_criterion(active_system, criterion=active_crit, merit_function=active_merit)
        entries: List[InverseSensitivityEntry] = []

        def _eval_perturbed(elem_idx: int, param_name: str, signed_delta: float) -> float:
            sys_clone = clone_optical_system(active_system)
            target_el = _resolve_target_element(sys_clone, elem_idx)
            perturb_element_parameter(target_el, param_name, signed_delta, system=sys_clone)
            if hasattr(sys_clone, "_update_accumulated_positions"):
                sys_clone._update_accumulated_positions()
            if self.compensator_enabled:
                solve_system_compensator(
                    sys_clone,
                    comp_idx,
                    criterion=active_crit,
                    merit_function=active_merit,
                    delta_range=comp_range,
                )
            val = evaluate_tolerancing_criterion(sys_clone, criterion=active_crit, merit_function=active_merit)
            return max(0.0, val - nominal_merit)

        def _solve_directional_limit(elem_idx: int, param_name: str, base_delta: float, sign: float) -> Tuple[float, float]:
            d0 = max(1e-7, abs(base_delta))
            d_min = d0 * min_multiplier
            d_max = d0 * max_multiplier

            deg0 = _eval_perturbed(elem_idx, param_name, sign * d0)
            if deg0 <= 1e-12:
                deg_max = _eval_perturbed(elem_idx, param_name, sign * d_max)
                return d_max, deg_max

            # First power-law / quadratic estimate
            ratio0 = target_deg / deg0
            d1 = float(np.clip(d0 * math.sqrt(ratio0), d_min, d_max))
            if abs(d1 - d0) / d0 < 0.05:
                return d0, deg0

            deg1 = _eval_perturbed(elem_idx, param_name, sign * d1)
            if deg1 <= 1e-12:
                return d1, deg1

            # Estimate local scaling exponent p in deg ~ d^p
            if abs(math.log(d1 / d0)) > 1e-6 and deg1 > 1e-12 and deg0 > 1e-12:
                p = math.log(deg1 / deg0) / math.log(d1 / d0)
                p = float(np.clip(p, 0.5, 2.5))
            else:
                p = 1.5

            d2 = float(np.clip(d1 * ((target_deg / deg1) ** (1.0 / p)), d_min, d_max))
            deg2 = _eval_perturbed(elem_idx, param_name, sign * d2)
            return d2, deg2

        for spec in active_tols:
            elem_idx = spec.surface_index if spec.surface_index is not None else spec.element_index
            target_nominal = _resolve_target_element(active_system, elem_idx)
            nom_val = get_element_parameter(target_nominal, spec.parameter_name)

            p_upper = spec.parameter_name.upper().strip()
            op_type = p_upper if p_upper in ToleranceOperandType.__members__ else "TOL"
            init_delta = max(1e-6, abs(spec.delta))

            solved_plus, deg_plus = _solve_directional_limit(elem_idx, spec.parameter_name, init_delta, +1.0)
            solved_minus_mag, deg_minus = _solve_directional_limit(elem_idx, spec.parameter_name, init_delta, -1.0)
            sym_delta = min(solved_plus, solved_minus_mag)

            entries.append(
                InverseSensitivityEntry(
                    parameter_name=spec.parameter_name,
                    element_index=elem_idx,
                    surface_index=elem_idx,
                    operand_type=op_type,
                    nominal_value=nom_val,
                    initial_delta=init_delta,
                    solved_plus_delta=solved_plus,
                    solved_minus_delta=-solved_minus_mag,
                    solved_symmetric_delta=sym_delta,
                    target_degradation=target_deg,
                    achieved_plus_degradation=deg_plus,
                    achieved_minus_degradation=deg_minus,
                )
            )

        return InverseSensitivityReport(
            entries=entries,
            nominal_merit=nominal_merit,
            target_degradation=target_deg,
            criterion_name=crit_label,
        )

    def run_inverse_sensitivity(
        self,
        tolerances: Optional[Sequence[PerturbationSpec]] = None,
        allowable_degradation: float = 0.005,
        allowable_degradation_um: Optional[float] = None,
        **kwargs: Any,
    ) -> InverseSensitivityReport:
        """Alias for inverse_sensitivity_analysis using bound system and parameters."""
        return self.inverse_sensitivity_analysis(
            self.system,
            self.merit_function,
            tolerances=tolerances,
            allowable_degradation=allowable_degradation,
            allowable_degradation_um=allowable_degradation_um,
            **kwargs,
        )

    def monte_carlo_simulation(
        self,
        system: Optional[Any] = None,
        merit_function: Optional[Any] = None,
        tolerances: Optional[Sequence[PerturbationSpec]] = None,
        num_trials: int = 1000,
        trials: Optional[int] = None,
        seed: Optional[int] = None,
        random_seed: Optional[int] = None,
        criterion: Optional[Union[TolerancingCriterion, str]] = None,
        compensator_surface_idx: Optional[int] = None,
        compensator_range_mm: Optional[float] = None,
        **kwargs: Any,
    ) -> MonteCarloReport:
        """
        Execute multivariate Monte Carlo simulation across combined tolerance distributions.
        Supports Uniform, Gaussian/Normal (3-sigma), and Parabolic distributions.
        """
        active_system = system if system is not None else self.system
        active_merit = merit_function if merit_function is not None else self.merit_function
        active_tols = list(tolerances) if tolerances is not None else []
        active_crit = criterion if criterion is not None else self.criterion
        comp_idx = compensator_surface_idx if compensator_surface_idx is not None else self.compensator_surface_idx
        comp_range = compensator_range_mm if compensator_range_mm is not None else self.compensator_range_mm
        total_trials = trials if trials is not None else num_trials

        if active_system is None:
            raise ValueError("No optical system provided for Monte Carlo simulation.")

        chosen_seed = seed if seed is not None else (random_seed if random_seed is not None else self.random_seed)
        rng = np.random.default_rng(chosen_seed)

        nominal_merit = evaluate_tolerancing_criterion(active_system, criterion=active_crit, merit_function=active_merit)

        trial_records: List[MonteCarloTrialRecord] = []

        for trial_idx in range(total_trials):
            trial_deltas: Dict[str, float] = {}
            cloned_sys = clone_optical_system(active_system)

            # Sample and apply combined perturbations
            for spec in active_tols:
                elem_idx = spec.surface_index if spec.surface_index is not None else spec.element_index
                target_cloned = _resolve_target_element(cloned_sys, elem_idx)

                delta_sample = sample_tolerance_delta(
                    delta=spec.delta,
                    distribution=spec.distribution,
                    rng=rng,
                    min_delta=spec.min_delta,
                    max_delta=spec.max_delta,
                )

                param_key = f"surf{elem_idx}_{spec.parameter_name}"
                trial_deltas[param_key] = delta_sample

                perturb_element_parameter(target_cloned, spec.parameter_name, delta_sample, system=cloned_sys)

            if hasattr(cloned_sys, "_update_accumulated_positions"):
                cloned_sys._update_accumulated_positions()

            # Compensator optimization (auto-refocus)
            comp_shift = 0.0
            if self.compensator_enabled:
                comp_shift = solve_system_compensator(cloned_sys, comp_idx, criterion=active_crit, merit_function=active_merit, delta_range=comp_range)

            # Performance evaluation
            trial_merit = evaluate_tolerancing_criterion(cloned_sys, criterion=active_crit, merit_function=active_merit)
            trial_efl, trial_bfl = extract_system_cardinal_metrics(cloned_sys)

            # Spot sizes
            rms_spot = compute_system_spot_radius(cloned_sys, geometric=False)
            geo_spot = compute_system_spot_radius(cloned_sys, geometric=True)

            trial_records.append(
                MonteCarloTrialRecord(
                    trial_index=trial_idx,
                    perturbations=trial_deltas,
                    merit=trial_merit,
                    efl=trial_efl,
                    bfl=trial_bfl,
                    compensator_adjustment=comp_shift,
                    spot_radius_rms=rms_spot,
                    spot_radius_geo=geo_spot,
                )
            )

        # Statistical compilation
        merits = np.array([t.merit for t in trial_records], dtype=float) if trial_records else np.array([nominal_merit])
        efls = np.array([t.efl for t in trial_records], dtype=float) if trial_records else np.array([0.0])
        bfls = np.array([t.bfl for t in trial_records], dtype=float) if trial_records else np.array([0.0])
        comps = np.array([t.compensator_adjustment for t in trial_records], dtype=float) if trial_records else np.array([0.0])

        best_idx = int(np.argmin(merits)) if len(merits) > 0 else 0
        worst_idx = int(np.argmax(merits)) if len(merits) > 0 else 0

        # Percentiles (strictly non-decreasing)
        p_keys = [50, 80, 90, 95, 98, 99]
        percentile_dict = {p: float(np.percentile(merits, p)) for p in p_keys}

        crit_label = str(active_crit.value if hasattr(active_crit, "value") else active_crit).replace("_", " ").title()

        return MonteCarloReport(
            trials=trial_records,
            num_trials=len(trial_records),
            nominal_merit=nominal_merit,
            mean_merit=float(np.mean(merits)),
            std_merit=float(np.std(merits)),
            min_merit=float(np.min(merits)),
            max_merit=float(np.max(merits)),
            percentiles=percentile_dict,
            mean_efl=float(np.mean(efls)),
            std_efl=float(np.std(efls)),
            mean_bfl=float(np.mean(bfls)),
            std_bfl=float(np.std(bfls)),
            mean_compensator=float(np.mean(comps)),
            std_compensator=float(np.std(comps)),
            criterion_name=crit_label,
            best_trial_index=best_idx,
            worst_trial_index=worst_idx,
        )

    def run_monte_carlo(self, tolerances: Optional[Sequence[PerturbationSpec]] = None, num_trials: int = 1000, **kwargs: Any) -> MonteCarloReport:
        """Alias for monte_carlo_simulation."""
        return self.monte_carlo_simulation(self.system, self.merit_function, tolerances=tolerances, num_trials=num_trials, **kwargs)

    def run_full_analysis(
        self,
        tolerances: Optional[Sequence[PerturbationSpec]] = None,
        num_trials: int = 500,
        **kwargs
    ) -> Tuple[SensitivityReport, MonteCarloReport]:
        """Run complete Zemax tolerancing pipeline: Sensitivity Analysis + Monte Carlo Simulation."""
        sens = self.sensitivity_analysis(tolerances=tolerances, **kwargs)
        mc = self.monte_carlo_simulation(tolerances=tolerances, num_trials=num_trials, **kwargs)
        return sens, mc


# Alias for Zemax Tolerancing Engine
ZemaxTolerancingEngine = TolerancingEngine


# =============================================================================
# 9. STANDARD COMMERCIAL & PRECISION TOLERANCE PRESETS
# =============================================================================

def create_commercial_tolerances(system: Any) -> List[PerturbationSpec]:
    """
    Generate standard commercial grade tolerances (Thorlabs / Edmund standard optics).
    - Radius: +/- 0.2%
    - Thickness: +/- 0.05 mm
    - Surface Decenter: +/- 0.025 mm
    - Surface Tilt: +/- 1.0 mrad (+/- 0.057 deg)
    - Index: +/- 0.001
    - Abbe: +/- 1.0%
    """
    tols: List[PerturbationSpec] = []
    surfaces = getattr(system, "surfaces", [])
    if not surfaces and hasattr(system, "components"):
        surfaces = system.components

    for i, s in enumerate(surfaces):
        stype = getattr(s, "surface_type", None)
        stype_str = str(stype.name) if hasattr(stype, "name") else str(stype)
        if stype_str in ("OBJECT", "COORDINATE_BREAK", "DETECTOR"):
            continue

        r = float(getattr(s, "radius", 0.0))
        t = float(getattr(s, "thickness", 0.0))
        glass = str(getattr(s, "glass", "AIR")).upper()

        # Radius tolerance
        if abs(r) > 1e-6:
            r_delta = abs(r) * 0.002  # 0.2%
            tols.append(PerturbationSpec(parameter_name="trad", element_index=i, surface_index=i, delta=r_delta, distribution="gaussian", notes=f"Surf {i} Radius 0.2%"))

        # Thickness tolerance
        if t > 0.0 and i < len(surfaces) - 1:
            tols.append(PerturbationSpec(parameter_name="tthi", element_index=i, surface_index=i, delta=0.05, distribution="uniform", notes=f"Surf {i} Thickness 0.05mm"))

        # Decenter and tilt
        tols.append(PerturbationSpec(parameter_name="tsdy", element_index=i, surface_index=i, delta=0.025, distribution="uniform", notes=f"Surf {i} Decenter 0.025mm"))
        tols.append(PerturbationSpec(parameter_name="tstx", element_index=i, surface_index=i, delta=0.057, distribution="uniform", notes=f"Surf {i} Tilt 1.0mrad"))

        # Refractive index for glass
        if glass not in ("AIR", "MIRROR", ""):
            tols.append(PerturbationSpec(parameter_name="tind", element_index=i, surface_index=i, delta=0.001, distribution="uniform", notes=f"Surf {i} Index 0.001"))

    return tols


def create_precision_tolerances(system: Any) -> List[PerturbationSpec]:
    """
    Generate precision grade optical tolerances (Research / High-End Imaging).
    - Radius: +/- 0.1%
    - Thickness: +/- 0.02 mm
    - Surface Decenter: +/- 0.010 mm
    - Surface Tilt: +/- 0.5 mrad (+/- 0.028 deg)
    - Index: +/- 0.0005
    """
    tols: List[PerturbationSpec] = []
    surfaces = getattr(system, "surfaces", [])
    if not surfaces and hasattr(system, "components"):
        surfaces = system.components

    for i, s in enumerate(surfaces):
        stype = getattr(s, "surface_type", None)
        stype_str = str(stype.name) if hasattr(stype, "name") else str(stype)
        if stype_str in ("OBJECT", "COORDINATE_BREAK", "DETECTOR"):
            continue

        r = float(getattr(s, "radius", 0.0))
        t = float(getattr(s, "thickness", 0.0))
        glass = str(getattr(s, "glass", "AIR")).upper()

        if abs(r) > 1e-6:
            r_delta = abs(r) * 0.001  # 0.1%
            tols.append(PerturbationSpec(parameter_name="trad", element_index=i, surface_index=i, delta=r_delta, distribution="gaussian", notes=f"Surf {i} Radius 0.1%"))

        if t > 0.0 and i < len(surfaces) - 1:
            tols.append(PerturbationSpec(parameter_name="tthi", element_index=i, surface_index=i, delta=0.02, distribution="uniform", notes=f"Surf {i} Thickness 0.02mm"))

        tols.append(PerturbationSpec(parameter_name="tsdy", element_index=i, surface_index=i, delta=0.010, distribution="uniform", notes=f"Surf {i} Decenter 0.010mm"))
        tols.append(PerturbationSpec(parameter_name="tstx", element_index=i, surface_index=i, delta=0.028, distribution="uniform", notes=f"Surf {i} Tilt 0.5mrad"))

        if glass not in ("AIR", "MIRROR", ""):
            tols.append(PerturbationSpec(parameter_name="tind", element_index=i, surface_index=i, delta=0.0005, distribution="uniform", notes=f"Surf {i} Index 0.0005"))

    return tols


def create_high_precision_tolerances(system: Any) -> List[PerturbationSpec]:
    """
    Generate high-precision / laser grade optical tolerances.
    - Radius: +/- 0.05%
    - Thickness: +/- 0.01 mm
    - Surface Decenter: +/- 0.005 mm
    - Surface Tilt: +/- 0.2 mrad (+/- 0.011 deg)
    - Index: +/- 0.0002
    """
    tols: List[PerturbationSpec] = []
    surfaces = getattr(system, "surfaces", [])
    if not surfaces and hasattr(system, "components"):
        surfaces = system.components

    for i, s in enumerate(surfaces):
        stype = getattr(s, "surface_type", None)
        stype_str = str(stype.name) if hasattr(stype, "name") else str(stype)
        if stype_str in ("OBJECT", "COORDINATE_BREAK", "DETECTOR"):
            continue

        r = float(getattr(s, "radius", 0.0))
        t = float(getattr(s, "thickness", 0.0))
        glass = str(getattr(s, "glass", "AIR")).upper()

        if abs(r) > 1e-6:
            r_delta = abs(r) * 0.0005  # 0.05%
            tols.append(PerturbationSpec(parameter_name="trad", element_index=i, surface_index=i, delta=r_delta, distribution="gaussian", notes=f"Surf {i} Radius 0.05%"))

        if t > 0.0 and i < len(surfaces) - 1:
            tols.append(PerturbationSpec(parameter_name="tthi", element_index=i, surface_index=i, delta=0.01, distribution="uniform", notes=f"Surf {i} Thickness 0.01mm"))

        tols.append(PerturbationSpec(parameter_name="tsdy", element_index=i, surface_index=i, delta=0.005, distribution="uniform", notes=f"Surf {i} Decenter 0.005mm"))
        tols.append(PerturbationSpec(parameter_name="tstx", element_index=i, surface_index=i, delta=0.011, distribution="uniform", notes=f"Surf {i} Tilt 0.2mrad"))

        if glass not in ("AIR", "MIRROR", ""):
            tols.append(PerturbationSpec(parameter_name="tind", element_index=i, surface_index=i, delta=0.0002, distribution="uniform", notes=f"Surf {i} Index 0.0002"))

    return tols


# =============================================================================
# 10. ZEMAX OPTICSTUDIO ASCII REPORT & FILE EXPORTS
# =============================================================================

def export_zemax_tolerance_report(
    sensitivity_report: Optional[SensitivityReport] = None,
    monte_carlo_report: Optional[MonteCarloReport] = None,
    inverse_sensitivity_report: Optional[InverseSensitivityReport] = None,
    system_title: str = "2D Optics Optical Prescription",
    filepath: Optional[str] = None,
) -> str:
    """
    Export full Zemax OpticStudio formatted ASCII tolerance report.
    """
    sections = [
        "=" * 105,
        "ZEMAX OPTICSTUDIO TOLERANCE ANALYSIS REPORT",
        f"Design File: {system_title}",
        f"Analysis Engine: 2D Optics Zemax Tolerancing Suite",
        "=" * 105,
        "",
    ]

    if sensitivity_report is not None:
        sections.append(sensitivity_report.format_table())
        sections.append("")

    if inverse_sensitivity_report is not None:
        sections.append(inverse_sensitivity_report.format_table())
        sections.append("")

    if monte_carlo_report is not None:
        sections.append(monte_carlo_report.format_report())
        sections.append("")

    report_text = "\n".join(sections)

    if filepath is not None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)

    return report_text


def export_zmx_with_tolerances(
    stack: SurfaceStack,
    tolerances: Sequence[PerturbationSpec],
    filepath: str,
    title: str = "Toleranced System Export",
) -> str:
    """
    Export a SurfaceStack as a Zemax OpticStudio (.zmx) prescription containing standard TOL operands.
    """
    try:
        from optics_app.io.zmx_parser import export_to_zmx_string
    except ImportError:
        from v2.io.zmx_parser import export_to_zmx_string

    zmx_base = export_to_zmx_string(stack, title=title)
    tol_lines = ["", "! --- ZEMAX TOLERANCE DATA OPERANDS ---"]

    for tol in tolerances:
        op = tol.parameter_name.upper().strip()
        surf = tol.surface_index if tol.surface_index is not None else tol.element_index
        delta = abs(tol.delta)
        dist_code = 0 if tol.distribution == "uniform" else (1 if tol.distribution in ("normal", "gaussian") else 2)
        tol_lines.append(f"TOL {op} {surf} {tol.delta} {dist_code} ! {tol.notes}")

    full_content = zmx_base + "\n" + "\n".join(tol_lines) + "\n"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_content)

    return full_content


def parse_zmx_tolerances(zmx_content_or_file: str) -> List[PerturbationSpec]:
    """Parse TOL lines from Zemax .zmx text or file path into PerturbationSpec objects."""
    import os
    content = zmx_content_or_file
    if os.path.exists(zmx_content_or_file):
        with open(zmx_content_or_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    tolerances = []
    dist_map = {0: "uniform", 1: "gaussian", 2: "parabolic"}

    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("TOL"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            op = parts[1].upper()
            try:
                surf = int(parts[2])
                delta = float(parts[3])
                dist_code = int(parts[4]) if len(parts) > 4 else 0
                dist_str = dist_map.get(dist_code, "uniform")
                notes = line.split("!")[1].strip() if "!" in line else f"Zemax {op} Surf {surf}"
                tolerances.append(PerturbationSpec(parameter_name=op, surface_index=surf, element_index=surf, delta=delta, distribution=dist_str, notes=notes))
            except ValueError:
                continue

    return tolerances


__all__ = [
    # Backward-compatible symbols
    "PerturbationSpec",
    "SensitivityEntry",
    "SensitivityReport",
    "InverseSensitivityEntry",
    "InverseSensitivityReport",
    "MonteCarloTrialRecord",
    "MonteCarloReport",
    "TolerancingEngine",
    "decenter_matrix_3x3",
    "tilt_matrix_3x3",
    "augmented_transfer_matrix_3x3",
    "clone_optical_system",
    "get_element_parameter",
    "perturb_element_parameter",
    # Zemax Tolerancing Suite symbols
    "ToleranceOperandType",
    "ToleranceDistribution",
    "TolerancingCriterion",
    "ZemaxToleranceOperand",
    "ZemaxSensitivityEntry",
    "ZemaxSensitivityReport",
    "ZemaxInverseSensitivityEntry",
    "ZemaxInverseSensitivityReport",
    "ZemaxMonteCarloTrial",
    "ZemaxMonteCarloReport",
    "ZemaxTolerancingEngine",
    "sample_tolerance_delta",
    "compute_system_spot_radius",
    "evaluate_tolerancing_criterion",
    "solve_system_compensator",
    "create_commercial_tolerances",
    "create_precision_tolerances",
    "create_high_precision_tolerances",
    "export_zemax_tolerance_report",
    "export_zmx_with_tolerances",
    "parse_zmx_tolerances",
]
