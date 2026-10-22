"""
Physical Thick Lens & Paraxial Optical Element Backends.

Implements the OpticalElement abstract base class, encapsulated ThickLens model
with glass catalog integration and analytical Gullstrand power checks, ThinLens paraxial model,
and AchromaticDoublet extensions. Fully ready for Zemax-style tolerancing and transparent
manual stepping.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

from optics_app.engine.matrix_optics import (
    TransferMatrix,
    CardinalPoints,
    CalculationStep,
    CalculationTrace,
    refraction_matrix,
    space_matrix,
    thin_lens_matrix,
    thick_lens_matrix,
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.utils.materials import get_refractive_index


class OpticalElement(ABC):
    """
    Abstract Base Class for modular paraxial and physical optical components.

    Enables extensible lens design, multi-component optical assemblies,
    first-order optimization, and Zemax-style optical tolerancing.
    """

    @abstractmethod
    def transfer_matrix(self, wavelength_nm: float = 587.6) -> TransferMatrix:
        """
        Compute the composite 2x2/3x3 ray transfer matrix through this element.

        Args:
            wavelength_nm: Optical wavelength in nanometers (default 587.6 nm, Fraunhofer d-line).
        """
        pass

    @abstractmethod
    def cardinal_points(self, wavelength_nm: float = 587.6) -> CardinalPoints:
        """
        Extract paraxial cardinal points (EFL, BFL, FFL, P1, P2, N1, N2).

        Args:
            wavelength_nm: Optical wavelength in nanometers.
        """
        pass

    def to_surfaces(self) -> List[Surface]:
        """
        Bridge to sequential 3D Surface stack for non-sequential ray tracing and LDE display.
        Default returns an empty list for elements that do not model explicit physical surfaces.
        """
        return []

    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """
        Expose all physical parameters (radii, thickness, index, tilt, decenter) as a dictionary.
        """
        pass

    @abstractmethod
    def set_parameters(self, params: Dict[str, float]) -> None:
        """
        Update physical parameters from a dictionary.
        """
        pass

    def perturb(self, param_name: str, delta: float) -> None:
        """
        Perturb a named parameter by delta for sensitivity analysis and Monte Carlo tolerancing.
        Default implementation updates parameter via set_parameters if supported.
        """
        params = self.get_parameters()
        key_lower = param_name.lower().strip()
        for k, v in params.items():
            if k.lower().strip() == key_lower:
                self.set_parameters({k: v + delta})
                return
        raise KeyError(
            f"Parameter '{param_name}' not recognized in {self.__class__.__name__}. "
            f"Available: {list(params.keys())}"
        )

    def get_parameter(self, param_name: str) -> float:
        """Query an individual parameter value."""
        params = self.get_parameters()
        key_lower = param_name.lower().strip()
        for k, v in params.items():
            if k.lower().strip() == key_lower:
                return v
        raise KeyError(
            f"Parameter '{param_name}' not recognized in {self.__class__.__name__}. "
            f"Available: {list(params.keys())}"
        )

    def set_parameter(self, param_name: str, value: float) -> None:
        """Set an individual parameter value."""
        self.set_parameters({param_name: value})

    def clone(self) -> Any:
        """Create an independent copy of this optical element."""
        import copy
        return copy.deepcopy(self)

    def augmented_matrix_3x3(
        self,
        decenter_y: Optional[float] = None,
        tilt_theta: Optional[float] = None,
    ) -> np.ndarray:
        """
        Construct 3x3 augmented transfer matrix for decenter and tilt tolerancing.
        Defaults decenter_y to self.decenter and tilt_theta to self.tilt if not explicitly passed.
        """
        mat = self.transfer_matrix()
        A = float(mat.A)
        B = float(mat.B)
        C = float(mat.C)
        D = float(mat.D)

        dy = float(decenter_y) if decenter_y is not None else float(getattr(self, "decenter", 0.0))
        theta = float(tilt_theta) if tilt_theta is not None else float(getattr(self, "tilt", 0.0))

        E = (1.0 - A) * dy - B * theta
        F = -C * dy + (1.0 - D) * theta
        return np.array([
            [A, B, E],
            [C, D, F],
            [0.0, 0.0, 1.0]
        ], dtype=float)

    def step_by_step_calculation(self, wavelength_nm: float = 587.6) -> CalculationTrace:
        """
        Return a comprehensive step-by-step calculation trace for manual debugging.
        """
        mat = self.transfer_matrix(wavelength_nm=wavelength_nm)
        if mat.trace is not None:
            return mat.trace
        trace = CalculationTrace(f"Calculation for {self.__class__.__name__}")
        trace.add_step(
            CalculationStep(
                name="Transfer Matrix Evaluation",
                formula="M = transfer_matrix(wavelength_nm)",
                inputs={"wavelength_nm": wavelength_nm},
                matrix=mat,
            )
        )
        return trace

    @property
    def efl(self) -> float:
        """Effective Focal Length (mm) at nominal wavelength (587.6 nm)."""
        return self.cardinal_points().efl

    @property
    def bfl(self) -> float:
        """Back Focal Length (mm) at nominal wavelength (587.6 nm)."""
        return self.cardinal_points().bfl

    @property
    def ffl(self) -> float:
        """Front Focal Length (mm) at nominal wavelength (587.6 nm)."""
        return self.cardinal_points().ffl

    @property
    def p1(self) -> float:
        """Front Principal Plane location (mm from V1) at nominal wavelength (587.6 nm)."""
        return self.cardinal_points().p1

    @property
    def p2(self) -> float:
        """Rear Principal Plane location (mm from V2) at nominal wavelength (587.6 nm)."""
        return self.cardinal_points().p2

    @property
    def focal_length(self) -> float:
        """Alias for EFL."""
        return self.efl

    @focal_length.setter
    def focal_length(self, value: float) -> None:
        """Allow setting focal length directly on elements that support it."""
        if hasattr(self, "_focal_length"):
            self._focal_length = float(value)
        else:
            self.set_parameters({"focal_length": float(value)})

    @property
    def index(self) -> float:
        """Refractive index at nominal wavelength (587.6 nm)."""
        if hasattr(self, "_index"):
            return float(self._index)
        if hasattr(self, "refractive_index") and callable(self.refractive_index):
            return float(self.refractive_index())
        return float(getattr(self, "glass_or_index", 1.0))

    @index.setter
    def index(self, value: float) -> None:
        """Set refractive index or glass material."""
        self._index = float(value)
        if hasattr(self, "glass_or_index"):
            self.glass_or_index = float(value)




class ThickLens(OpticalElement):
    """
    Physical Thick Lens Optical Component.

    Constructed with front radius R1, rear radius R2, center thickness t,
    material glass name or constant refractive index, and diameter.
    Supports Gullstrand power checks, transparent diagnostic stepping,
    and native tolerancing (R1, R2, t, n, decenter, tilt).
    """

    def __init__(
        self,
        r1: float,
        r2: float,
        thickness: float,
        glass_or_index: Union[str, float],
        diameter: float = 25.4,
        decenter: float = 0.0,
        tilt: float = 0.0,
        name: str = "ThickLens",
    ):
        """
        Initialize ThickLens.

        Args:
            r1: Front surface radius of curvature (mm). Positive = convex to light from left.
            r2: Rear surface radius of curvature (mm). Negative = convex to light from left.
            thickness: Center thickness along optical axis (mm).
            glass_or_index: Glass name (e.g. 'N-BK7') or numeric refractive index (e.g. 1.5).
            diameter: Clear aperture outer diameter (mm).
            decenter: Transverse decenter Delta y (mm).
            tilt: Paraxial tilt angle around x-axis (degrees).
            name: Label identifier.
        """
        self.r1 = float(r1)
        self.r2 = float(r2)

        # Parameter physical validation
        if float(thickness) <= 0.0:
            raise ValueError(f"Center thickness must be strictly positive (> 0), got {thickness}")
        self.thickness = float(thickness)

        if float(diameter) <= 0.0:
            raise ValueError(f"Lens diameter must be strictly positive (> 0), got {diameter}")
        self.diameter = float(diameter)

        if isinstance(glass_or_index, (int, float)) and float(glass_or_index) <= 0.0:
            raise ValueError(f"Refractive index must be strictly positive (> 0), got {glass_or_index}")
        if isinstance(glass_or_index, str):
            try:
                num_val = float(glass_or_index)
                if num_val <= 0.0:
                    raise ValueError(f"Refractive index must be strictly positive (> 0), got {num_val}")
            except ValueError as e:
                if "strictly positive" in str(e):
                    raise
        self.glass_or_index = glass_or_index

        self.decenter = float(decenter)
        self.tilt = float(tilt)
        self.name = name

    @property
    def glass_name(self) -> str:
        """Return glass catalog identifier string."""
        if isinstance(self.glass_or_index, str):
            return self.glass_or_index
        return f"INDEX_{float(self.glass_or_index):.4f}"

    def refractive_index(self, wavelength_nm: float = 587.6) -> float:
        """
        Evaluate refractive index at the specified wavelength (nm).
        """
        if isinstance(self.glass_or_index, (int, float)):
            return float(self.glass_or_index)
        if isinstance(self.glass_or_index, str):
            try:
                return float(self.glass_or_index)
            except ValueError:
                pass
            wavelength_um = wavelength_nm / 1000.0
            return float(get_refractive_index(self.glass_or_index, wavelength_um))
        return 1.5

    def analytical_power(self, wavelength_nm: float = 587.6) -> float:
        """
        Compute analytical optical power Phi via Gullstrand's thick lens equation:
            Phi = Phi1 + Phi2 - (t / n) * Phi1 * Phi2
        where:
            Phi1 = (n - 1) / R1 (or 0 if flat)
            Phi2 = (1 - n) / R2 (or 0 if flat)
        """
        n = self.refractive_index(wavelength_nm)
        is_r1_flat = (self.r1 == 0.0 or math.isinf(self.r1) or abs(self.r1) > 1e12)
        is_r2_flat = (self.r2 == 0.0 or math.isinf(self.r2) or abs(self.r2) > 1e12)

        phi1 = (n - 1.0) / self.r1 if not is_r1_flat else 0.0
        phi2 = (1.0 - n) / self.r2 if not is_r2_flat else 0.0

        phi_total = phi1 + phi2 - (self.thickness / n) * phi1 * phi2
        return float(phi_total)

    def analytical_efl(self, wavelength_nm: float = 587.6) -> float:
        """
        Compute analytical Effective Focal Length (EFL) via Gullstrand's equation:
            EFL = 1 / Phi
        """
        phi = self.analytical_power(wavelength_nm)
        if abs(phi) < 1e-12:
            return float("inf")
        return 1.0 / phi

    def transfer_matrix(self, wavelength_nm: float = 587.6) -> TransferMatrix:
        """
        Compute composite ABCD transfer matrix through the thick lens,
        applying 3x3 augmented perturbations for decenter and tilt if non-zero.
        """
        n = self.refractive_index(wavelength_nm)
        mat = thick_lens_matrix(
            R1=self.r1,
            R2=self.r2,
            thickness=self.thickness,
            n_lens=n,
            n_ambient=1.0,
        )
        mat.label = self.name

        # Apply decenter and tilt if specified
        if abs(self.decenter) > 1e-12:
            mat = mat.with_decenter(self.decenter)
        if abs(self.tilt) > 1e-12:
            mat = mat.with_tilt(self.tilt, in_degrees=True)

        return mat

    def cardinal_points(self, wavelength_nm: float = 587.6) -> CardinalPoints:
        """Extract paraxial cardinal points (EFL, BFL, FFL, P1, P2, N1, N2)."""
        return self.transfer_matrix(wavelength_nm).cardinal_points()

    def clone(self) -> "ThickLens":
        """Return an independent copy of this lens."""
        return ThickLens(
            r1=self.r1,
            r2=self.r2,
            thickness=self.thickness,
            glass_or_index=self.glass_or_index,
            diameter=self.diameter,
            decenter=self.decenter,
            tilt=self.tilt,
            name=f"{self.name}_Clone",
        )

    @property
    def efl(self) -> float:
        """Effective Focal Length at 587.6 nm."""
        return self.cardinal_points().efl

    @property
    def bfl(self) -> float:
        """Back Focal Length at 587.6 nm."""
        return self.cardinal_points().bfl

    @property
    def ffl(self) -> float:
        """Front Focal Length at 587.6 nm."""
        return self.cardinal_points().ffl

    @property
    def p1(self) -> float:
        """Front Principal Plane location from front vertex at 587.6 nm."""
        return self.cardinal_points().p1

    @property
    def p2(self) -> float:
        """Rear Principal Plane location from rear vertex at 587.6 nm."""
        return self.cardinal_points().p2

    @property
    def focal_length(self) -> float:
        """Alias for EFL."""
        return self.efl

    @property
    def index(self) -> float:
        """Refractive index at nominal wavelength (587.6 nm)."""
        return float(self.refractive_index())

    @index.setter
    def index(self, value: float) -> None:
        """Set refractive index directly."""
        self.glass_or_index = float(value)

    def step_by_step_calculation(self, wavelength_nm: float = 587.6) -> CalculationTrace:
        """
        Perform a transparent step-by-step derivation with Gullstrand verification.
        """
        n = self.refractive_index(wavelength_nm)
        trace = CalculationTrace(f"Thick Lens Analysis: {self.name}")

        # Matrix tracing with step recording
        mat = thick_lens_matrix(
            R1=self.r1,
            R2=self.r2,
            thickness=self.thickness,
            n_lens=n,
            n_ambient=1.0,
            trace=trace,
        )

        # Analytical Gullstrand comparison step
        gullstrand_power = self.analytical_power(wavelength_nm)
        gullstrand_efl = self.analytical_efl(wavelength_nm)
        matrix_power = -mat.C
        matrix_efl = mat.effective_focal_length()
        discrepancy = abs(matrix_efl - gullstrand_efl)

        verification_step = CalculationStep(
            name="Analytical Gullstrand Power Verification",
            formula="Phi = Phi1 + Phi2 - (t / n) * Phi1 * Phi2",
            inputs={
                "R1": self.r1,
                "R2": self.r2,
                "thickness": self.thickness,
                "index": n,
            },
            intermediate_values={
                "Gullstrand_Power": gullstrand_power,
                "Gullstrand_EFL": gullstrand_efl,
                "Matrix_Power": matrix_power,
                "Matrix_EFL": matrix_efl,
                "Absolute_Difference": discrepancy,
            },
            matrix=mat,
            notes=f"Cross-check: Gullstrand EFL vs Matrix EFL agree within {discrepancy:.2e} mm.",
        )
        trace.add_step(verification_step)
        mat.trace = trace
        return trace

    def to_surfaces(self) -> List[Surface]:
        """
        Convert to a list of two Surface objects bridging to 3D ray tracing.
        """
        semi_diam = self.diameter / 2.0
        s1 = Surface(
            name=f"{self.name}_Front",
            surface_type=SurfaceType.STANDARD,
            radius=self.r1,
            thickness=self.thickness,
            glass=self.glass_name,
            semi_diameter=semi_diam,
            decenter_y=self.decenter,
            tilt_x=self.tilt,
        )
        s2 = Surface(
            name=f"{self.name}_Rear",
            surface_type=SurfaceType.STANDARD,
            radius=self.r2,
            thickness=0.0,
            glass="AIR",
            semi_diameter=semi_diam,
            decenter_y=self.decenter,
            tilt_x=self.tilt,
        )
        return [s1, s2]

    def get_parameters(self) -> Dict[str, float]:
        """
        Return dictionary of all perturbable parameters for optimization and tolerancing.
        """
        return {
            "r1": self.r1,
            "r2": self.r2,
            "thickness": self.thickness,
            "index": self.refractive_index(),
            "diameter": self.diameter,
            "decenter": self.decenter,
            "tilt": self.tilt,
        }

    def set_parameters(self, params: Dict[str, float]) -> None:
        """
        Update physical parameters from dictionary.
        """
        param_map = {k.lower().strip(): v for k, v in params.items()}

        if "r1" in param_map:
            self.r1 = float(param_map["r1"])
        if "r2" in param_map:
            self.r2 = float(param_map["r2"])
        if "thickness" in param_map:
            self.thickness = float(param_map["thickness"])
        elif "t" in param_map:
            self.thickness = float(param_map["t"])

        if "index" in param_map:
            self.glass_or_index = float(param_map["index"])
        elif "n" in param_map:
            self.glass_or_index = float(param_map["n"])

        if "diameter" in param_map:
            self.diameter = float(param_map["diameter"])
        elif "diam" in param_map or "d" in param_map:
            self.diameter = float(param_map.get("diam", param_map.get("d")))

        if "decenter" in param_map:
            self.decenter = float(param_map["decenter"])
        elif "dy" in param_map:
            self.decenter = float(param_map["dy"])

        if "tilt" in param_map:
            self.tilt = float(param_map["tilt"])
        elif "theta" in param_map:
            self.tilt = float(param_map["theta"])

    def perturb(self, param_name: str, delta: float) -> None:
        """
        Perturb a parameter by delta.
        """
        key = param_name.lower().strip()
        if key in ("r1",):
            self.r1 += delta
        elif key in ("r2",):
            self.r2 += delta
        elif key in ("thickness", "t"):
            self.thickness += delta
        elif key in ("index", "n"):
            current_n = self.refractive_index()
            self.glass_or_index = current_n + delta
        elif key in ("diameter", "diam", "d"):
            self.diameter += delta
        elif key in ("decenter", "dy"):
            self.decenter += delta
        elif key in ("tilt", "theta"):
            self.tilt += delta
        else:
            raise KeyError(
                f"Unknown parameter '{param_name}' to perturb. "
                f"Supported: r1, r2, thickness, index, diameter, decenter, tilt."
            )

    def __repr__(self) -> str:
        return (
            f"ThickLens({self.name}, R1={self.r1}, R2={self.r2}, "
            f"t={self.thickness}, glass={self.glass_name}, D={self.diameter})"
        )


class ThinLens(OpticalElement):
    """
    Paraxial Thin Lens Component.

    Characterized by an effective focal length f and clear diameter.
    Infinitesimal thickness (t = 0), principal planes at the lens plane (P1 = P2 = 0).
    """

    def __init__(
        self,
        focal_length: float,
        diameter: float = 25.4,
        decenter: float = 0.0,
        tilt: float = 0.0,
        name: str = "ThinLens",
    ):
        """
        Initialize ThinLens.

        Args:
            focal_length: Effective focal length (mm). Positive = converging, negative = diverging.
            diameter: Clear aperture diameter (mm).
            decenter: Transverse decenter Delta y (mm).
            tilt: Paraxial tilt angle (degrees).
            name: Element label.
        """
        if focal_length == 0.0:
            raise ValueError("Thin lens focal length cannot be zero.")
        self._focal_length = float(focal_length)
        self.diameter = float(diameter)
        self.decenter = float(decenter)
        self.tilt = float(tilt)
        self.name = name

    @property
    def focal_length(self) -> float:
        """Effective focal length (mm)."""
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value: float) -> None:
        if value == 0.0:
            raise ValueError("Thin lens focal length cannot be zero.")
        self._focal_length = float(value)


    def transfer_matrix(self, wavelength_nm: float = 587.6) -> TransferMatrix:
        """Compute thin lens ABCD matrix."""
        mat = thin_lens_matrix(self.focal_length)
        mat.label = self.name
        if abs(self.decenter) > 1e-12:
            mat = mat.with_decenter(self.decenter)
        if abs(self.tilt) > 1e-12:
            mat = mat.with_tilt(self.tilt, in_degrees=True)
        return mat

    def cardinal_points(self, wavelength_nm: float = 587.6) -> CardinalPoints:
        """Extract cardinal points for thin lens."""
        f = self.focal_length
        power = 1.0 / f
        return CardinalPoints(
            efl=f,
            bfl=f,
            ffl=f,
            p1=0.0,
            p2=0.0,
            f1=-f,
            f2=f,
            n1=0.0,
            n2=0.0,
            power=power,
            is_afocal=False,
            angular_magnification=1.0,
        )

    @property
    def efl(self) -> float:
        return self.focal_length

    @property
    def bfl(self) -> float:
        return self.focal_length

    @property
    def ffl(self) -> float:
        return self.focal_length

    @property
    def p1(self) -> float:
        return 0.0

    @property
    def p2(self) -> float:
        return 0.0

    def to_surfaces(self) -> List[Surface]:
        """Convert to a single ideal Surface for 3D layout integration."""
        return [
            Surface(
                name=self.name,
                surface_type=SurfaceType.STANDARD,
                radius=0.0,
                thickness=0.0,
                glass="AIR",
                semi_diameter=self.diameter / 2.0,
                decenter_y=self.decenter,
                tilt_x=self.tilt,
            )
        ]

    def get_parameters(self) -> Dict[str, float]:
        return {
            "focal_length": self.focal_length,
            "diameter": self.diameter,
            "decenter": self.decenter,
            "tilt": self.tilt,
        }

    def set_parameters(self, params: Dict[str, float]) -> None:
        param_map = {k.lower().strip(): v for k, v in params.items()}
        if "focal_length" in param_map:
            self.focal_length = float(param_map["focal_length"])
        elif "f" in param_map or "efl" in param_map:
            self.focal_length = float(param_map.get("f", param_map.get("efl")))

        if "diameter" in param_map:
            self.diameter = float(param_map["diameter"])
        elif "diam" in param_map or "d" in param_map:
            self.diameter = float(param_map.get("diam", param_map.get("d")))

        if "decenter" in param_map:
            self.decenter = float(param_map["decenter"])
        elif "dy" in param_map:
            self.decenter = float(param_map["dy"])

        if "tilt" in param_map:
            self.tilt = float(param_map["tilt"])
        elif "theta" in param_map:
            self.tilt = float(param_map["theta"])

    def perturb(self, param_name: str, delta: float) -> None:
        key = param_name.lower().strip()
        if key in ("focal_length", "f", "efl"):
            self.focal_length += delta
        elif key in ("diameter", "diam", "d"):
            self.diameter += delta
        elif key in ("decenter", "dy"):
            self.decenter += delta
        elif key in ("tilt", "theta"):
            self.tilt += delta
        else:
            raise KeyError(f"Unknown parameter '{param_name}' to perturb in ThinLens.")

    def step_by_step_calculation(self, wavelength_nm: float = 587.6) -> CalculationTrace:
        trace = CalculationTrace(f"Thin Lens Analysis: {self.name}")
        mat = thin_lens_matrix(self.focal_length, trace=trace)
        return trace

    def __repr__(self) -> str:
        return f"ThinLens({self.name}, f={self.focal_length}, D={self.diameter})"


class AchromaticDoublet(OpticalElement):
    """
    Achromatic Cemented Doublet Lens.

    Composed of two lenses sharing an internal cemented spherical boundary:
        Surface 1: Crown glass front (R1)
        Internal: Cemented boundary (R2), Crown -> Flint transition
        Surface 3: Flint glass rear (R3)
    """

    def __init__(
        self,
        r1: float,
        r2: float,
        r3: float,
        t1: float,
        t2: float,
        glass1: Union[str, float],
        glass2: Union[str, float],
        diameter: float = 25.4,
        decenter: float = 0.0,
        tilt: float = 0.0,
        name: str = "AchromaticDoublet",
    ):
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.r3 = float(r3)
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.glass1 = glass1
        self.glass2 = glass2
        self.diameter = float(diameter)
        self.decenter = float(decenter)
        self.tilt = float(tilt)
        self.name = name

    def refractive_indices(self, wavelength_nm: float = 587.6) -> Tuple[float, float]:
        def _get_n(g: Union[str, float]) -> float:
            if isinstance(g, (int, float)):
                return float(g)
            try:
                return float(g)
            except ValueError:
                pass
            return float(get_refractive_index(g, wavelength_nm / 1000.0))

        return _get_n(self.glass1), _get_n(self.glass2)

    def transfer_matrix(self, wavelength_nm: float = 587.6) -> TransferMatrix:
        n1, n2 = self.refractive_indices(wavelength_nm)

        M1 = refraction_matrix(self.r1, 1.0, n1)
        Mt1 = space_matrix(self.t1, n=n1)
        M2 = refraction_matrix(self.r2, n1, n2)
        Mt2 = space_matrix(self.t2, n=n2)
        M3 = refraction_matrix(self.r3, n2, 1.0)

        mat = M3 @ Mt2 @ M2 @ Mt1 @ M1
        mat.label = self.name
        if abs(self.decenter) > 1e-12:
            mat = mat.with_decenter(self.decenter)
        if abs(self.tilt) > 1e-12:
            mat = mat.with_tilt(self.tilt, in_degrees=True)
        return mat

    def cardinal_points(self, wavelength_nm: float = 587.6) -> CardinalPoints:
        return self.transfer_matrix(wavelength_nm).cardinal_points()

    def to_surfaces(self) -> List[Surface]:
        semi_diam = self.diameter / 2.0
        g1_str = self.glass1 if isinstance(self.glass1, str) else f"INDEX_{float(self.glass1):.4f}"
        g2_str = self.glass2 if isinstance(self.glass2, str) else f"INDEX_{float(self.glass2):.4f}"

        s1 = Surface(
            name=f"{self.name}_S1",
            surface_type=SurfaceType.STANDARD,
            radius=self.r1,
            thickness=self.t1,
            glass=g1_str,
            semi_diameter=semi_diam,
            decenter_y=self.decenter,
            tilt_x=self.tilt,
        )
        s2 = Surface(
            name=f"{self.name}_Cemented",
            surface_type=SurfaceType.STANDARD,
            radius=self.r2,
            thickness=self.t2,
            glass=g2_str,
            semi_diameter=semi_diam,
            decenter_y=self.decenter,
            tilt_x=self.tilt,
        )
        s3 = Surface(
            name=f"{self.name}_S3",
            surface_type=SurfaceType.STANDARD,
            radius=self.r3,
            thickness=0.0,
            glass="AIR",
            semi_diameter=semi_diam,
            decenter_y=self.decenter,
            tilt_x=self.tilt,
        )
        return [s1, s2, s3]

    def get_parameters(self) -> Dict[str, float]:
        n1, n2 = self.refractive_indices()
        return {
            "r1": self.r1,
            "r2": self.r2,
            "r3": self.r3,
            "t1": self.t1,
            "t2": self.t2,
            "index1": n1,
            "index2": n2,
            "diameter": self.diameter,
            "decenter": self.decenter,
            "tilt": self.tilt,
        }

    def set_parameters(self, params: Dict[str, float]) -> None:
        param_map = {k.lower().strip(): v for k, v in params.items()}
        if "r1" in param_map: self.r1 = float(param_map["r1"])
        if "r2" in param_map: self.r2 = float(param_map["r2"])
        if "r3" in param_map: self.r3 = float(param_map["r3"])
        if "t1" in param_map: self.t1 = float(param_map["t1"])
        if "t2" in param_map: self.t2 = float(param_map["t2"])
        if "index1" in param_map: self.glass1 = float(param_map["index1"])
        if "index2" in param_map: self.glass2 = float(param_map["index2"])
        if "diameter" in param_map: self.diameter = float(param_map["diameter"])
        if "decenter" in param_map: self.decenter = float(param_map["decenter"])
        if "tilt" in param_map: self.tilt = float(param_map["tilt"])

    def perturb(self, param_name: str, delta: float) -> None:
        key = param_name.lower().strip()
        if key in ("r1",): self.r1 += delta
        elif key in ("r2",): self.r2 += delta
        elif key in ("r3",): self.r3 += delta
        elif key in ("t1",): self.t1 += delta
        elif key in ("t2",): self.t2 += delta
        elif key in ("index1", "n1"):
            n1, _ = self.refractive_indices()
            self.glass1 = n1 + delta
        elif key in ("index2", "n2"):
            _, n2 = self.refractive_indices()
            self.glass2 = n2 + delta
        elif key in ("diameter", "d"): self.diameter += delta
        elif key in ("decenter", "dy"): self.decenter += delta
        elif key in ("tilt", "theta"): self.tilt += delta
        else:
            raise KeyError(f"Unknown parameter '{param_name}' to perturb in AchromaticDoublet.")

    def step_by_step_calculation(self, wavelength_nm: float = 587.6) -> CalculationTrace:
        trace = CalculationTrace(f"Achromatic Doublet: {self.name}")
        n1, n2 = self.refractive_indices(wavelength_nm)
        M1 = refraction_matrix(self.r1, 1.0, n1, trace=trace)
        Mt1 = space_matrix(self.t1, n=n1, trace=trace)
        M2 = refraction_matrix(self.r2, n1, n2, trace=trace)
        Mt2 = space_matrix(self.t2, n=n2, trace=trace)
        M3 = refraction_matrix(self.r3, n2, 1.0, trace=trace)
        M_tot = M3 @ Mt2 @ M2 @ Mt1 @ M1
        cp = M_tot.cardinal_points()
        trace.add_step(
            CalculationStep(
                name="Doublet Composite Matrix",
                formula="M = M3 @ Mt2 @ M2 @ Mt1 @ M1",
                inputs={"r1": self.r1, "r2": self.r2, "r3": self.r3, "t1": self.t1, "t2": self.t2},
                intermediate_values={"EFL": cp.efl, "BFL": cp.bfl, "det": M_tot.determinant()},
                matrix=M_tot,
            )
        )
        return trace
