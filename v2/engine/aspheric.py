"""
Aspheric Surface Formulations & Aspheric Physical Lens Component Engine.

Implements modular, extensible mathematical models for even polynomial aspheric surfaces,
conic sections, exact analytical surface normals, paraxial vertex curvature equivalence,
and the AsphericLens optical element adhering to the OpticalElement interface contract.
Based on standard optical design formulations (ISO 10110-12, Greivenkamp, University of Colorado).
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Sequence, Union
import numpy as np

from optics_app.engine.matrix_optics import (
    TransferMatrix,
    CardinalPoints,
    CalculationStep,
    CalculationTrace,
    thick_lens_matrix,
)
from optics_app.engine.thick_lens import OpticalElement
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.utils.materials import get_refractive_index


class SurfaceModel(ABC):
    """
    Abstract Base Class for mathematical surface models in 2D and 3D optics.

    Decouples geometric surface definition (sag, derivatives, surface normals, curvature)
    from paraxial matrix optics, sequential ray tracing, and optimization.
    Allows trivial extension with new surface physics (Zernike, Q-type, freeform, DOE)
    without modifying core engines.
    """

    @abstractmethod
    def sag(self, r: float) -> float:
        """
        Calculate exact geometric surface sag z(r) along the optical axis.

        Args:
            r: Radial distance from optical axis in mm (r = sqrt(x^2 + y^2)).

        Returns:
            Surface axial sag z in mm (z=0 at vertex r=0).
        """
        pass

    @abstractmethod
    def dsag_dr(self, r: float) -> float:
        """
        Calculate first radial derivative dz / dr of surface sag.

        Args:
            r: Radial distance from optical axis in mm.

        Returns:
            First radial derivative dz/dr (dimensionless slope).
        """
        pass

    @abstractmethod
    def normal(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        Calculate exact outward unit normal vector N_hat = (Nx, Ny, Nz) at coordinate (x, y).

        Convention:
            Outward normal pointing along positive z (toward direction of light propagation)
            with magnitude exactly equal to 1.0.

        Args:
            x: Transverse horizontal coordinate (mm).
            y: Transverse vertical coordinate (mm).

        Returns:
            Tuple (Nx, Ny, Nz) representing normalized 3D unit normal vector.
        """
        pass

    @abstractmethod
    def paraxial_curvature(self) -> float:
        """
        Calculate paraxial vertex curvature c = 1 / R (mm^-1).

        In Gaussian paraxial optics, first-order optical power is governed strictly
        by the vertex curvature c = d^2 z / d r^2 at r = 0, independent of conic
        constant k or higher-order aspheric coefficients.
        """
        pass


class AsphericSurfaceModel(SurfaceModel):
    """
    Standard Even Polynomial Aspheric Surface Model (ISO 10110-12 standard).

    Mathematical Surface Sag Formulation:
        z(r) = (c * r^2) / (1 + sqrt(1 - (1 + k) * c^2 * r^2)) + sum_{j=0}^{M-1} alpha_{2(j+2)} * r^{2(j+2)}

    where:
        c = 1 / R: Vertex curvature (mm^-1)
        k: Conic constant (Schwarzschild constant):
            k > 0: Oblate spheroid
            k = 0: Sphere
            -1 < k < 0: Prolate spheroid (ellipse)
            k = -1: Paraboloid
            k < -1: Hyperboloid
        alpha_{2(j+2)}: Even aspheric deformation coefficients:
            alpha_4 (r^4), alpha_6 (r^6), alpha_8 (r^8), alpha_10 (r^10), ...

    Exact Analytical Radial Derivative:
        dz / dr = (c * r) / sqrt(1 - (1 + k) * c^2 * r^2) + sum_{j=0}^{M-1} 2(j+2) * alpha_{2(j+2)} * r^{2(j+2) - 1}

    Exact 3D Outward Unit Normal Vector:
        grad(z - z(r)) = (-dz/dx, -dz/dy, 1) / sqrt(1 + (dz/dr)^2)
        where dz/dx = (dz/dr) * (x/r), dz/dy = (dz/dr) * (y/r).

    Paraxial Curvature & Vertex Equivalence Theorem:
        lim_{r -> 0} d^2 z / d r^2 = c = 1 / R.
        The conic constant k and even aspheric coefficients alpha contribute zero power
        to first-order paraxial ABCD transfer matrices.
    """

    def __init__(
        self,
        radius: float = float("inf"),
        conic: float = 0.0,
        aspheric_coeffs: Sequence[float] = (),
        curvature: Optional[float] = None,
        name: str = "AsphericSurface",
    ):
        """
        Initialize AsphericSurfaceModel.

        Args:
            radius: Vertex radius of curvature R in mm (inf or 0 denotes flat plane).
            conic: Conic constant k (default 0.0 for spherical base).
            aspheric_coeffs: Sequence of higher-order even polynomial coefficients
                             [alpha_4, alpha_6, alpha_8, ...].
            curvature: Optional direct curvature c = 1/R (mm^-1). Overrides radius if given.
            name: Descriptive surface label.
        """
        if curvature is not None:
            self._c = float(curvature)
            self._radius = 1.0 / self._c if abs(self._c) > 1e-15 else float("inf")
        else:
            is_flat = (radius == 0.0 or math.isinf(radius) or abs(radius) > 1e12)
            if is_flat:
                self._c = 0.0
                self._radius = float("inf") if radius != 0.0 else 0.0
            else:
                self._radius = float(radius)
                self._c = 1.0 / self._radius

        self._k = float(conic)
        self._aspheric_coeffs: List[float] = [float(a) for a in aspheric_coeffs]
        self.name = name

    @property
    def radius(self) -> float:
        """Vertex radius of curvature R (mm)."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        is_flat = (value == 0.0 or math.isinf(value) or abs(value) > 1e12)
        if is_flat:
            self._c = 0.0
            self._radius = float("inf") if value != 0.0 else 0.0
        else:
            self._radius = float(value)
            self._c = 1.0 / self._radius

    @property
    def curvature(self) -> float:
        """Vertex curvature c = 1/R (mm^-1)."""
        return self._c

    @curvature.setter
    def curvature(self, value: float) -> None:
        self._c = float(value)
        self._radius = 1.0 / self._c if abs(self._c) > 1e-15 else float("inf")

    @property
    def conic(self) -> float:
        """Conic constant k."""
        return self._k

    @conic.setter
    def conic(self, value: float) -> None:
        self._k = float(value)

    @property
    def k(self) -> float:
        """Alias for conic constant k."""
        return self._k

    @k.setter
    def k(self, value: float) -> None:
        self._k = float(value)

    @property
    def aspheric_coeffs(self) -> List[float]:
        """List of even polynomial coefficients [alpha_4, alpha_6, ...]."""
        return self._aspheric_coeffs

    @aspheric_coeffs.setter
    def aspheric_coeffs(self, coeffs: Sequence[float]) -> None:
        self._aspheric_coeffs = [float(a) for a in coeffs]

    @property
    def coeffs(self) -> List[float]:
        """Alias for aspheric_coeffs."""
        return self._aspheric_coeffs

    def paraxial_curvature(self) -> float:
        """
        Return paraxial vertex curvature c = 1 / R (mm^-1).

        Satisfies the Paraxial Curvature & Vertex Equivalence Theorem:
        for first-order optics, vertex power is determined strictly by c.
        """
        return self._c

    def sag(self, r: float) -> float:
        """
        Calculate exact geometric surface sag z(r) at radial height r (mm).
        """
        r_val = float(abs(r))
        if r_val < 1e-15:
            return 0.0

        # Conic section base sag
        if abs(self._c) < 1e-15:
            z_conic = 0.0
        else:
            c = self._c
            k = self._k
            r2 = r_val * r_val
            radicand = 1.0 - (1.0 + k) * (c * c) * r2
            if radicand < 0.0:
                raise ValueError(
                    f"Radial height r={r} mm is outside the real surface domain: "
                    f"radicand = 1 - (1+k)*c^2*r^2 = {radicand} < 0."
                )
            z_conic = (c * r2) / (1.0 + math.sqrt(radicand))

        # Higher-order even polynomial terms: sum alpha_{2(j+2)} * r^{2(j+2)}
        z_poly = 0.0
        if self._aspheric_coeffs:
            r2 = r_val * r_val
            r_pow = r2 * r2  # r^4
            for alpha in self._aspheric_coeffs:
                z_poly += alpha * r_pow
                r_pow *= r2

        return float(z_conic + z_poly)

    def dsag_dr(self, r: float) -> float:
        """
        Calculate exact analytical first radial derivative dz/dr at height r (mm).
        """
        r_val = float(r)
        if abs(r_val) < 1e-15:
            return 0.0

        r_abs = abs(r_val)
        r2 = r_abs * r_abs

        # Conic section derivative: c * r / sqrt(1 - (1+k)*c^2*r^2)
        if abs(self._c) < 1e-15:
            dz_conic = 0.0
        else:
            c = self._c
            k = self._k
            radicand = 1.0 - (1.0 + k) * (c * c) * r2
            if radicand <= 0.0:
                raise ValueError(
                    f"Radial height r={r} mm is outside the real surface domain: "
                    f"radicand = {radicand} <= 0."
                )
            dz_conic = (c * r_val) / math.sqrt(radicand)

        # Higher-order even polynomial derivative: sum 2(j+2) * alpha_{2(j+2)} * r^{2(j+2) - 1}
        dz_poly = 0.0
        if self._aspheric_coeffs:
            sign = 1.0 if r_val >= 0 else -1.0
            r_pow = r_abs * r2  # r^3
            for j, alpha in enumerate(self._aspheric_coeffs):
                order = 2 * (j + 2)
                dz_poly += order * alpha * r_pow
                r_pow *= r2
            dz_poly *= sign

        return float(dz_conic + dz_poly)

    def normal(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        Calculate exact outward unit normal vector N_hat = (Nx, Ny, Nz) at (x, y).

        Formula:
            r = sqrt(x^2 + y^2)
            dz/dx = (dz/dr) * (x/r),  dz/dy = (dz/dr) * (y/r)
            N_hat = (-dz/dx, -dz/dy, 1) / sqrt(1 + (dz/dr)^2)
        """
        x_val = float(x)
        y_val = float(y)
        r = math.hypot(x_val, y_val)

        if r < 1e-15:
            return (0.0, 0.0, 1.0)

        dzdr = self.dsag_dr(r)
        dz_dx = dzdr * (x_val / r)
        dz_dy = dzdr * (y_val / r)

        norm = math.sqrt(1.0 + dzdr * dzdr)
        nx = -dz_dx / norm
        ny = -dz_dy / norm
        nz = 1.0 / norm

        return (float(nx), float(ny), float(nz))

    def __repr__(self) -> str:
        coeffs_str = f", coeffs={self._aspheric_coeffs}" if self._aspheric_coeffs else ""
        return f"AsphericSurfaceModel({self.name}, R={self._radius}, k={self._k}{coeffs_str})"


class SphericalSurfaceModel(AsphericSurfaceModel):
    """
    Spherical Optical Surface Model.

    Special case of AsphericSurfaceModel with conic constant k = 0.0 and zero polynomial terms.
    """

    def __init__(self, radius: float = float("inf"), name: str = "SphericalSurface"):
        super().__init__(radius=radius, conic=0.0, aspheric_coeffs=(), name=name)


class PlanarSurfaceModel(AsphericSurfaceModel):
    """
    Planar Flat Optical Surface Model (R = inf, c = 0.0, k = 0.0).
    """

    def __init__(self, name: str = "PlanarSurface"):
        super().__init__(radius=float("inf"), conic=0.0, aspheric_coeffs=(), name=name)


class AsphericLens(OpticalElement):
    """
    Physical Aspheric Lens Component.

    Encapsulates front and rear optical surfaces (supporting spherical, planar,
    conic, or even polynomial aspheric profiles), center thickness, glass material
    or constant refractive index, clear aperture diameter, and 3x3 augmented
    tolerancing parameters (decenter, tilt).

    Conforms to the OpticalElement interface contract for full compatibility with:
        - matrix_optics (ABCD TransferMatrix, CardinalPoints)
        - multi_component (MultiComponentSystem cascading)
        - lens_merit & lens_optimizer (TargetEFLOperand, TRF & DLS optimization)
        - tolerancing (PerturbableElement, SensitivityAnalysis, MonteCarloSimulation)
        - surface / lde (3D Surface stack bridging)
    """

    def __init__(
        self,
        r1: Optional[Union[float, SurfaceModel]] = None,
        r2: Optional[Union[float, SurfaceModel]] = None,
        thickness: float = 5.0,
        glass_or_index: Union[str, float] = 1.5168,
        k1: float = 0.0,
        k2: float = 0.0,
        coeffs1: Sequence[float] = (),
        coeffs2: Sequence[float] = (),
        diameter: float = 25.4,
        decenter: float = 0.0,
        tilt: float = 0.0,
        name: str = "AsphericLens",
        surface1: Optional[SurfaceModel] = None,
        surface2: Optional[SurfaceModel] = None,
    ):
        """
        Initialize AsphericLens.

        Args:
            r1: Front surface radius (mm) or SurfaceModel instance.
            r2: Rear surface radius (mm) or SurfaceModel instance.
            thickness: Center thickness along optical axis (mm).
            glass_or_index: Glass catalog name (e.g. 'N-BK7') or refractive index (e.g. 1.5).
            k1: Front surface conic constant k1 (ignored if surface1 is provided).
            k2: Rear surface conic constant k2 (ignored if surface2 is provided).
            coeffs1: Front surface even aspheric coefficients [alpha_4, alpha_6, ...].
            coeffs2: Rear surface even aspheric coefficients [alpha_4, alpha_6, ...].
            diameter: Outer clear aperture diameter (mm).
            decenter: Transverse decenter Delta y (mm).
            tilt: Paraxial tilt angle around x-axis (degrees).
            name: Component identifier string.
            surface1: Optional explicit front SurfaceModel instance.
            surface2: Optional explicit rear SurfaceModel instance.
        """
        # Validate thickness
        if thickness < 0.0:
            raise ValueError(f"Lens thickness cannot be negative: {thickness}")
        self.thickness = float(thickness)

        # Validate diameter
        if diameter <= 0.0:
            raise ValueError(f"Lens diameter must be strictly positive: {diameter}")
        self.diameter = float(diameter)

        # Validate refractive index / material
        self.glass_or_index = glass_or_index
        n_test = self.refractive_index()
        if n_test <= 0.0:
            raise ValueError(f"Refractive index must be strictly positive: {n_test}")

        self.decenter = float(decenter)
        self.tilt = float(tilt)
        self.name = name

        # Front SurfaceModel initialization
        if surface1 is not None:
            self.surface1: SurfaceModel = surface1
        elif isinstance(r1, SurfaceModel):
            self.surface1 = r1
        else:
            r1_val = 50.0 if r1 is None else float(r1)
            self.surface1 = AsphericSurfaceModel(
                radius=r1_val,
                conic=k1,
                aspheric_coeffs=coeffs1,
                name=f"{name}_Front",
            )

        # Rear SurfaceModel initialization
        if surface2 is not None:
            self.surface2: SurfaceModel = surface2
        elif isinstance(r2, SurfaceModel):
            self.surface2 = r2
        else:
            r2_val = -50.0 if r2 is None else float(r2)
            self.surface2 = AsphericSurfaceModel(
                radius=r2_val,
                conic=k2,
                aspheric_coeffs=coeffs2,
                name=f"{name}_Rear",
            )

    @property
    def c1(self) -> float:
        """Front surface vertex curvature (mm^-1)."""
        return self.surface1.paraxial_curvature()

    @property
    def c2(self) -> float:
        """Rear surface vertex curvature (mm^-1)."""
        return self.surface2.paraxial_curvature()

    @property
    def r1(self) -> float:
        """Front surface vertex radius (mm)."""
        c = self.c1
        return 1.0 / c if abs(c) > 1e-15 else float("inf")

    @r1.setter
    def r1(self, value: float) -> None:
        if isinstance(self.surface1, AsphericSurfaceModel):
            self.surface1.radius = float(value)
        else:
            self.surface1 = AsphericSurfaceModel(radius=float(value), name=f"{self.name}_Front")

    @property
    def r2(self) -> float:
        """Rear surface vertex radius (mm)."""
        c = self.c2
        return 1.0 / c if abs(c) > 1e-15 else float("inf")

    @r2.setter
    def r2(self, value: float) -> None:
        if isinstance(self.surface2, AsphericSurfaceModel):
            self.surface2.radius = float(value)
        else:
            self.surface2 = AsphericSurfaceModel(radius=float(value), name=f"{self.name}_Rear")

    @property
    def k1(self) -> float:
        """Front surface conic constant k1."""
        return getattr(self.surface1, "conic", 0.0)

    @k1.setter
    def k1(self, value: float) -> None:
        if hasattr(self.surface1, "conic"):
            self.surface1.conic = float(value)

    @property
    def k2(self) -> float:
        """Rear surface conic constant k2."""
        return getattr(self.surface2, "conic", 0.0)

    @k2.setter
    def k2(self, value: float) -> None:
        if hasattr(self.surface2, "conic"):
            self.surface2.conic = float(value)

    @property
    def glass_name(self) -> str:
        """Return glass catalog identifier string."""
        if isinstance(self.glass_or_index, str):
            return self.glass_or_index
        return f"INDEX_{float(self.glass_or_index):.4f}"

    def refractive_index(self, wavelength_nm: float = 587.6) -> float:
        """
        Evaluate refractive index at optical wavelength (nm).
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

    def transfer_matrix(self, wavelength_nm: float = 587.6) -> TransferMatrix:
        """
        Compute paraxial 2x2/3x3 ABCD ray transfer matrix through the aspheric lens.

        Paraxial Curvature & Vertex Equivalence Theorem:
            In Gaussian paraxial ray optics, higher-order aspheric coefficients and conic
            constants contribute zero first-order power. The paraxial transfer matrix is
            governed strictly by the vertex curvatures c1 and c2.
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

        # Apply 3x3 augmented tolerancing perturbations if non-zero
        if abs(self.decenter) > 1e-12:
            mat = mat.with_decenter(self.decenter)
        if abs(self.tilt) > 1e-12:
            mat = mat.with_tilt(self.tilt, in_degrees=True)

        return mat

    def cardinal_points(self, wavelength_nm: float = 587.6) -> CardinalPoints:
        """
        Extract paraxial cardinal points (EFL, BFL, FFL, P1, P2, N1, N2).
        """
        return self.transfer_matrix(wavelength_nm).cardinal_points()

    def step_by_step_calculation(self, wavelength_nm: float = 587.6) -> CalculationTrace:
        """
        Produce transparent, step-by-step diagnostic calculation trace.
        """
        n = self.refractive_index(wavelength_nm)
        trace = CalculationTrace(f"Aspheric Lens Analysis: {self.name}")

        mat = thick_lens_matrix(
            R1=self.r1,
            R2=self.r2,
            thickness=self.thickness,
            n_lens=n,
            n_ambient=1.0,
            trace=trace,
        )

        cp = mat.cardinal_points()
        asphere_step = CalculationStep(
            name="Aspheric Surface Parameters & Vertex Equivalence Verification",
            formula="Phi_paraxial = (n - 1)*c1 + (1 - n)*c2 - (t/n)*(n - 1)*(1 - n)*c1*c2",
            inputs={
                "c1": self.c1,
                "c2": self.c2,
                "k1": self.k1,
                "k2": self.k2,
                "thickness": self.thickness,
                "index": n,
            },
            intermediate_values={
                "R1_vertex": self.r1,
                "R2_vertex": self.r2,
                "EFL": cp.efl,
                "BFL": cp.bfl,
                "P1": cp.p1,
                "P2": cp.p2,
                "det": mat.determinant(),
            },
            matrix=mat,
            notes=(
                "Vertex Equivalence Theorem: Paraxial power is governed strictly by vertex "
                "curvatures c1 and c2; conic constants and higher-order terms control aberrations."
            ),
        )
        trace.add_step(asphere_step)
        mat.trace = trace
        return trace

    def to_surfaces(self) -> List[Surface]:
        """
        Bridge to sequential 3D Surface stack with full aspheric coefficients.
        """
        semi_diam = self.diameter / 2.0

        k1_val = getattr(self.surface1, "k", 0.0)
        c1_coeffs = getattr(self.surface1, "aspheric_coeffs", [])
        is_aspheric1 = (abs(k1_val) > 1e-12 or len(c1_coeffs) > 0)

        s1 = Surface(
            name=f"{self.name}_Front",
            surface_type=SurfaceType.ASPHERIC if is_aspheric1 else SurfaceType.STANDARD,
            radius=self.r1,
            thickness=self.thickness,
            glass=self.glass_name,
            semi_diameter=semi_diam,
            conic=k1_val,
            asphere_coeffs=list(c1_coeffs),
            decenter_y=self.decenter,
            tilt_x=self.tilt,
        )

        k2_val = getattr(self.surface2, "k", 0.0)
        c2_coeffs = getattr(self.surface2, "aspheric_coeffs", [])
        is_aspheric2 = (abs(k2_val) > 1e-12 or len(c2_coeffs) > 0)

        s2 = Surface(
            name=f"{self.name}_Rear",
            surface_type=SurfaceType.ASPHERIC if is_aspheric2 else SurfaceType.STANDARD,
            radius=self.r2,
            thickness=0.0,
            glass="AIR",
            semi_diameter=semi_diam,
            conic=k2_val,
            asphere_coeffs=list(c2_coeffs),
            decenter_y=self.decenter,
            tilt_x=self.tilt,
        )

        return [s1, s2]

    def get_parameters(self) -> Dict[str, float]:
        """
        Return dictionary of all perturbable parameters for optimization and tolerancing.
        """
        return {
            "c1": self.c1,
            "c2": self.c2,
            "r1": self.r1,
            "r2": self.r2,
            "k1": self.k1,
            "k2": self.k2,
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

        if "c1" in param_map:
            c1_val = float(param_map["c1"])
            if isinstance(self.surface1, AsphericSurfaceModel):
                self.surface1.curvature = c1_val
            else:
                r1_new = 1.0 / c1_val if abs(c1_val) > 1e-15 else float("inf")
                self.surface1 = AsphericSurfaceModel(radius=r1_new, name=f"{self.name}_Front")
        elif "r1" in param_map:
            self.r1 = float(param_map["r1"])

        if "c2" in param_map:
            c2_val = float(param_map["c2"])
            if isinstance(self.surface2, AsphericSurfaceModel):
                self.surface2.curvature = c2_val
            else:
                r2_new = 1.0 / c2_val if abs(c2_val) > 1e-15 else float("inf")
                self.surface2 = AsphericSurfaceModel(radius=r2_new, name=f"{self.name}_Rear")
        elif "r2" in param_map:
            self.r2 = float(param_map["r2"])

        if "k1" in param_map:
            self.k1 = float(param_map["k1"])
        if "k2" in param_map:
            self.k2 = float(param_map["k2"])

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
        Perturb an individual parameter for sensitivity analysis or tolerancing.
        """
        key = param_name.lower().strip()
        if key in ("r1",):
            self.r1 += delta
        elif key in ("r2",):
            self.r2 += delta
        elif key in ("c1",):
            if isinstance(self.surface1, AsphericSurfaceModel):
                self.surface1.curvature += delta
            else:
                new_c = self.c1 + delta
                self.r1 = 1.0 / new_c if abs(new_c) > 1e-15 else float("inf")
        elif key in ("c2",):
            if isinstance(self.surface2, AsphericSurfaceModel):
                self.surface2.curvature += delta
            else:
                new_c = self.c2 + delta
                self.r2 = 1.0 / new_c if abs(new_c) > 1e-15 else float("inf")
        elif key in ("k1",):
            self.k1 += delta
        elif key in ("k2",):
            self.k2 += delta
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
                f"Unknown parameter '{param_name}' to perturb in AsphericLens. "
                f"Supported: r1, r2, c1, c2, k1, k2, thickness, index, diameter, decenter, tilt."
            )

    def clone(self) -> "AsphericLens":
        """Return an independent copy of this lens."""
        c1_coeffs = list(getattr(self.surface1, "aspheric_coeffs", []))
        c2_coeffs = list(getattr(self.surface2, "aspheric_coeffs", []))
        return AsphericLens(
            r1=self.r1,
            r2=self.r2,
            thickness=self.thickness,
            glass_or_index=self.glass_or_index,
            k1=self.k1,
            k2=self.k2,
            coeffs1=c1_coeffs,
            coeffs2=c2_coeffs,
            diameter=self.diameter,
            decenter=self.decenter,
            tilt=self.tilt,
            name=f"{self.name}_Clone",
        )

    def __repr__(self) -> str:
        return (
            f"AsphericLens({self.name}, R1={self.r1:.4g}, R2={self.r2:.4g}, "
            f"k1={self.k1:.4g}, k2={self.k2:.4g}, t={self.thickness:.4g}, "
            f"glass={self.glass_name}, D={self.diameter:.4g})"
        )


__all__ = [
    "SurfaceModel",
    "AsphericSurfaceModel",
    "SphericalSurfaceModel",
    "PlanarSurfaceModel",
    "AsphericLens",
]
