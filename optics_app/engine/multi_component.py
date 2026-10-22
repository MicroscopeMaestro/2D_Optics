"""
Multi-Component Optical System & Galileo Imaging Formulations.

Implements paraxial multi-component optical train management, transfer matrix cascading,
afocal diagnostics, angular and lateral magnification calculations, calculation tracing,
and factory functions for Galilean imaging layouts.
Based on standard textbook formulations (University of Colorado Optical Engineering, Greivenkamp).
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Tuple, Sequence, Union, Iterator
import numpy as np

from optics_app.engine.matrix_optics import (
    TransferMatrix,
    CardinalPoints,
    CalculationStep,
    CalculationTrace,
    space_matrix,
)
from optics_app.engine.thick_lens import OpticalElement, ThinLens, ThickLens
from optics_app.engine.surface import Surface



class MultiComponentSystem(OpticalElement):
    """
    Multi-Component Optical System Manager.

    Coordinates an ordered sequence of optical components (ThickLens, ThinLens, AsphericLens,
    AchromaticDoublet, etc.) separated by axial propagation distances in homogeneous media.
    Cascades paraxial transfer matrices:
        M_sys = M_N @ T_{N-1} @ M_{N-1} @ ... @ T_1 @ M_1

    Provides:
        - Exact system ABCD ray transfer matrix and determinant checks
        - Paraxial angular magnification M_alpha = D (for afocal systems in air)
        - Paraxial lateral magnification m = A (for afocal systems in air)
        - Afocal condition detection (|C| < 1e-9 mm^-1)
        - Paraxial ray tracing (propagate_ray)
        - 3D sequential Surface stack conversion (to_surfaces)
        - Full parameter inspection, modification, and perturbation for tolerancing
        - Step-by-step diagnostic calculation trace
    """

    def __init__(
        self,
        elements: Optional[Sequence[Union[Tuple[OpticalElement, float], OpticalElement]]] = None,
        separations: Optional[Sequence[float]] = None,
        name: str = "MultiComponentSystem",
    ):
        """
        Initialize MultiComponentSystem.

        Args:
            elements: Sequence of either:
                      - Tuples: (OpticalElement, distance_to_next_mm)
                      - OpticalElement instances (with distances specified via separations)
            separations: Optional sequence of inter-element separation distances (mm).
                         Used if elements contains only OpticalElement instances.
            name: Descriptive system identifier.
        """
        self.name = name
        self.elements: List[Tuple[OpticalElement, float]] = []
        self._elements: List[OpticalElement] = []
        self._separations: List[float] = []

        if elements is not None:
            for i, item in enumerate(elements):
                if isinstance(item, tuple):
                    elem, dist = item
                    self.elements.append((elem, float(dist)))
                    self._elements.append(elem)
                    self._separations.append(float(dist))
                else:
                    elem = item
                    dist = float(separations[i]) if (separations is not None and i < len(separations)) else 0.0
                    self.elements.append((elem, dist))
                    self._elements.append(elem)
                    self._separations.append(dist)

    @property
    def components(self) -> List[OpticalElement]:
        """Return list of optical elements contained in this system."""
        return self._elements

    def add_element(self, element: OpticalElement, distance_to_next: float = 0.0) -> None:
        """
        Append an optical component with its axial separation to the subsequent component.

        Args:
            element: OpticalElement instance to append.
            distance_to_next: Axial distance (mm) from this element's rear vertex to next element.
        """
        self.elements.append((element, float(distance_to_next)))
        self._elements.append(element)
        self._separations.append(float(distance_to_next))

    def system_matrix(self, wavelength_nm: float = 587.6) -> TransferMatrix:
        """
        Compute the composite paraxial transfer matrix cascading all elements and air spaces.

        Multiplication order:
            Light enters element 0, translates d_0, enters element 1, ..., through element N-1.
            M_sys = M_{N-1} @ T_{N-2} @ M_{N-2} @ ... @ T_0 @ M_0
        """
        if not self._elements:
            return TransferMatrix(A=1.0, B=0.0, C=0.0, D=1.0, label="EmptySystem")

        trace = CalculationTrace(f"Multi-Component System Derivation: {self.name}")

        # First component
        first_elem = self._elements[0]
        curr_mat = first_elem.transfer_matrix(wavelength_nm)
        trace.add_step(
            CalculationStep(
                name=f"Component 1: {getattr(first_elem, 'name', 'Element_1')}",
                formula="M_1 = element_1.transfer_matrix()",
                inputs={"index": 0, "name": getattr(first_elem, "name", "Element_1")},
                matrix=curr_mat,
                notes="Initial entrance component",
            )
        )

        # Cascade through remaining components
        for i in range(len(self._elements) - 1):
            d = self._separations[i]
            T_mat = space_matrix(d, n=1.0, trace=trace)
            curr_mat = T_mat @ curr_mat

            next_elem = self._elements[i + 1]
            M_next = next_elem.transfer_matrix(wavelength_nm)
            curr_mat = M_next @ curr_mat

            trace.add_step(
                CalculationStep(
                    name=f"Cascade through Component {i + 2}: {getattr(next_elem, 'name', f'Element_{i+2}')}",
                    formula=f"M_{{cumulative}} = M_{{{i+2}}} @ T_{{{i+1}}} @ M_{{prev}}",
                    inputs={"separation_d": d, "element_index": i + 1, "name": getattr(next_elem, "name", f"Element_{i+2}")},
                    matrix=curr_mat,
                    notes=f"Separation {d} mm followed by component {i+2}",
                )
            )

        curr_mat.trace = trace
        curr_mat.label = self.name
        return curr_mat

    def transfer_matrix(self, wavelength_nm: float = 587.6) -> TransferMatrix:
        """Alias for system_matrix adhering to OpticalElement interface."""
        return self.system_matrix(wavelength_nm)

    def cardinal_points(self, wavelength_nm: float = 587.6) -> CardinalPoints:
        """
        Extract paraxial cardinal points from the net system matrix.
        """
        return self.system_matrix(wavelength_nm).cardinal_points()

    def angular_magnification(self, wavelength_nm: float = 587.6) -> float:
        """
        Calculate paraxial angular magnification M_alpha = d theta_out / d theta_in.

        For systems in air (n_in = n_out = 1.0), M_alpha = D.
        For an afocal Galilean telescope: M_alpha = -f1 / f2.
        """
        mat = self.system_matrix(wavelength_nm)
        return float(mat.D)

    def lateral_magnification(self, wavelength_nm: float = 587.6) -> float:
        """
        Calculate paraxial lateral magnification m = d y_out / d y_in.

        For systems in air (n_in = n_out = 1.0), m = A.
        For an afocal Galilean telescope: m = 1 / M_alpha.
        """
        mat = self.system_matrix(wavelength_nm)
        return float(mat.A)

    def is_afocal(self, wavelength_nm: float = 587.6) -> bool:
        """
        Return True if system optical power |C| < 1e-9 mm^-1 (afocal condition).
        """
        mat = self.system_matrix(wavelength_nm)
        return abs(mat.C) < 1e-9

    def propagate_ray(
        self,
        y: float = 0.0,
        theta: float = 0.0,
        u: Optional[float] = None,
        wavelength_nm: float = 587.6,
    ) -> Tuple[float, float]:
        """
        Propagate paraxial ray [y, theta]^T through the multi-component system.

        Args:
            y: Entrance ray height in mm.
            theta: Entrance ray angle in radians.
            u: Optional alias for theta (radians).
            wavelength_nm: Optical wavelength in nm.

        Returns:
            Tuple (y_out, theta_out) where y_out is height (mm) and theta_out is angle (rad).
        """
        angle = float(u if u is not None else theta)
        mat = self.system_matrix(wavelength_nm)
        return mat.propagate_ray(y=float(y), u=angle)

    def propagate(
        self,
        y: float = 0.0,
        theta: float = 0.0,
        u: Optional[float] = None,
        wavelength_nm: float = 587.6,
    ) -> Tuple[float, float]:
        """Convenience alias for propagate_ray."""
        return self.propagate_ray(y=y, theta=theta, u=u, wavelength_nm=wavelength_nm)

    def step_by_step_calculation(self, wavelength_nm: float = 587.6) -> CalculationTrace:
        """
        Return comprehensive step-by-step calculation trace for manual debugging.
        """
        mat = self.system_matrix(wavelength_nm)
        if mat.trace is not None:
            return mat.trace
        trace = CalculationTrace(f"Calculation for {self.name}")
        trace.add_step(
            CalculationStep(
                name="System Matrix Cascade",
                formula="M_sys = M_N @ ... @ T_1 @ M_1",
                inputs={"num_components": len(self._elements)},
                matrix=mat,
            )
        )
        return trace

    def to_surfaces(self) -> List[Surface]:
        """
        Bridge all contained elements into a contiguous sequential 3D Surface stack.

        Applies inter-element separation distances as the axial thickness of the
        exit surface of each component.
        """
        all_surfaces: List[Surface] = []
        for i, elem in enumerate(self._elements):
            surfs = elem.to_surfaces()
            if not surfs:
                continue
            # Assign inter-element distance as the thickness of the last surface of this element
            if i < len(self._separations):
                sep = self._separations[i]
                surfs[-1].thickness = float(sep)
            all_surfaces.extend(surfs)
        return all_surfaces

    def get_parameters(self) -> Dict[str, float]:
        """
        Return dictionary of all perturbable parameters across components and separations.

        Keys:
            elem{i}_{param_name}: Parameter of component i.
            d{i}: Separation distance between component i and i+1.
        """
        params: Dict[str, float] = {}
        for i, elem in enumerate(self._elements):
            elem_params = elem.get_parameters()
            for k, v in elem_params.items():
                params[f"elem{i}_{k}"] = float(v)
        for i, sep in enumerate(self._separations):
            params[f"d{i}"] = float(sep)
        return params

    def set_parameters(self, params: Dict[str, float]) -> None:
        """
        Update physical parameters from dictionary.
        """
        for k, v in params.items():
            k_lower = k.lower().strip()
            if k_lower.startswith("d") and k_lower[1:].isdigit():
                idx = int(k_lower[1:])
                if 0 <= idx < len(self._separations):
                    val = float(v)
                    self._separations[idx] = val
                    self.elements[idx] = (self._elements[idx], val)
            elif k_lower.startswith("elem"):
                parts = k_lower.split("_", 1)
                if len(parts) == 2 and parts[0][4:].isdigit():
                    elem_idx = int(parts[0][4:])
                    sub_param = parts[1]
                    if 0 <= elem_idx < len(self._elements):
                        self._elements[elem_idx].set_parameters({sub_param: float(v)})

    def perturb(self, param_name: str, delta: float) -> None:
        """
        Perturb a parameter by delta for sensitivity analysis or tolerancing.
        """
        k_lower = param_name.lower().strip()
        if k_lower.startswith("d") and k_lower[1:].isdigit():
            idx = int(k_lower[1:])
            if 0 <= idx < len(self._separations):
                self._separations[idx] += delta
                self.elements[idx] = (self._elements[idx], self._separations[idx])
        elif k_lower.startswith("elem"):
            parts = k_lower.split("_", 1)
            if len(parts) == 2 and parts[0][4:].isdigit():
                elem_idx = int(parts[0][4:])
                sub_param = parts[1]
                if 0 <= elem_idx < len(self._elements):
                    self._elements[elem_idx].perturb(sub_param, delta)
        else:
            raise KeyError(
                f"Unknown parameter '{param_name}' to perturb in MultiComponentSystem. "
                f"Available prefixes: d0, d1, ..., elem0_*, elem1_*."
            )

    def __len__(self) -> int:
        return len(self._elements)

    def __getitem__(self, idx: int) -> OpticalElement:
        return self._elements[idx]

    def __iter__(self) -> Iterator[OpticalElement]:
        return iter(self._elements)

    def __repr__(self) -> str:
        elems_str = ", ".join(getattr(e, "name", e.__class__.__name__) for e in self._elements)
        return f"MultiComponentSystem({self.name}, N={len(self._elements)}, elements=[{elems_str}])"


def create_galileo_system(
    f1: float,
    f2: float,
    d: Optional[float] = None,
    diameter1: float = 25.4,
    diameter2: float = 12.7,
    thick: bool = False,
    n: float = 1.5,
    t1: float = 5.0,
    t2: float = 3.0,
) -> MultiComponentSystem:
    """
    Factory function constructing a classic Galilean imaging / telescope layout.

    Galilean Configuration:
        - Objective: Converging lens with focal length f1 > 0
        - Eyepiece:  Diverging lens with focal length f2 < 0
        - Default Afocal Separation:
            d = f1 + f2 = f1 - |f2|

    Afocal Benchmark Guarantees:
        - Net optical power C == 0.0 mm^-1 (det(M) == 1.0)
        - Paraxial angular magnification M_alpha = D = -f1 / f2 = +4.0 (for f1=100, f2=-25)
        - Paraxial lateral beam expansion factor m = A = 1 / M_alpha = +0.25

    Thick-Lens Mode (thick=True):
        Synthesizes physical thick lenses targeting exact EFLs f1 and f2, then sets
        the physical vertex-to-vertex air gap d_air = (f1 + f2) + P12 - P21
        accounting for principal plane offsets to strictly maintain afocal condition.

    Args:
        f1: Objective focal length in mm (positive).
        f2: Eyepiece focal length in mm (negative; if positive magnitude provided, converted to -abs(f2)).
        d: Optional explicit axial separation distance (mm). Defaults to afocal separation.
        diameter1: Objective clear aperture diameter (mm).
        diameter2: Eyepiece clear aperture diameter (mm).
        thick: Whether to construct physical thick lenses (True) or thin lenses (False).
        n: Glass refractive index for thick lens construction (default 1.5).
        t1: Objective center thickness in mm (for thick lens mode).
        t2: Eyepiece center thickness in mm (for thick lens mode).

    Returns:
        MultiComponentSystem configured as an afocal Galilean telescope / beam expander.
    """
    f1_val = float(f1)
    f2_val = float(f2)

    # In Galilean telescope, eyepiece is diverging (f2 < 0)
    # Support callers passing either signed f2=-25 or unsigned magnitude f2=25
    f2_eff = -abs(f2_val) if f2_val > 0 else f2_val

    if not thick:
        # Standard Paraxial Thin Lens Configuration
        obj = ThinLens(focal_length=f1_val, diameter=diameter1, name="Galileo_Objective")
        eye = ThinLens(focal_length=f2_eff, diameter=diameter2, name="Galileo_Eyepiece")

        sep = float(d) if d is not None else (f1_val + f2_eff)
        return MultiComponentSystem(
            elements=[(obj, sep), (eye, 0.0)],
            name=f"GalileoSystem(f1={f1_val}, f2={f2_eff})",
        )

    # Physical Thick Lens Galilean Configuration
    # Analytical solution for symmetric lens radii:
    # Phi = 2*(n-1)/R - (t/n)*((n-1)/R)^2 => let x = (n-1)/R
    # (t/n)*x^2 - 2*x + 1/f = 0
    # x = (1 - sqrt(1 - t/(n*f))) / (t/n)

    def _thick_symmetric_radii(f_target: float, t_lens: float, n_lens: float) -> float:
        arg = 1.0 - t_lens / (n_lens * f_target)
        if arg < 0:
            raise ValueError(f"Cannot achieve focal length {f_target} with thickness {t_lens} and index {n_lens}")
        x = (1.0 - math.sqrt(arg)) / (t_lens / n_lens)
        r = (n_lens - 1.0) / x
        return float(r)

    # Objective (biconvex)
    r_obj = _thick_symmetric_radii(f1_val, t1, n)
    obj_thick = ThickLens(
        r1=r_obj,
        r2=-r_obj,
        thickness=t1,
        glass_or_index=n,
        diameter=diameter1,
        name="Galileo_Thick_Objective",
    )

    # Eyepiece (biconcave)
    r_eye = _thick_symmetric_radii(f2_eff, t2, n)
    eye_thick = ThickLens(
        r1=r_eye,
        r2=-r_eye,
        thickness=t2,
        glass_or_index=n,
        diameter=diameter2,
        name="Galileo_Thick_Eyepiece",
    )

    if d is not None:
        d_air = float(d)
    else:
        # Principal plane separation: d_P = EFL1 + EFL2
        # Air gap: d_air = d_P + P12 - P21
        p12 = obj_thick.p2
        p21 = eye_thick.p1
        d_p = f1_val + f2_eff
        d_air = d_p + p12 - p21

    return MultiComponentSystem(
        elements=[(obj_thick, d_air), (eye_thick, 0.0)],
        name=f"GalileoThickSystem(f1={f1_val}, f2={f2_eff})",
    )


__all__ = [
    "MultiComponentSystem",
    "create_galileo_system",
]
