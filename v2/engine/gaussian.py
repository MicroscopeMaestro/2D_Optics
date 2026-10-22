"""
Gaussian Beam Propagation Engine via Complex q-Parameter / ABCD Matrix Methods.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class GaussianBeamNode:
    surface_index: int
    surface_name: str
    z_position: float
    q: complex
    w: float
    R: float
    n: float


@dataclass
class GaussianBeamTraceResult:
    w0: float
    wavelength: float
    nodes: List[GaussianBeamNode]
    final_w: float
    final_R: float


class GaussianBeam:
    def __init__(self, w0: float, wavelength: float = 0.5876, n: float = 1.0, z: float = 0.0):
        self.w0 = float(w0)
        self.wavelength = float(wavelength)  # in mm (or um converted to mm if needed, default um)
        # Convert wavelength to mm if > 0.01 (assuming um input, e.g. 0.5876 um = 0.0005876 mm)
        self.lambda_mm = self.wavelength * 1e-3 if self.wavelength > 0.01 else self.wavelength
        self.n = float(n)
        self.z = float(z)

        # Rayleigh range zR = pi * n * w0^2 / lambda
        self.z_R = (np.pi * self.n * (self.w0**2)) / self.lambda_mm

        # Initial q at waist z=0: q0 = i * zR
        self.q = complex(z, self.z_R)

    @property
    def spot_radius(self) -> float:
        """Compute spot radius w(z) from complex q: 1/q = 1/R - i * lambda / (pi * n * w^2)."""
        inv_q = 1.0 / self.q
        im_inv_q = inv_q.imag
        if im_inv_q >= 0:
            # Numerical fallback
            return self.w0
        val = -self.lambda_mm / (np.pi * self.n * im_inv_q)
        return float(np.sqrt(np.maximum(1e-15, val)))

    @property
    def radius_of_curvature(self) -> float:
        """Compute wavefront radius of curvature R(z) from complex q: 1/R = Re(1/q)."""
        inv_q = 1.0 / self.q
        re_inv_q = inv_q.real
        if abs(re_inv_q) < 1e-12:
            return float('inf')
        return float(1.0 / re_inv_q)

    @property
    def divergence_half_angle(self) -> float:
        """Divergence half angle theta = lambda / (pi * n * w0)."""
        return float(self.lambda_mm / (np.pi * self.n * self.w0))

    def propagate_distance(self, d: float) -> None:
        """Free space propagation by distance d: q_2 = q_1 + d."""
        self.q = self.q + float(d)
        self.z += float(d)

    def apply_abcd(self, A: float, B: float, C: float, D: float, n_next: Optional[float] = None) -> None:
        """Apply 2x2 ABCD matrix transformation to complex q: q_2 = (A*q_1 + B) / (C*q_1 + D)."""
        denom = C * self.q + D
        if abs(denom) < 1e-15:
            denom = 1e-15
        self.q = (A * self.q + B) / denom
        if n_next is not None and n_next > 0:
            self.n = float(n_next)

    def refract_surface(self, radius: float, n1: float, n2: float) -> None:
        """Refraction across a curved interface with radius R from index n1 to n2."""
        if radius == 0.0 or np.isinf(radius):
            # Flat interface: A=1, B=0, C=0, D=n1/n2 => q2 = (n2/n1)*q1
            A, B, C, D = 1.0, 0.0, 0.0, n1 / n2
        else:
            C = -(n2 - n1) / (n2 * radius)
            D = n1 / n2
            A, B = 1.0, 0.0
        self.apply_abcd(A, B, C, D, n_next=n2)

    def reflect_mirror(self, radius: float) -> None:
        """Reflection off a curved mirror with radius of curvature R."""
        if radius == 0.0 or np.isinf(radius):
            A, B, C, D = 1.0, 0.0, 0.0, 1.0
        else:
            A, B = 1.0, 0.0
            C = -2.0 / radius
            D = 1.0
        self.apply_abcd(A, B, C, D)

    def pass_thin_lens(self, f: float) -> None:
        """Pass through a thin lens of focal length f."""
        if f == 0.0 or np.isinf(f):
            A, B, C, D = 1.0, 0.0, 0.0, 1.0
        else:
            A, B = 1.0, 0.0
            C = -1.0 / f
            D = 1.0
        self.apply_abcd(A, B, C, D)


def trace_system_gaussian_beam(
    surface_stack: Any,
    w0: float,
    wavelength: float = 0.5876,
    source_z: float = 0.0
) -> GaussianBeamTraceResult:
    """
    Sequential Gaussian beam trace through an OpticalSystem / SurfaceStack.
    """
    from optics_app.utils.materials import get_refractive_index

    surfaces = surface_stack.get_active_sequence() if hasattr(surface_stack, 'get_active_sequence') else surface_stack.surfaces

    beam = GaussianBeam(w0=w0, wavelength=wavelength, n=1.0, z=source_z)

    nodes: List[GaussianBeamNode] = []
    current_z = source_z
    current_n = 1.0

    # Initial node at source
    nodes.append(GaussianBeamNode(
        surface_index=0,
        surface_name=surfaces[0].name if surfaces else "Source",
        z_position=current_z,
        q=beam.q,
        w=beam.spot_radius,
        R=beam.radius_of_curvature,
        n=current_n
    ))

    for i, surf in enumerate(surfaces):
        if i == 0:
            continue

        prev_surf = surfaces[i-1]
        dist = float(prev_surf.thickness)

        if dist > 0:
            beam.propagate_distance(dist)
            current_z += dist

        next_n = get_refractive_index(surf.glass, wavelength) if surf.glass else 1.0
        is_mirror = (surf.glass and surf.glass.upper() in ["MIRROR", "MIRR"]) or ("MIRROR" in surf.name.upper()) or (surf.surface_type and surf.surface_type.upper() == "MIRROR")

        if is_mirror:
            beam.reflect_mirror(surf.radius)
        elif surf.surface_type not in ("Detector", "Stop", "Object"):
            beam.refract_surface(surf.radius, current_n, next_n)
            current_n = next_n

        nodes.append(GaussianBeamNode(
            surface_index=i,
            surface_name=surf.name,
            z_position=current_z,
            q=beam.q,
            w=beam.spot_radius,
            R=beam.radius_of_curvature,
            n=current_n
        ))

    return GaussianBeamTraceResult(
        w0=w0,
        wavelength=wavelength,
        nodes=nodes,
        final_w=beam.spot_radius,
        final_R=beam.radius_of_curvature
    )
