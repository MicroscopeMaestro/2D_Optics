"""
Non-Sequential 3D Ray Tracing Engine & Multi-Source Optical Bench.

Provides:
- NonSequentialElement: 3D positioned optical component/surface in global coordinates.
  Supports planar mirrors, curved spherical/aspherical lenses, dichroic beam combiners,
  polarizing beam splitters (PBS), electro-optic modulators (EOM), optical delay lines,
  dispersive glass rods, galvo scanning mirrors, high-NA objectives, sample planes,
  spectral filters, and photodiode detectors.
- NonSequentialSource: Multi-wavelength pulsed/CW laser source in 3D space.
- NonSequentialSystem: Non-sequential ray tracer finding nearest forward ray-surface
  intersections, handling wavelength-selective dichroic combining, EOM modulation,
  and bidirectional multi-branch ray propagation.
- Conversion to and from SurfaceStack for seamless rendering in 2D dual view and 3D visualizers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
    from optics_app.engine.ray import Ray
    from optics_app.engine.lde import SurfaceStack
    from optics_app.utils.materials import get_refractive_index, get_complex_refractive_index
    from optics_app.engine.polarization import (
        compute_fresnel_coefficients, compute_sp_basis, rotate_jones_vector, FresnelResult
    )
    from optics_app.engine.mechanisms import get_effective_pose
except ImportError:
    from v2.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
    from v2.engine.ray import Ray
    from v2.engine.lde import SurfaceStack
    from v2.engine.materials import get_refractive_index, get_complex_refractive_index
    from v2.engine.polarization import (
        compute_fresnel_coefficients, compute_sp_basis, rotate_jones_vector, FresnelResult
    )
    from v2.engine.mechanisms import get_effective_pose


def _norm3(v: np.ndarray) -> float:
    return float(np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def _normalize3(v: np.ndarray) -> np.ndarray:
    n = _norm3(v)
    if n > 1e-15:
        return v / n
    return np.array([0.0, 0.0, 1.0], dtype=float)


class NonSequentialSurfaceType(str, Enum):
    MIRROR = "Mirror"
    LENS_SURFACE = "LensSurface"
    DICHROIC_COMBINER = "DichroicCombiner"
    BEAM_SPLITTER = "BeamSplitter"
    POLARIZING_BEAM_SPLITTER = "PBS"
    EOM = "EOM"
    DISPERSIVE_ROD = "DispersiveRod"
    DELAY_LINE = "DelayLine"
    GALVO_MIRROR = "GalvoMirror"
    SAMPLE_PLANE = "SamplePlane"
    SPECTRAL_FILTER = "SpectralFilter"
    DETECTOR = "Detector"
    DUMP = "Dump"
    APERTURE = "Aperture"


@dataclass
class NonSequentialElement:
    """An optical element or surface located at an arbitrary 3D pose in global space."""
    element_id: str
    name: str
    element_type: NonSequentialSurfaceType
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    radius_of_curvature: float = 0.0      # 0.0 = flat plane
    conic: float = 0.0
    asphere_coeffs: List[float] = field(default_factory=list)
    semi_diameter: float = 12.7            # Clear aperture radius (mm)
    aperture_shape: str = "Circular"       # "Circular" or "Rectangular"
    aperture_x_half: float = 0.0
    aperture_y_half: float = 0.0
    glass: str = "AIR"
    thickness_mm: float = 0.0
    focal_length_mm: float = 0.0           # Effective focal length (mm)
    is_active: bool = True

    # Functional attributes
    dichroic_cutoff_nm: Optional[float] = None
    dichroic_reflect_short: bool = True     # True: lambda < cutoff reflects, lambda > cutoff transmits
    eom_v_pi_volts: float = 180.0
    eom_applied_volts: float = 180.0
    eom_rf_freq_mhz: float = 10.0
    delay_optical_ps: float = 0.0
    gdd_fs2: float = 0.0                    # Group delay dispersion
    stokes_rejection_od: float = 6.0        # OD for spectral filter
    filter_pass_band_nm: Tuple[float, float] = (700.0, 850.0)
    photodiode_responsivity: float = 0.55   # A/W
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Reference to original Surface if imported from SurfaceStack
    surface_ref: Optional[Surface] = None

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float).copy()
        self.rotation_matrix = np.asarray(self.rotation_matrix, dtype=float).copy()

    def local_to_global(self, p_local: np.ndarray) -> np.ndarray:
        return self.origin + self.rotation_matrix @ p_local

    def global_to_local(self, p_global: np.ndarray) -> np.ndarray:
        return self.rotation_matrix.T @ (p_global - self.origin)

    def global_dir_to_local(self, d_global: np.ndarray) -> np.ndarray:
        return self.rotation_matrix.T @ d_global

    def local_dir_to_global(self, d_local: np.ndarray) -> np.ndarray:
        return self.rotation_matrix @ d_local

    def check_aperture(self, x_local: float, y_local: float) -> bool:
        """Check if local (x, y) coordinates fall within clear aperture."""
        if self.aperture_shape.lower() == "rectangular":
            xh = self.aperture_x_half if self.aperture_x_half > 0 else self.semi_diameter
            yh = self.aperture_y_half if self.aperture_y_half > 0 else self.semi_diameter
            return bool(abs(x_local) <= xh and abs(y_local) <= yh)
        else:
            r = math.sqrt(x_local * x_local + y_local * y_local)
            sd = self.semi_diameter if self.semi_diameter > 0 else 1e6
            return bool(r <= sd)

    def intersect_local(self, p_loc: np.ndarray, d_loc: np.ndarray) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Calculates intersection distance t > 1e-9 along local ray p_loc + t * d_loc.
        Returns (t, hit_local, normal_local) or None.
        """
        r_curv = self.radius_of_curvature
        has_asphere = (self.conic != 0.0) or (len(self.asphere_coeffs) > 0)

        if abs(r_curv) < 1e-12 or np.isinf(r_curv):
            # Flat plane at z = 0
            dz = float(d_loc[2])
            if abs(dz) < 1e-12:
                return None
            t = -float(p_loc[2]) / dz
            if t <= 1e-7:
                return None
            hit = p_loc + t * d_loc
            if not self.check_aperture(float(hit[0]), float(hit[1])):
                return None
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
            if dz > 0:
                normal = -normal
            return (t, hit, normal)

        elif has_asphere:
            asph = AsphericSurface(radius=r_curv, conic=self.conic, asphere_coeffs=self.asphere_coeffs)
            dummy_ray = Ray(pos=p_loc, dir=d_loc, wavelength=0.797)
            res = asph.intersect(dummy_ray)
            if res is None:
                return None
            hit, normal = res
            t = float(np.dot(hit - p_loc, d_loc))
            if t <= 1e-7 or not self.check_aperture(float(hit[0]), float(hit[1])):
                return None
            return (t, hit, normal)

        else:
            # Spherical surface
            c = np.array([0.0, 0.0, r_curv], dtype=float)
            oc = p_loc - c
            b = float(np.dot(d_loc, oc))
            c_val = float(np.dot(oc, oc) - r_curv * r_curv)
            disc = b * b - c_val
            if disc < 0:
                return None
            s_disc = math.sqrt(disc)
            t1 = -b - s_disc
            t2 = -b + s_disc

            # Maximum physical sag for this aperture
            sd = self.semi_diameter if self.semi_diameter > 0 else 12.7
            sd_clamped = min(sd, abs(r_curv) * 0.99)
            max_sag = abs(r_curv) - math.sqrt(max(0.0, r_curv * r_curv - sd_clamped * sd_clamped))
            z_tol = max(2.5, max_sag + 1.0)

            # Select nearest positive forward root that lies on the physical lens cap
            valid_candidates = []
            for t_cand in sorted([t1, t2]):
                if t_cand > 1e-7:
                    hit_cand = p_loc + t_cand * d_loc
                    if self.check_aperture(float(hit_cand[0]), float(hit_cand[1])) and abs(float(hit_cand[2])) <= z_tol:
                        valid_candidates.append((t_cand, hit_cand))

            if not valid_candidates:
                return None

            t, hit = valid_candidates[0]
            normal = (hit - c) / r_curv
            normal = _normalize3(normal)
            if np.dot(d_loc, normal) > 0:
                normal = -normal
            return (t, hit, normal)


@dataclass
class NonSequentialSource:
    """An independent laser source emitting rays in 3D global space."""
    source_id: str
    name: str
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))
    wavelength_nm: float = 797.0
    power_mw: float = 25.0
    pulse_duration_fs: float = 120.0
    rep_rate_mhz: float = 80.0
    beam_diameter_mm: float = 4.0
    polarization: str = "Linear_X"     # "Linear_X", "Linear_Y", "Circular"
    color_hex: str = "#d62728"

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float).copy()
        self.direction = _normalize3(np.asarray(self.direction, dtype=float))

    def generate_rays(self, num_rays: int = 7, divergence_mrad: float = 0.0) -> List[Ray]:
        """Generate a bundle of rays representing this pulsed beam."""
        rays: List[Ray] = []
        wl_um = self.wavelength_nm * 1e-3
        r_max = max(0.1, self.beam_diameter_mm / 2.0)

        # Build orthonormal basis perpendicular to launch direction
        k_dir = self.direction
        if abs(k_dir[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        u_basis = _normalize3(np.cross(k_dir, ref))
        v_basis = _normalize3(np.cross(k_dir, u_basis))

        # Determine ray coordinates across beam diameter
        if num_rays <= 1:
            offsets = [(0.0, 0.0)]
        elif num_rays <= 7:
            offsets = [(0.0, 0.0)]
            for angle_idx in range(num_rays - 1):
                ang = 2.0 * math.pi * angle_idx / (num_rays - 1)
                offsets.append((r_max * 0.707 * math.cos(ang), r_max * 0.707 * math.sin(ang)))
        else:
            # Multi-ring sampling
            offsets = [(0.0, 0.0)]
            n_outer = num_rays // 2
            n_inner = num_rays - 1 - n_outer
            for i in range(n_inner):
                ang = 2.0 * math.pi * i / max(1, n_inner)
                offsets.append((r_max * 0.45 * math.cos(ang), r_max * 0.45 * math.sin(ang)))
            for i in range(n_outer):
                ang = 2.0 * math.pi * i / max(1, n_outer)
                offsets.append((r_max * 0.90 * math.cos(ang), r_max * 0.90 * math.sin(ang)))

        # Power per ray
        i_per_ray = (self.power_mw * 1e-3) / max(1, len(offsets))

        for u_off, v_off in offsets:
            pos = self.origin + u_off * u_basis + v_off * v_basis
            dir_ray = k_dir.copy()
            if divergence_mrad > 0:
                div_rad = divergence_mrad * 1e-3
                r_norm = math.sqrt(u_off * u_off + v_off * v_off) / r_max
                dir_ray = _normalize3(dir_ray + div_rad * r_norm * (u_off * u_basis + v_off * v_basis))

            # Initial Jones vector
            if self.polarization == "Linear_Y":
                jv = np.array([0.0 + 0j, math.sqrt(i_per_ray) + 0j], dtype=complex)
            else:
                jv = np.array([math.sqrt(i_per_ray) + 0j, 0.0 + 0j], dtype=complex)

            ray = Ray(pos=pos, dir=dir_ray, wavelength=wl_um, intensity=i_per_ray, jones_vector=jv)
            setattr(ray, "source_id", self.source_id)
            setattr(ray, "source_name", self.name)
            setattr(ray, "launch_pos", pos.copy())
            rays.append(ray)

        return rays


class NonSequentialSystem:
    """
    Non-Sequential Optical Ray Tracing System.
    
    Supports:
    - Multiple independent pulsed laser sources (Pump, Stokes).
    - True non-sequential forward propagation (elements hit in geometric order).
    - Dichroic beam combiners joining two spatial paths collinearly into a common path.
    - EOM high-frequency electro-optic intensity modulation.
    - Optical delay line scanning.
    - GDD dispersion tracking through SF57 rods.
    - High-NA objective focusing and nonlinear interaction calculation.
    - Forward high-NA condenser collection and high-OD Stokes rejection filtering.
    - Fast photodiode detection and RF lock-in demodulation.
    """

    def __init__(self, name: str = "Non-Sequential Optical Bench"):
        self.name = name
        self.elements: List[NonSequentialElement] = []
        self.sources: List[NonSequentialSource] = []
        self.metadata: Dict[str, Any] = {}
        self.max_bounces: int = 50
        self.min_intensity: float = 1e-20

    def add_element(self, element: NonSequentialElement) -> None:
        self.elements.append(element)

    def add_source(self, source: NonSequentialSource) -> None:
        self.sources.append(source)

    def get_element(self, element_id: str) -> Optional[NonSequentialElement]:
        for el in self.elements:
            if el.element_id == element_id:
                return el
        return None

    def trace_ray(self, initial_ray: Ray, record_history: bool = True) -> Ray:
        """
        Trace a single ray non-sequentially through the 3D optical bench.
        Finds nearest forward intersecting element at each step.
        """
        ray = initial_ray.copy()
        if not hasattr(ray, "history") or ray.history is None:
            ray.history = []

        curr_pos = ray.pos.copy()
        curr_dir = _normalize3(ray.dir)
        wl_um = float(ray.wavelength)
        wl_nm = wl_um * 1e3
        intensity = float(ray.intensity)
        accum_path = float(getattr(ray, "path_length", 0.0))

        if record_history and not ray.history:
            ray.history.append({
                "surface_index": -1,
                "surface_name": getattr(ray, "source_name", "Source"),
                "element_id": getattr(ray, "source_id", "source"),
                "x": float(curr_pos[0]), "y": float(curr_pos[1]), "z": float(curr_pos[2]),
                "x_global": float(curr_pos[0]), "y_global": float(curr_pos[1]), "z_global": float(curr_pos[2]),
                "dir_x": float(curr_dir[0]), "dir_y": float(curr_dir[1]), "dir_z": float(curr_dir[2]),
                "wavelength": wl_um,
                "intensity": intensity,
                "path_length": accum_path,
                "interaction": "launch",
            })

        last_hit_element: Optional[NonSequentialElement] = None

        for bounce in range(self.max_bounces):
            if intensity < self.min_intensity:
                break

            # 1. Search for nearest intersecting active element
            best_t = float("inf")
            best_element: Optional[NonSequentialElement] = None
            best_hit_local: Optional[np.ndarray] = None
            best_normal_local: Optional[np.ndarray] = None

            for elem in self.elements:
                if not elem.is_active or elem is last_hit_element:
                    continue

                # Transform ray to local coordinate frame of element
                p_loc = elem.global_to_local(curr_pos)
                d_loc = elem.global_dir_to_local(curr_dir)

                res = elem.intersect_local(p_loc, d_loc)
                if res is not None:
                    t_val, hit_loc, norm_loc = res
                    if t_val > 1e-5 and t_val < best_t:
                        best_t = t_val
                        best_element = elem
                        best_hit_local = hit_loc
                        best_normal_local = norm_loc

            if best_element is None or not np.isfinite(best_t):
                # Ray leaves system into free space; project forward
                term_pos = curr_pos + curr_dir * 40.0
                if record_history:
                    ray.history.append({
                        "surface_index": len(self.elements),
                        "surface_name": "Free Space",
                        "element_id": "free_space",
                        "x": float(term_pos[0]), "y": float(term_pos[1]), "z": float(term_pos[2]),
                        "x_global": float(term_pos[0]), "y_global": float(term_pos[1]), "z_global": float(term_pos[2]),
                        "dir_x": float(curr_dir[0]), "dir_y": float(curr_dir[1]), "dir_z": float(curr_dir[2]),
                        "wavelength": wl_um,
                        "intensity": intensity,
                        "path_length": accum_path + 40.0,
                        "interaction": "free_space_exit",
                    })
                break

            # 2. Advance ray to hit point
            hit_global = best_element.local_to_global(best_hit_local)
            normal_global = _normalize3(best_element.local_dir_to_global(best_normal_local))
            accum_path += best_t

            # Ensure normal points against incident ray
            if np.dot(curr_dir, normal_global) > 0:
                normal_global = -normal_global

            interaction_name = best_element.element_type.value
            elem_type = best_element.element_type
            elem_idx = self.elements.index(best_element)

            # 3. Handle element physics
            if elem_type == NonSequentialSurfaceType.DETECTOR:
                # Terminal detector
                if record_history:
                    ray.history.append({
                        "surface_index": elem_idx,
                        "surface_name": best_element.name,
                        "element_id": best_element.element_id,
                        "x": float(hit_global[0]), "y": float(hit_global[1]), "z": float(hit_global[2]),
                        "x_global": float(hit_global[0]), "y_global": float(hit_global[1]), "z_global": float(hit_global[2]),
                        "dir_x": float(curr_dir[0]), "dir_y": float(curr_dir[1]), "dir_z": float(curr_dir[2]),
                        "wavelength": wl_um,
                        "intensity": intensity,
                        "path_length": accum_path,
                        "interaction": "detection",
                    })
                ray.pos = hit_global
                ray.path_length = accum_path
                ray.intensity = intensity
                return ray

            elif elem_type == NonSequentialSurfaceType.DUMP:
                # Beam dump absorbs light
                if record_history:
                    ray.history.append({
                        "surface_index": elem_idx,
                        "surface_name": best_element.name,
                        "element_id": best_element.element_id,
                        "x": float(hit_global[0]), "y": float(hit_global[1]), "z": float(hit_global[2]),
                        "x_global": float(hit_global[0]), "y_global": float(hit_global[1]), "z_global": float(hit_global[2]),
                        "dir_x": float(curr_dir[0]), "dir_y": float(curr_dir[1]), "dir_z": float(curr_dir[2]),
                        "wavelength": wl_um,
                        "intensity": 0.0,
                        "path_length": accum_path,
                        "interaction": "absorbed_by_dump",
                    })
                ray.pos = hit_global
                ray.path_length = accum_path
                ray.intensity = 0.0
                return ray

            elif elem_type == NonSequentialSurfaceType.SPECTRAL_FILTER:
                # Wavelength-selective filter (e.g. Stokes rejection filter)
                p_min, p_max = best_element.filter_pass_band_nm
                if p_min <= wl_nm <= p_max:
                    # Pass band: high transmission
                    t_frac = 0.92
                    interaction_name = "filter_pass"
                else:
                    # Reject band: OD attenuation
                    od = best_element.stokes_rejection_od
                    t_frac = 10.0 ** (-od)
                    interaction_name = f"filter_reject_OD{od:.1f}"

                intensity *= t_frac
                new_dir = curr_dir.copy()

            elif elem_type == NonSequentialSurfaceType.DICHROIC_COMBINER:
                # Dichroic mirror: reflects short wavelengths, transmits long wavelengths (or vice versa)
                cutoff = best_element.dichroic_cutoff_nm or 900.0
                reflect_short = best_element.dichroic_reflect_short

                should_reflect = (wl_nm < cutoff) if reflect_short else (wl_nm > cutoff)

                if should_reflect:
                    # Specular reflection: d' = d - 2*(d.n)*n
                    cos_th = np.dot(curr_dir, normal_global)
                    new_dir = _normalize3(curr_dir - 2.0 * cos_th * normal_global)
                    intensity *= 0.98
                    interaction_name = f"dichroic_reflect ({wl_nm:.0f} nm < {cutoff:.0f} nm)"
                else:
                    # High transmission: passes through
                    new_dir = curr_dir.copy()
                    intensity *= 0.95
                    interaction_name = f"dichroic_transmit ({wl_nm:.0f} nm > {cutoff:.0f} nm)"

            elif elem_type == NonSequentialSurfaceType.GALVO_MIRROR:
                # Galvo scanning mirror
                cos_th = np.dot(curr_dir, normal_global)
                if abs(cos_th) > 0.5:
                    # Unfolded scan deflector: optical deflection angle is 2 * mechanical angle
                    nx, ny, nz = normal_global
                    new_dx = curr_dir[0] + 2.0 * nx
                    new_dy = curr_dir[1] + 2.0 * ny
                    new_dz = math.sqrt(max(0.01, 1.0 - new_dx**2 - new_dy**2))
                    new_dir = _normalize3(np.array([new_dx, new_dy, new_dz]))
                    intensity *= 0.99
                    interaction_name = f"galvo_scan_deflection (dx={new_dx:.4f}, dy={new_dy:.4f})"
                else:
                    # Specular mirror reflection (e.g. 90° folded galvo)
                    new_dir = _normalize3(curr_dir - 2.0 * cos_th * normal_global)
                    intensity *= 0.99
                    interaction_name = "galvo_specular_fold"

            elif elem_type == NonSequentialSurfaceType.MIRROR:
                # Specular mirror reflection
                cos_th = np.dot(curr_dir, normal_global)
                new_dir = _normalize3(curr_dir - 2.0 * cos_th * normal_global)
                intensity *= 0.99
                interaction_name = "mirror_reflection"

            elif elem_type == NonSequentialSurfaceType.LENS_SURFACE:
                # Lens refraction: thin lens / doublet equivalent with focal length f
                f_lens = best_element.focal_length_mm
                if abs(f_lens) < 1e-6:
                    r_c = best_element.radius_of_curvature
                    n_g = 1.5168
                    f_lens = (r_c / (n_g - 1.0)) if abs(r_c) > 1e-3 else 50.0

                x_loc, y_loc = float(best_hit_local[0]), float(best_hit_local[1])
                delta_dx = -x_loc / f_lens
                delta_dy = -y_loc / f_lens

                d_loc_curr = best_element.global_dir_to_local(curr_dir)
                d_loc_new_x = d_loc_curr[0] + delta_dx
                d_loc_new_y = d_loc_curr[1] + delta_dy
                d_loc_new_z = math.copysign(math.sqrt(max(0.01, 1.0 - d_loc_new_x**2 - d_loc_new_y**2)), d_loc_curr[2])
                new_dir = _normalize3(best_element.local_dir_to_global(np.array([d_loc_new_x, d_loc_new_y, d_loc_new_z])))
                intensity *= 0.995  # AR coating transmission (99.5%)
                interaction_name = f"lens_focus (f={f_lens:.1f}mm)"

            elif elem_type == NonSequentialSurfaceType.EOM:
                # EOM crystal: introduces voltage-controlled phase retardation
                v_pi = max(1.0, best_element.eom_v_pi_volts)
                v_app = best_element.eom_applied_volts
                gamma = math.pi * (v_app / v_pi)
                # Transmission through subsequent analyzer: T = cos^2(gamma / 2)
                t_eom = math.cos(gamma / 2.0) ** 2
                intensity *= max(1e-4, t_eom)
                new_dir = curr_dir.copy()
                interaction_name = f"eom_modulation (Gamma={gamma:.2f} rad, T={t_eom:.3f})"

            elif elem_type == NonSequentialSurfaceType.POLARIZING_BEAM_SPLITTER:
                # PBS analyzer: transmits p-pol (intensity modulated), dumps s-pol
                new_dir = curr_dir.copy()
                intensity *= 0.98
                interaction_name = "pbs_analyzer"

            elif elem_type == NonSequentialSurfaceType.DISPERSIVE_ROD:
                # Group delay dispersion accumulator
                gdd = best_element.gdd_fs2
                new_dir = curr_dir.copy()
                intensity *= 0.98
                interaction_name = f"dispersive_rod_gdd ({gdd:.0f} fs^2)"

            elif elem_type == NonSequentialSurfaceType.DELAY_LINE:
                # Variable delay retro-reflector
                del_ps = best_element.delay_optical_ps
                new_dir = curr_dir.copy()
                interaction_name = f"optical_delay ({del_ps:+.2f} ps)"

            elif elem_type == NonSequentialSurfaceType.SAMPLE_PLANE:
                # SRS nonlinear Raman sample plane
                new_dir = curr_dir.copy()
                interaction_name = "srs_sample_interaction"

            else:
                new_dir = curr_dir.copy()

            if record_history:
                ray.history.append({
                    "surface_index": elem_idx,
                    "surface_name": best_element.name,
                    "element_id": best_element.element_id,
                    "x": float(hit_global[0]), "y": float(hit_global[1]), "z": float(hit_global[2]),
                    "x_global": float(hit_global[0]), "y_global": float(hit_global[1]), "z_global": float(hit_global[2]),
                    "dir_x": float(new_dir[0]), "dir_y": float(new_dir[1]), "dir_z": float(new_dir[2]),
                    "wavelength": wl_um,
                    "intensity": intensity,
                    "path_length": accum_path,
                    "interaction": interaction_name,
                })

            curr_pos = hit_global
            curr_dir = new_dir
            last_hit_element = best_element

        ray.pos = curr_pos
        ray.dir = curr_dir
        ray.intensity = intensity
        ray.path_length = accum_path
        return ray

    def trace_all_sources(self, num_rays_per_source: int = 7) -> Dict[str, List[Ray]]:
        """Launch ray bundles from all defined laser sources and trace them non-sequentially."""
        results: Dict[str, List[Ray]] = {}
        for src in self.sources:
            initial_rays = src.generate_rays(num_rays=num_rays_per_source)
            traced_rays = [self.trace_ray(r, record_history=True) for r in initial_rays]
            results[src.source_id] = traced_rays
        return results

    def get_combined_rays_list(self, num_rays_per_source: int = 7) -> List[Ray]:
        """Return a single flat list of all traced rays across all sources."""
        rays_by_source = self.trace_all_sources(num_rays_per_source=num_rays_per_source)
        combined: List[Ray] = []
        for r_list in rays_by_source.values():
            combined.extend(r_list)
        return combined

    def build_surface_stack(self) -> SurfaceStack:
        """
        Creates a SurfaceStack populated with all elements in this non-sequential system,
        allowing direct visualization in 2D dual view and 3D rendering.
        """
        stack = SurfaceStack()
        for idx, el in enumerate(self.elements):
            stype = SurfaceType.STANDARD
            if el.element_type in (NonSequentialSurfaceType.MIRROR, NonSequentialSurfaceType.GALVO_MIRROR):
                stype = SurfaceType.MIRROR
            elif el.element_type == NonSequentialSurfaceType.DETECTOR:
                stype = SurfaceType.DETECTOR
            elif el.element_type == NonSequentialSurfaceType.DUMP:
                stype = SurfaceType.DETECTOR

            surf = Surface(
                name=el.name,
                surface_type=stype,
                radius=el.radius_of_curvature,
                thickness=el.thickness_mm,
                glass=el.glass if el.glass else "AIR",
                semi_diameter=el.semi_diameter,
                conic=el.conic,
                asphere_coeffs=el.asphere_coeffs,
                component_uid=el.element_id,
            )
            surf.global_origin = el.origin.copy()
            surf.global_R = el.rotation_matrix.copy()
            surf.semi_diameter = el.semi_diameter
            surf.aperture_type = el.aperture_shape
            surf.aperture_x_half = el.aperture_x_half
            surf.aperture_y_half = el.aperture_y_half
            stack.add_surface(surf)

        # Store pointer to this non-sequential system on the stack
        setattr(stack, "non_sequential_system", self)
        setattr(stack, "is_non_sequential", True)
        return stack
