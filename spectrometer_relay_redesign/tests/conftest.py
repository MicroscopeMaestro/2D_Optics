"""
Pytest fixtures and optical analysis helper utilities for Spectrometer Relay Redesign
and NIR (900-1700 nm) Layout Verification.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pytest

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray
from v2.analysis.spot import SpotDiagram, calculate_spot_metrics


# ==============================================================================
# CONSTANTS & DEFAULT PATHS
# ==============================================================================

PROJECT_DIR = REPO_ROOT / "spectrometer_relay_redesign"
PRIMARY_RELAY_JSON_PATH = PROJECT_DIR / "spectrometer_primary_relay_redesign.json"
NIR_LAYOUT_JSON_PATH = PROJECT_DIR / "spectrometer_nir_900_1700nm.json"
VERIFY_SCRIPT_PATH = PROJECT_DIR / "verify_optical_prescriptions.py"

NIR_DESIGN_WAVELENGTHS = [0.900, 1.100, 1.225, 1.350, 1.700]  # in micrometers (µm)
PRIMARY_RELAY_WAVELENGTHS = [1.100, 1.225, 1.350]             # in micrometers (µm)

DETECTOR_RADIUS_MAX_MM = 0.4000  # 0.8 mm diameter -> 0.4 mm radius
NA_NOMINAL = 0.12                # Entrance Slit NA = 0.12 (half-angle ~6.892 deg)


# ==============================================================================
# HELPER FUNCTIONS: RAY GENERATION & ANALYSIS
# ==============================================================================

def create_chief_ray(wavelength_um: float = 1.225, origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Ray:
    """Create on-axis chief ray along optical axis (+Z)."""
    return Ray(
        pos=np.array(origin, dtype=float),
        dir=np.array([0.0, 0.0, 1.0], dtype=float),
        wavelength=wavelength_um
    )


def create_marginal_rays(
    na: float = NA_NOMINAL,
    wavelength_um: float = 1.225,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> List[Ray]:
    """
    Generate extreme marginal rays at the numerical aperture boundary (sin(theta) = NA).
    Generates 4 cardinal boundary rays (+X, -X, +Y, -Y) and 4 diagonal boundary rays.
    """
    rays = []
    sin_theta = float(np.clip(na, 0.0, 0.999))
    cos_theta = math.sqrt(1.0 - sin_theta**2)

    # 4 Cardinal directions: +X, -X, +Y, -Y
    cardinal_dirs = [
        np.array([sin_theta, 0.0, cos_theta], dtype=float),
        np.array([-sin_theta, 0.0, cos_theta], dtype=float),
        np.array([0.0, sin_theta, cos_theta], dtype=float),
        np.array([0.0, -sin_theta, cos_theta], dtype=float),
    ]

    for d in cardinal_dirs:
        d_norm = d / np.linalg.norm(d)
        rays.append(Ray(pos=np.array(origin, dtype=float), dir=d_norm, wavelength=wavelength_um))

    # 4 Diagonal directions
    diag_component = sin_theta / math.sqrt(2.0)
    diag_dirs = [
        np.array([diag_component, diag_component, cos_theta], dtype=float),
        np.array([-diag_component, diag_component, cos_theta], dtype=float),
        np.array([diag_component, -diag_component, cos_theta], dtype=float),
        np.array([-diag_component, -diag_component, cos_theta], dtype=float),
    ]

    for d in diag_dirs:
        d_norm = d / np.linalg.norm(d)
        rays.append(Ray(pos=np.array(origin, dtype=float), dir=d_norm, wavelength=wavelength_um))

    return rays


def create_pupil_grid(
    na: float = NA_NOMINAL,
    num_rings: int = 5,
    rays_per_ring: int = 8,
    wavelength_um: float = 1.225,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> List[Ray]:
    """
    Generate polar pupil ray grid inside numerical aperture cone sin(theta) <= NA.
    """
    rays = [create_chief_ray(wavelength_um=wavelength_um, origin=origin)]
    max_theta = math.asin(min(1.0, max(0.0, na)))

    for ring_idx in range(1, num_rings + 1):
        frac = ring_idx / float(num_rings)
        theta = frac * max_theta
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        n_pts = int(rays_per_ring * ring_idx)

        for p in range(n_pts):
            phi = 2.0 * math.pi * p / n_pts
            dx = sin_t * math.cos(phi)
            dy = sin_t * math.sin(phi)
            dz = cos_t
            d_vec = np.array([dx, dy, dz], dtype=float)
            d_vec /= np.linalg.norm(d_vec)
            rays.append(Ray(pos=np.array(origin, dtype=float), dir=d_vec, wavelength=wavelength_um))

    return rays


def create_hexapolar_pupil_grid(
    na: float = NA_NOMINAL,
    num_rings: int = 8,
    wavelength_um: float = 1.225,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> List[Ray]:
    """
    Generate dense hexapolar pupil ray grid inside NA cone (yields 1 + 3*N*(N+1) rays).
    For num_rings=8, yields 217 rays.
    """
    rays = [create_chief_ray(wavelength_um=wavelength_um, origin=origin)]
    max_theta = math.asin(min(1.0, max(0.0, na)))

    for ring_idx in range(1, num_rings + 1):
        frac = ring_idx / float(num_rings)
        theta = frac * max_theta
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        n_pts = 6 * ring_idx

        for p in range(n_pts):
            phi = 2.0 * math.pi * p / n_pts
            dx = sin_t * math.cos(phi)
            dy = sin_t * math.sin(phi)
            dz = cos_t
            d_vec = np.array([dx, dy, dz], dtype=float)
            d_vec /= np.linalg.norm(d_vec)
            rays.append(Ray(pos=np.array(origin, dtype=float), dir=d_vec, wavelength=wavelength_um))

    return rays


def create_ray_fan(
    na: float = NA_NOMINAL,
    num_rays: int = 21,
    plane: str = "tangential",
    wavelength_um: float = 1.225,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> List[Ray]:
    """
    Generate 1D ray fan in tangential (YZ) or sagittal (XZ) plane.
    """
    rays = []
    max_theta = math.asin(min(1.0, max(0.0, na)))
    pupil_coords = np.linspace(-1.0, 1.0, num_rays)

    for py in pupil_coords:
        theta = py * max_theta
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)

        if plane.lower() in ("tangential", "yz", "y"):
            d_vec = np.array([0.0, sin_t, cos_t], dtype=float)
        else:
            d_vec = np.array([sin_t, 0.0, cos_t], dtype=float)

        d_vec /= np.linalg.norm(d_vec)
        rays.append(Ray(pos=np.array(origin, dtype=float), dir=d_vec, wavelength=wavelength_um))

    return rays


def compute_detector_spot_metrics(
    system: OpticalSystem,
    rays: List[Ray],
    target_surface_idx: int = -1
) -> Dict[str, Any]:
    """
    Trace rays through OpticalSystem and compute detector hit metrics:
    - N_launched: Total rays launched
    - N_passed: Total unvignetted rays reaching detector
    - N_vignetted: Total vignetted rays
    - transmission: N_passed / N_launched
    - x_hits, y_hits: Hit coordinates on detector plane
    - r_max: Maximum radius sqrt(x^2 + y^2)
    - rms_radius: RMS spot radius relative to centroid
    - geo_radius: 100% geometric spot radius relative to centroid
    - centroid_x, centroid_y: Spot centroid
    - all_contained_0p4mm: True if r_max <= 0.4000 mm
    """
    traced_rays = system.trace_rays(rays)
    n_launched = len(rays)
    x_hits: List[float] = []
    y_hits: List[float] = []
    radii: List[float] = []
    n_vignetted = 0

    for r in traced_rays:
        if r.vignetted:
            n_vignetted += 1
            continue

        if len(r.history) > 0:
            idx = target_surface_idx if target_surface_idx != -1 else len(r.history) - 1
            if idx < len(r.history):
                hit = r.history[idx]
                xh = float(hit["x"])
                yh = float(hit["y"])
                x_hits.append(xh)
                y_hits.append(yh)
                radii.append(math.sqrt(xh**2 + yh**2))

    n_passed = len(x_hits)
    transmission = (n_passed / n_launched) if n_launched > 0 else 0.0
    r_max = max(radii) if radii else 0.0

    if x_hits:
        spot = SpotDiagram(x_hits, y_hits, wavelength_um=rays[0].wavelength if rays else 1.225)
        cx, cy = spot.calculate_centroid()
        rms_r = spot.calculate_rms_radius()
        geo_r = spot.calculate_geo_radius()
    else:
        cx, cy, rms_r, geo_r = 0.0, 0.0, 0.0, 0.0

    return {
        "n_launched": n_launched,
        "n_passed": n_passed,
        "n_vignetted": n_vignetted,
        "transmission": transmission,
        "x_hits": np.array(x_hits, dtype=float),
        "y_hits": np.array(y_hits, dtype=float),
        "radii": np.array(radii, dtype=float),
        "r_max": r_max,
        "rms_radius": rms_r,
        "geo_radius": geo_r,
        "centroid_x": cx,
        "centroid_y": cy,
        "all_contained_0p4mm": (r_max <= DETECTOR_RADIUS_MAX_MM and n_passed > 0 and n_vignetted == 0),
        "traced_rays": traced_rays
    }


# ==============================================================================
# PYTEST FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return repository root path."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def primary_layout_path() -> Path:
    """Return path to primary spectrometer relay redesign JSON layout."""
    env_p = os.environ.get("SPECTROMETER_PRIMARY_RELAY_PATH")
    if env_p and Path(env_p).is_file():
        return Path(env_p)
    return PRIMARY_RELAY_JSON_PATH


@pytest.fixture(scope="session")
def nir_layout_path() -> Path:
    """Return path to NIR 900-1700nm spectrometer JSON layout."""
    env_p = os.environ.get("SPECTROMETER_NIR_LAYOUT_PATH")
    if env_p and Path(env_p).is_file():
        return Path(env_p)
    return NIR_LAYOUT_JSON_PATH


@pytest.fixture(scope="session")
def verify_script_path() -> Path:
    """Return path to mathematical verification script."""
    return VERIFY_SCRIPT_PATH


@pytest.fixture(scope="session")
def nir_wavelengths() -> List[float]:
    """Return list of standard NIR design wavelengths in micrometers."""
    return NIR_DESIGN_WAVELENGTHS


@pytest.fixture(scope="session")
def primary_wavelengths() -> List[float]:
    """Return list of standard primary relay design wavelengths in micrometers."""
    return PRIMARY_RELAY_WAVELENGTHS


@pytest.fixture
def primary_layout_json(primary_layout_path: Path) -> Dict[str, Any]:
    """Load and return raw dictionary from primary relay redesign JSON layout."""
    if not primary_layout_path.is_file():
        pytest.skip(f"Primary relay layout JSON not found at {primary_layout_path} (Milestone M1 pending)")
    with open(primary_layout_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def nir_layout_json(nir_layout_path: Path) -> Dict[str, Any]:
    """Load and return raw dictionary from NIR 900-1700nm JSON layout."""
    if not nir_layout_path.is_file():
        pytest.skip(f"NIR layout JSON not found at {nir_layout_path} (Milestone M2 pending)")
    with open(nir_layout_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def primary_optical_system(primary_layout_path: Path) -> OpticalSystem:
    """Load OpticalSystem from primary relay redesign JSON layout."""
    if not primary_layout_path.is_file():
        pytest.skip(f"Primary relay layout JSON not found at {primary_layout_path} (Milestone M1 pending)")
    return OpticalSystem.from_json(str(primary_layout_path))


@pytest.fixture
def nir_optical_system(nir_layout_path: Path) -> OpticalSystem:
    """Load OpticalSystem from NIR 900-1700nm JSON layout."""
    if not nir_layout_path.is_file():
        pytest.skip(f"NIR layout JSON not found at {nir_layout_path} (Milestone M2 pending)")
    return OpticalSystem.from_json(str(nir_layout_path))


@pytest.fixture
def pupil_grid_factory():
    """Fixture returning create_pupil_grid helper function."""
    return create_pupil_grid


@pytest.fixture
def hexapolar_grid_factory():
    """Fixture returning create_hexapolar_pupil_grid helper function."""
    return create_hexapolar_pupil_grid


@pytest.fixture
def marginal_rays_factory():
    """Fixture returning create_marginal_rays helper function."""
    return create_marginal_rays


@pytest.fixture
def ray_fan_factory():
    """Fixture returning create_ray_fan helper function."""
    return create_ray_fan


@pytest.fixture
def spot_analyzer():
    """Fixture returning compute_detector_spot_metrics helper function."""
    return compute_detector_spot_metrics
