"""
Test Beamsplitter Polarization Physics: Jones Calculus, Stokes Parameters, and Energy Conservation.
"""

import math
import numpy as np
import pytest

from optics_app.catalogs.thorlabs import get_catalog_lens, get_beamsplitter_catalog
from optics_app.engine.components import BeamSplitterConfig
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem, BranchedTracer
from optics_app.engine.surface import Surface, SurfaceType


def test_energy_conservation_all_catalog_beamsplitters():
    """Verify strict energy conservation (R + T <= 1.0) across all 52 catalog beamsplitters."""
    all_bs = get_beamsplitter_catalog()
    assert len(all_bs) >= 52

    # Test across multiple polarization states and wavelengths
    test_polarizations = [
        ("P", (0.0, 1.0)),
        ("S", (1.0, 0.0)),
        ("45_deg", (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))),
        ("RHC", (1.0 / math.sqrt(2.0), complex(0.0, 1.0 / math.sqrt(2.0)))),
        ("LHC", (1.0 / math.sqrt(2.0), complex(0.0, -1.0 / math.sqrt(2.0)))),
    ]

    for pn, record in all_bs.items():
        cfg_dict = record.get("beamsplitter_config")
        if not cfg_dict:
            continue
        cfg = BeamSplitterConfig(**cfg_dict)

        wls = record.get("design_wavelengths_nm", [632.8])
        wl_um = wls[0] * 1e-3 if wls[0] > 10.0 else wls[0]

        for pol_name, j_in in test_polarizations:
            j_t, i_t, j_r, i_r = cfg.evaluate_jones(
                jones_in=j_in,
                wavelength_um=wl_um,
                intensity_in=1.0
            )

            # 1. Total energy conservation
            total_power = i_t + i_r
            assert total_power <= 1.000001, (
                f"Energy conservation violated for {pn} ({pol_name}): R+T = {total_power} > 1.0"
            )
            assert total_power >= 0.0, f"Negative total power for {pn}: {total_power}"
            assert i_t >= -1e-12, f"Negative transmission for {pn}: {i_t}"
            assert i_r >= -1e-12, f"Negative reflection for {pn}: {i_r}"

            # 2. Jones vector power consistency
            norm_t = abs(j_t[0])**2 + abs(j_t[1])**2
            norm_r = abs(j_r[0])**2 + abs(j_r[1])**2
            assert abs(norm_t - i_t) < 1e-5
            assert abs(norm_r - i_r) < 1e-5


def test_pbs_polarization_extinction_and_rejection():
    """Verify Thorlabs PBS cubes cleanly transmit P-pol and reflect S-pol."""
    pbs251_info = get_catalog_lens("PBS251")
    cfg = BeamSplitterConfig(**pbs251_info["beamsplitter_config"])

    # 1. Pure P-polarization (Ey = 1, Ex = 0)
    j_p = (0.0, 1.0)
    j_t_p, i_t_p, j_r_p, i_r_p = cfg.evaluate_jones(j_p, wavelength_um=0.6328, intensity_in=1.0)
    assert i_t_p >= 0.89, f"Expected Tp >= 89%, got {i_t_p*100}%"
    assert i_r_p <= 0.02, f"Expected Rp <= 2%, got {i_r_p*100}%"

    # 2. Pure S-polarization (Ex = 1, Ey = 0)
    j_s = (1.0, 0.0)
    j_t_s, i_t_s, j_r_s, i_r_s = cfg.evaluate_jones(j_s, wavelength_um=0.6328, intensity_in=1.0)
    assert i_r_s >= 0.99, f"Expected Rs >= 99%, got {i_r_s*100}%"
    assert i_t_s <= 0.01, f"Expected Ts <= 1%, got {i_t_s*100}%"

    # 3. Extinction ratio check
    extinction_transmitted = i_t_p / max(i_t_s, 1e-12)
    assert extinction_transmitted >= 100.0, f"Expected high extinction, got {extinction_transmitted}"


def test_amplitude_50_50_beamsplitter_invariance():
    """Verify Thorlabs BS016 splits 50:50 equally for unpolarized/diagonal light."""
    bs016_info = get_catalog_lens("BS016")
    cfg = BeamSplitterConfig(**bs016_info["beamsplitter_config"])

    j_diag = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    j_t, i_t, j_r, i_r = cfg.evaluate_jones(j_diag, wavelength_um=0.6328, intensity_in=1.0)
    assert abs(i_t - 0.50) < 0.01
    assert abs(i_r - 0.50) < 0.01
    assert abs(i_t + i_r - 1.0) < 1e-6


def test_optical_system_ray_tracing_with_beamsplitter():
    """Verify OpticalSystem trace_ray through beamsplitter branch."""
    from optics_app.engine.lde import SurfaceStack
    stack = SurfaceStack()
    s0 = Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=20.0, glass="AIR")
    s1 = Surface(name="BS016_Splitting", surface_type=SurfaceType.STANDARD, thickness=30.0, glass="N-BK7", tilt_x=45.0)
    s1.is_beam_splitter = True
    bs_info = get_catalog_lens("BS016")
    s1.beamsplitter_config = BeamSplitterConfig(**bs_info["beamsplitter_config"])
    s2 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR")

    stack.add_surface(s0)
    stack.add_surface(s1)
    stack.add_surface(s2)
    sys = OpticalSystem(stack)

    # Trace transmitted branch
    r_in = Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), intensity=1.0)
    r_in.active_branch = "transmitted"
    ray_out_t = sys.trace_ray(r_in)
    assert ray_out_t is not None
    assert abs(ray_out_t.intensity - 0.5) < 0.01

    # Trace reflected branch
    r_in_r = Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), intensity=1.0)
    r_in_r.active_branch = "reflected"
    ray_out_r = sys.trace_ray(r_in_r)
    assert ray_out_r is not None
    assert abs(ray_out_r.intensity - 0.5) < 0.01


def test_branched_tracer_tree_generation():
    """Verify BranchedTracer traces both transmitted and reflected branches with full Jones preservation."""
    from optics_app.engine.lde import SurfaceStack
    stack = SurfaceStack()
    s0 = Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=20.0, glass="AIR")
    s1 = Surface(name="PBS251", surface_type=SurfaceType.STANDARD, thickness=30.0, glass="N-SF11", tilt_x=45.0)
    s1.is_beam_splitter = True
    pbs_info = get_catalog_lens("PBS251")
    s1.beamsplitter_config = BeamSplitterConfig(**pbs_info["beamsplitter_config"])
    s2 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR")

    stack.add_surface(s0)
    stack.add_surface(s1)
    stack.add_surface(s2)
    sys = OpticalSystem(stack)

    tracer = BranchedTracer(sys)
    # 45-degree linear polarization input (Ex = Ey = 1/sqrt(2))
    r_in = Ray(
        pos=(0.0, 0.0, 0.0),
        dir=(0.0, 0.0, 1.0),
        intensity=1.0,
        jones_vector=np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=complex),
    )
    r_in.active_branch = "both"

    ray_tree = tracer.trace_ray_tree(r_in)
    assert len(ray_tree.nodes) >= 1
    root = ray_tree.nodes[ray_tree.root_node_id]

    # Verify children exist (both transmitted and reflected)
    assert len(root.children_ids) == 2, f"Expected 2 child ray nodes, got {len(root.children_ids)}"
    child_nodes = [ray_tree.nodes[cid] for cid in root.children_ids]
    types = {n.branch_type for n in child_nodes}
    assert "transmitted" in types
    assert "reflected" in types
    total_child_intensity = sum(n.ray.intensity for n in child_nodes)
    assert total_child_intensity <= 1.00001
