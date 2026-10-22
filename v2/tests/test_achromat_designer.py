"""
Unit & Integration Test Suite for Achromatic Doublet Lens Designer & Optimization.
==================================================================================
Tests:
- Glass catalog retrieval, Abbe number (Vd) calculation, and partial dispersion (P_{F,d}).
- Crown vs. Flint glass classification and Abbe diagram representation.
- Achromatic pair ranking, secondary spectrum estimation, and scoring.
- Analytical thin-lens synthesis and Seidel bending root-finding (S_I = 0).
- Non-linear thick doublet surface optimization (curvatures and thicknesses).
- Elimination of primary chromatic aberration (BFL_F = BFL_C, C_L -> 0).
- Drastic reduction of spherical aberration (S_I -> 0, transverse ray fans).
- Air-spaced doublet optimization.
- Ray tracing through OpticalSystem and SurfaceStack.
- Interactive AchromatDesignerDialog GUI instantiation.
- Nemotron AI tool execution.
"""

import math
import json
import pytest
import numpy as np

from v2.engine.materials import GlassCatalog, get_refractive_index
from v2.engine.achromat_designer import (
    GlassInfo,
    GlassPairRecommendation,
    AchromatDesignSpec,
    AchromatDesignResult,
    get_all_catalog_glasses,
    get_glass_info,
    rank_achromatic_pairs,
    get_dispersion_curve_points,
    solve_thin_achromat,
    estimate_center_thicknesses,
    optimize_achromatic_doublet,
    FRAUNHOFER_D_UM,
    FRAUNHOFER_F_UM,
    FRAUNHOFER_C_UM,
)
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
from v2.ai.tools import get_default_tool_registry


# =============================================================================
# 1. Glass Catalog & Abbe Representation Tests
# =============================================================================

def test_catalog_glasses_extraction_and_abbe_values():
    """Verify that catalog glasses are extracted with accurate nd, Vd, and P_{F,d}."""
    glasses = get_all_catalog_glasses()
    assert len(glasses) >= 15, f"Expected at least 15 catalog glasses, found {len(glasses)}"

    crowns = [g for g in glasses if g.glass_type == "Crown"]
    flints = [g for g in glasses if g.glass_type == "Flint"]
    assert len(crowns) >= 5, "Expected at least 5 Crown glasses"
    assert len(flints) >= 8, "Expected at least 8 Flint glasses"

    # Verify N-BK7
    bk7 = get_glass_info("N-BK7")
    assert bk7.name == "N-BK7"
    assert bk7.glass_type == "Crown"
    assert abs(bk7.nd - 1.5168) < 0.005
    assert abs(bk7.vd - 64.17) < 0.5
    assert 0.65 < bk7.partial_dispersion < 0.75

    # Verify N-SF2 (Flint)
    sf2 = get_glass_info("N-SF2")
    assert sf2.glass_type == "Flint"
    assert abs(sf2.nd - 1.6478) < 0.005
    assert abs(sf2.vd - 33.79) < 0.5

    # Verify CaF2 (Fluorite Crown)
    caf2 = get_glass_info("CAF2")
    assert caf2.glass_type == "Crown"
    assert caf2.vd > 90.0  # Ultra-low dispersion


def test_dispersion_curves_monotonicity():
    """Verify refractive index dispersion curves are monotonically decreasing with wavelength."""
    for gname in ["N-BK7", "N-SF2", "N-SF11", "CAF2"]:
        wl_nm, n_vals = get_dispersion_curve_points(gname, 400.0, 750.0, 50)
        assert len(wl_nm) == 50
        # In standard optical transparency region, dn/dlambda < 0
        diffs = np.diff(n_vals)
        assert np.all(diffs < 0), f"Dispersion curve for {gname} must decrease monotonically with wavelength"


def test_achromatic_pair_ranking():
    """Verify Crown-Flint pair ranking algorithm finds and scores valid pairs."""
    glasses = get_all_catalog_glasses()
    pairs = rank_achromatic_pairs(glasses, min_delta_v=20.0)
    assert len(pairs) > 20, "Expected multiple viable Crown-Flint pairings"

    for p in pairs:
        assert p.delta_v >= 20.0
        assert p.crown.vd > p.flint.vd
        assert p.phi1_ratio > 0.0  # Crown power is positive
        assert p.phi2_ratio < 0.0  # Flint power is negative

    # Check top pairs contain recognized commercial combinations
    top_pairs_tuples = [(p.crown.name, p.flint.name) for p in pairs[:15]]
    has_bk7_sf = any("BK7" in c and "SF" in f for c, f in top_pairs_tuples)
    assert has_bk7_sf, "N-BK7 + N-SFx must be among top recommended pairs"


# =============================================================================
# 2. Analytical Thin-Lens Synthesis Tests
# =============================================================================

def test_solve_thin_achromat_bending():
    """Verify analytical thin-lens bending solve finds radii with near-zero S_I."""
    r1, r2, r3, phi1, phi2 = solve_thin_achromat(
        target_efl=100.0,
        crown_glass="N-BK7",
        flint_glass="N-SF2",
    )
    assert phi1 > 0.0, "Crown thin power must be positive"
    assert phi2 < 0.0, "Flint thin power must be negative"
    assert abs((phi1 + phi2) - 0.01) < 1e-4, "Sum of thin powers must equal 1/f = 0.01 mm^-1"

    # For standard Fraunhofer doublet (N-BK7 / N-SF2):
    # R1 is convex (> 0), R2 is cemented concave (< 0), R3 is weakly concave or flat
    assert r1 > 0.0, f"R1 should be convex (>0), got {r1}"
    assert r2 < 0.0, f"R2 should be concave (<0), got {r2}"


def test_center_thickness_estimation():
    """Verify center thickness estimation produces positive, manufacturable thicknesses."""
    t1, t2 = estimate_center_thicknesses(r1=68.0, r2=-38.0, r3=-110.0, diameter=25.4, min_edge_thickness=1.5)
    assert t1 >= 3.0, f"Crown thickness {t1} should be >= 3.0 mm"
    assert t2 >= 1.5, f"Flint thickness {t2} should be >= 1.5 mm"


# =============================================================================
# 3. Thick Doublet Optimization & Aberration Nulling Tests
# =============================================================================

def test_optimize_cemented_achromatic_doublet():
    """
    Core Goal Test: Optimize a cemented doublet to hit target EFL, null longitudinal
    color (C_L -> 0), and drastically reduce spherical aberration (S_I -> 0).
    """
    target_efl = 100.0
    spec = AchromatDesignSpec(
        target_efl=target_efl,
        diameter=25.4,
        crown_glass="N-BK7",
        flint_glass="N-SF2",
        doublet_type="cemented",
    )
    res = optimize_achromatic_doublet(spec)

    # 1. Target EFL Verification
    assert abs(res.efl - target_efl) < 0.01, f"Expected EFL ~ {target_efl} mm, got {res.efl:.4f} mm"

    # 2. Spherical Aberration Reduction
    # S_I magnitude should drop by at least 100x compared to unoptimized initial state
    assert abs(res.optimized_s_I) < 1e-5, f"Optimized S_I ({res.optimized_s_I:.4e}) must be < 1e-5"
    assert abs(res.optimized_s_I) < abs(res.initial_s_I), "Optimized S_I must be lower than initial S_I"

    # 3. Chromatic Aberration Nulling
    # Longitudinal color C_L should drop to near zero (< 1e-6)
    assert abs(res.optimized_c_L) < 1e-6, f"Optimized C_L ({res.optimized_c_L:.4e}) must be < 1e-6"

    # 4. Focal Shift Curve Check: BFL at F (486 nm) and C (656 nm) should be practically equal
    dbl = res.to_achromatic_doublet_element()
    cp_f = dbl.cardinal_points(486.13)
    cp_c = dbl.cardinal_points(656.27)
    bfl_diff = abs(cp_f.bfl - cp_c.bfl)
    assert bfl_diff < 0.005, f"BFL(F) and BFL(C) must agree within 5 um, diff = {bfl_diff:.6f} mm"


def test_optimize_air_spaced_doublet():
    """Verify air-spaced doublet optimization with independent contact curvatures."""
    spec = AchromatDesignSpec(
        target_efl=80.0,
        diameter=25.4,
        crown_glass="N-BK7",
        flint_glass="N-SF11",
        doublet_type="air_spaced",
        air_gap=0.3,
    )
    res = optimize_achromatic_doublet(spec)

    assert len(res.radii) == 4, f"Air-spaced doublet must have 4 radii, got {len(res.radii)}"
    assert abs(res.efl - 80.0) < 0.1, f"Expected EFL ~ 80.0 mm, got {res.efl:.4f} mm"
    assert abs(res.optimized_s_I) < 1e-4, "Optimized S_I must be small"


def test_nir_doublet_optimization():
    """Verify doublet optimization in the Near-Infrared (NIR) band (1064 nm)."""
    spec = AchromatDesignSpec(
        target_efl=75.0,
        diameter=25.4,
        crown_glass="N-LAK22",
        flint_glass="N-SF6",
        doublet_type="cemented",
        wavelength_d=1.064,
        wavelength_F=0.800,
        wavelength_C=1.300,
    )
    res = optimize_achromatic_doublet(spec)
    assert abs(res.efl - 75.0) < 0.05, f"Expected EFL ~ 75.0 mm, got {res.efl:.4f} mm"
    assert abs(res.optimized_s_I) < 1e-5


# =============================================================================
# 4. Ray Tracing & SurfaceStack Integration Tests
# =============================================================================

def test_surface_stack_and_ray_tracing_integration():
    """Verify that designed doublet converts to SurfaceStack and passes physical ray trace."""
    spec = AchromatDesignSpec(target_efl=100.0, diameter=25.4, crown_glass="N-BK7", flint_glass="N-SF2")
    res = optimize_achromatic_doublet(spec)

    surfaces = res.to_surfaces()
    assert len(surfaces) == 3

    stack = res.to_surface_stack()
    assert len(stack.surfaces) == 5  # Object, S1, S2, S3, Detector

    # Trace through OpticalSystem
    opt_sys = OpticalSystem(stack)

    # Trace on-axis ray
    ray_center = Ray(pos=np.array([0.0, 0.0, -10.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    out_center = opt_sys.trace_ray(ray_center)
    assert out_center is not None

    # Trace marginal ray at edge of aperture (y = 10 mm)
    ray_edge = Ray(pos=np.array([0.0, 10.0, -10.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    out_edge = opt_sys.trace_ray(ray_edge)
    assert out_edge is not None
    assert len(out_edge.history) >= 2


# =============================================================================
# 5. AI Tool Execution Tests
# =============================================================================

def test_ai_tools_achromat_execution():
    """Verify Nemotron AI tools correctly execute and return valid prescriptions."""
    registry = get_default_tool_registry()

    # 1. Test get_glass_dispersion_and_abbe
    out_str = registry.execute("get_glass_dispersion_and_abbe", {"glass_name": "N-BK7"})
    data = json.loads(out_str)
    assert data["glass"] == "N-BK7"
    assert data["type"] == "Crown"
    assert abs(data["nd"] - 1.5168) < 0.01

    # 2. Test list_recommended_achromatic_pairs
    out_pairs = registry.execute("list_recommended_achromatic_pairs", {"min_delta_v": 20.0, "limit": 5})
    p_data = json.loads(out_pairs)
    assert p_data["count"] > 0
    assert len(p_data["pairs"]) <= 5

    # 3. Test design_achromatic_doublet
    out_design = registry.execute("design_achromatic_doublet", {
        "target_efl_mm": 100.0,
        "diameter_mm": 25.4,
        "crown_glass": "N-BK7",
        "flint_glass": "N-SF2"
    })
    d_data = json.loads(out_design)
    assert d_data["status"] == "success"
    assert abs(d_data["actual_efl_mm"] - 100.0) < 0.05
    assert d_data["aberrations"]["spherical_reduction_factor"] > 10.0
    assert d_data["aberrations"]["color_reduction_factor"] > 10.0
    assert len(d_data["surfaces"]) == 3


# =============================================================================
# 6. Dialog Instantiation (Headless Qt) Test
# =============================================================================

def test_achromat_dialog_headless_instantiation(qapp):
    """Verify AchromatDesignerDialog initializes cleanly without crash."""
    from v2.gui.achromat_dialog import AchromatDesignerDialog
    dialog = AchromatDesignerDialog(model=None)
    assert dialog is not None
    assert dialog.current_result is not None
    assert dialog.current_result.efl > 0.0


def test_achromat_analysis_tab_in_suite(qapp):
    """Verify AchromatAnalysisTab integrates into AnalysisTabsWidget suite."""
    from v2.gui.analysis_tabs import AnalysisTabsWidget, AchromatAnalysisTab
    suite = AnalysisTabsWidget(model=None)
    assert "Achromatic Doublet Designer" in suite.tabs
    assert hasattr(suite, "tab_achromat")
    assert isinstance(suite.tab_achromat, AchromatAnalysisTab)
    switched = suite.set_active_tab("Achromatic Doublet Designer")
    assert switched is True
    assert suite.active_tab == "Achromatic Doublet Designer"
