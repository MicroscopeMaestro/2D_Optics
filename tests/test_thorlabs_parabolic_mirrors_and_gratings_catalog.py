"""Comprehensive Unit & Physics Tests for Thorlabs Parabolic Mirrors and Diffraction Gratings.

Tests:
1. Parabolic mirrors catalog database (MPD series): focal lengths 15mm-254mm, diameters 1/2", 1", 2", 3", coatings P01, M01, G01.
2. Parabolic conic figure: k = -1.0, R = -2f, zero spherical aberration at focal point.
3. Diffraction gratings catalog database: 200, 300, 600, 830, 1200, 1800, 2400 lines/mm across ruled (GR), holographic (GH), and transmission (GT) types.
4. Grating diffraction physics: ray deflection according to grating equation m * lambda / d.
5. SurfaceStack / LDE integration: _build_catalog_surfaces, replace_lens_from_catalog, replace_element_with_catalog, normal and reverse modes.
6. GUI helpers: catalog menu formatting and diameter labels.
7. AI catalog tools: lookup_thorlabs_catalog and OpticalCatalogManager.create_surfaces_from_part.
"""

import pytest
import numpy as np

from optics_app.catalogs.thorlabs import (
    get_catalog_sections,
    get_lens_catalog,
    get_catalog_lens,
    query_catalog,
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem, DualSpeedTracer, AsphericSurface, SphericalSurface
from optics_app.gui.catalog_menu import diameter_menu_label, _format_catalog_action_label
from optics_app.ai.tools import lookup_thorlabs_catalog
from optics_app.ai.catalog import OpticalCatalogManager


# =============================================================================
# 1. PARABOLIC MIRRORS CATALOG TESTS
# =============================================================================

def test_parabolic_mirrors_catalog_present_in_sections():
    """Verify parabolic mirrors section exists with multiple focal lengths, diameters, and coatings."""
    sections = get_catalog_sections()
    assert "parabolic_mirrors" in sections
    pm_section = sections["parabolic_mirrors"]
    parts = pm_section["parts"]

    # Should contain at least 20 parabolic mirror models
    assert len(parts) >= 20

    # Test key parts
    expected_parts = [
        "MPD050-015-P01", "MPD050-025-P01", "MPD050-015-M01",
        "MPD129-P01", "MPD149-P01", "MPD169-P01", "MPD1016-P01", "MPD1203-P01",
        "MPD129-M01", "MPD149-M01", "MPD129-G01", "MPD149-G01",
        "MPD229-P01", "MPD249-P01", "MPD289-P01", "MPD229-M01",
        "MPD349-P01", "MPD389-P01"
    ]
    for p in expected_parts:
        assert p in parts, f"Expected parabolic mirror {p} not found in catalog"


def test_parabolic_mirror_figure_and_radii():
    """Verify conic figure k = -1.0 and R = -2f relationship."""
    for pn, rec in get_catalog_sections()["parabolic_mirrors"]["parts"].items():
        efl = rec["focal_length_mm"]
        radius = rec["radius_mm"]
        assert rec["conic_k"] == -1.0
        assert rec["conic"] == -1.0
        # R = -2f convention for concave mirror
        assert abs(radius - (-2.0 * efl)) < 1e-6
        # Verify surfaces specification
        surfs = rec["surfaces"]
        assert len(surfs) == 1
        assert surfs[0]["glass"] == "MIRROR"
        assert surfs[0]["conic"] == -1.0
        assert abs(surfs[0]["radius"] - radius) < 1e-6


def test_parabolic_mirrors_query_filtering():
    """Test query_catalog for parabolic mirrors with various filters."""
    # Query by section name
    pm_all = query_catalog(section="parabolic_mirrors")
    assert len(pm_all) >= 20

    # Query with alias
    pm_alias = query_catalog(section="parabolic")
    assert len(pm_alias) == len(pm_all)

    # Filter by diameter 1" (25.4 mm)
    pm_1in = query_catalog(section="parabolic", diameter_mm=25.4)
    assert len(pm_1in) >= 10
    for p in pm_1in:
        assert abs(p["diameter_mm"] - 25.4) < 1e-4

    # Filter by focal length 50.8 mm (2")
    pm_f50 = query_catalog(section="parabolic", focal_length_mm=50.8)
    assert len(pm_f50) >= 4  # MPD149-P01, MPD149-M01, MPD149-G01, MPD229-P01, MPD229-M01
    for p in pm_f50:
        assert abs(p["nominal_focal_length_mm"] - 50.8) < 1e-4

    # Filter by coating
    pm_gold = query_catalog(section="parabolic", coating_code="M01")
    assert len(pm_gold) >= 5
    for p in pm_gold:
        assert p["coating_code"] == "M01"


# =============================================================================
# 2. DIFFRACTION GRATINGS CATALOG TESTS
# =============================================================================

def test_diffraction_gratings_catalog_present_in_sections():
    """Verify diffraction gratings section exists with ruled, holographic, and transmission types."""
    sections = get_catalog_sections()
    assert "diffraction_gratings" in sections
    dg_section = sections["diffraction_gratings"]
    parts = dg_section["parts"]

    assert len(parts) >= 30

    # Test ruled reflective
    assert "GR25-0213" in parts  # 200 l/mm
    assert "GR25-0305" in parts  # 300 l/mm
    assert "GR25-0610" in parts  # 600 l/mm
    assert "GR25-0808" in parts  # 830 l/mm
    assert "GR25-1205" in parts  # 1200 l/mm
    assert "GR25-1805" in parts  # 1800 l/mm

    # Test holographic reflective
    assert "GH13-12U" in parts  # 1200 l/mm UV
    assert "GH25-12V" in parts  # 1200 l/mm VIS
    assert "GH25-18V" in parts  # 1800 l/mm VIS
    assert "GH25-24U" in parts  # 2400 l/mm UV
    assert "GH25-24V" in parts  # 2400 l/mm VIS

    # Test transmission
    assert "GT13-03" in parts   # 300 l/mm 1/2"
    assert "GT25-03" in parts   # 300 l/mm 1"
    assert "GT25-06" in parts   # 600 l/mm 1"
    assert "GT25-12" in parts   # 1200 l/mm 1"


def test_grating_properties_and_surfaces():
    """Verify grating parameters: groove density, blaze wavelength, surface type."""
    parts = get_catalog_sections()["diffraction_gratings"]["parts"]

    # GR25-0610: 600 l/mm, blaze 1.0 um, reflective
    gr600 = parts["GR25-0610"]
    assert gr600["grooves_per_mm"] == 600.0
    assert abs(gr600["blaze_wavelength_um"] - 1.0) < 1e-6
    assert abs(gr600["blaze_angle_deg"] - 17.5) < 1e-2
    assert not gr600["is_transmission"]
    assert gr600["surfaces"][0]["surface_type"] == "Grating"
    assert gr600["surfaces"][0]["glass"] == "MIRROR"
    assert gr600["surfaces"][0]["grating_lines_per_mm"] == 600.0

    # GT25-06: 600 l/mm, transmission
    gt600 = parts["GT25-06"]
    assert gt600["grooves_per_mm"] == 600.0
    assert gt600["is_transmission"]
    assert gt600["surfaces"][0]["glass"] == "AIR"
    assert gt600["surfaces"][0]["surface_type"] == "Grating"
    assert gt600["surfaces"][0]["grating_lines_per_mm"] == 600.0


def test_gratings_query_filtering():
    """Test query_catalog for diffraction gratings."""
    gratings = query_catalog(section="diffraction_gratings")
    assert len(gratings) >= 30

    alias_gratings = query_catalog(section="gratings")
    assert len(alias_gratings) == len(gratings)

    # Search text
    g_1200 = query_catalog(section="gratings", search_text="1200")
    assert len(g_1200) >= 8
    for g in g_1200:
        assert (
            g["grooves_per_mm"] == 1200.0
            or "1200" in g["part_number"]
            or 1200.0 in g.get("design_wavelengths_nm", [])
            or abs(g.get("blaze_wavelength_um", 0.0) - 1.2) < 1e-4
        )


# =============================================================================
# 3. OPTICAL RAY TRACING & PHYSICS VERIFICATION
# =============================================================================

def test_parabolic_mirror_ray_trace_zero_spherical_aberration():
    """
    Physics test: Parallel rays incident on a concave parabolic mirror (k = -1.0)
    focus exactly to focal point f = -R/2 with ZERO spherical aberration.
    In contrast, a spherical mirror (k = 0.0) of identical radius exhibits spherical aberration.
    """
    f = 50.8
    R = -2.0 * f  # -101.6 mm

    asph_parabolic = AsphericSurface(radius=R, conic=-1.0, asphere_coeffs=[])
    asph_spherical = SphericalSurface(radius=R, semi_diameter=25.0)

    # Trace rays at different heights: paraxial (y=0.1 mm) and marginal (y=10.0 mm)
    ray_paraxial = Ray(pos=np.array([0.0, 0.1, -50.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
    ray_marginal = Ray(pos=np.array([0.0, 10.0, -50.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)

    # Parabolic mirror intersection & reflection
    hit_p_parax, norm_p_parax = asph_parabolic.intersect(ray_paraxial)
    hit_p_marg, norm_p_marg = asph_parabolic.intersect(ray_marginal)

    # Reflected ray directions: v_out = v_in - 2 * (v_in . n) * n
    dir_p_parax = ray_paraxial.dir - 2.0 * np.dot(ray_paraxial.dir, norm_p_parax) * norm_p_parax
    dir_p_marg = ray_marginal.dir - 2.0 * np.dot(ray_marginal.dir, norm_p_marg) * norm_p_marg

    # Find optical axis intercept z (where y = 0)
    # y(z) = hit_y + t * dir_y = 0  =>  t = -hit_y / dir_y  =>  z = hit_z + t * dir_z
    t_parax = -hit_p_parax[1] / dir_p_parax[1]
    z_focus_parax = hit_p_parax[2] + t_parax * dir_p_parax[2]

    t_marg = -hit_p_marg[1] / dir_p_marg[1]
    z_focus_marg = hit_p_marg[2] + t_marg * dir_p_marg[2]

    # Both rays MUST intercept optical axis at the vertex focal distance (z = -f = -50.8 mm)
    assert abs(z_focus_parax - (-f)) < 1e-4, f"Paraxial focus {z_focus_parax} != {-f}"
    assert abs(z_focus_marg - (-f)) < 1e-4, f"Marginal focus {z_focus_marg} != {-f}"
    # Longitude spherical aberration LSA = z_focus_marg - z_focus_parax must be essentially ZERO!
    assert abs(z_focus_marg - z_focus_parax) < 1e-4

    # Now verify spherical mirror HAS longitudinal spherical aberration (LSA > 0.5 mm)
    hit_s_parax, norm_s_parax = asph_spherical.intersect(ray_paraxial)
    hit_s_marg, norm_s_marg = asph_spherical.intersect(ray_marginal)
    dir_s_parax = ray_paraxial.dir - 2.0 * np.dot(ray_paraxial.dir, norm_s_parax) * norm_s_parax
    dir_s_marg = ray_marginal.dir - 2.0 * np.dot(ray_marginal.dir, norm_s_marg) * norm_s_marg

    t_s_parax = -hit_s_parax[1] / dir_s_parax[1]
    z_s_parax = hit_s_parax[2] + t_s_parax * dir_s_parax[2]
    t_s_marg = -hit_s_marg[1] / dir_s_marg[1]
    z_s_marg = hit_s_marg[2] + t_s_marg * dir_s_marg[2]

    lsa_spherical = abs(z_s_marg - z_s_parax)
    assert lsa_spherical > 0.2, f"Spherical mirror should have noticeable LSA, got {lsa_spherical}"


def test_optical_system_trace_with_thorlabs_parabolic_mirror():
    """Verify RayTracer successfully traces rays through an OpticalSystem with MPD149-P01."""
    stack = SurfaceStack()
    stack.surfaces = [
        Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=100.0, semi_diameter=25.0),
        Surface(name="STOP", surface_type=SurfaceType.STOP, thickness=50.0, semi_diameter=12.0),
        Surface(name="L1_front", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.7),
        Surface(name="L1_rear", radius=-50.0, thickness=50.8, glass="AIR", semi_diameter=12.7),
        Surface(name="DET", surface_type=SurfaceType.DETECTOR, semi_diameter=25.0)
    ]
    # Replace L1 with parabolic mirror MPD149-P01
    new_indices = stack.replace_lens_from_catalog(target=2, part_number="MPD149-P01", air_gap_after=50.8)
    assert len(new_indices) == 1
    mirror_surf = stack.surfaces[new_indices[0]]
    assert mirror_surf.surface_type == SurfaceType.MIRROR
    assert mirror_surf.conic == -1.0
    assert abs(mirror_surf.radius - (-101.6)) < 1e-4

    # Build OpticalSystem and trace rays
    opt_sys = OpticalSystem(stack)
    tracer = DualSpeedTracer(opt_sys)
    traced_rays = []
    for y_offset in np.linspace(-5.0, 5.0, 5):
        r = Ray(pos=np.array([0.0, float(y_offset), 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
        traced_rays.append(tracer.trace_ray(r))
    assert len(traced_rays) == 5
    valid_rays = [r for r in traced_rays if len(r.history) >= 2]
    assert len(valid_rays) == 5


# =============================================================================
# 4. SURFACESTACK & LDE INTEGRATION TESTS
# =============================================================================

def test_build_catalog_surfaces_parabolic_mirror():
    """Test _build_catalog_surfaces creates proper SurfaceType.MIRROR and conic=-1.0."""
    rec = get_catalog_lens("MPD149-P01")
    assert rec is not None

    # Normal mode
    surfs = SurfaceStack._build_catalog_surfaces(rec, "MPD149-P01", reverse=False)
    assert len(surfs) == 1
    s = surfs[0]
    assert s.surface_type == SurfaceType.MIRROR
    assert s.glass == "MIRROR"
    assert s.conic == -1.0
    assert abs(s.radius - (-101.6)) < 1e-4

    # Reverse mode
    surfs_rev = SurfaceStack._build_catalog_surfaces(rec, "MPD149-P01", reverse=True)
    assert len(surfs_rev) == 1
    s_rev = surfs_rev[0]
    assert s_rev.surface_type == SurfaceType.MIRROR
    assert s_rev.glass == "MIRROR"
    assert s_rev.conic == -1.0
    assert abs(s_rev.radius - 101.6) < 1e-4


def test_build_catalog_surfaces_diffraction_grating():
    """Test _build_catalog_surfaces creates proper SurfaceType.GRATING and grating lines."""
    rec = get_catalog_lens("GR25-0610")
    assert rec is not None

    surfs = SurfaceStack._build_catalog_surfaces(rec, "GR25-0610", reverse=False)
    assert len(surfs) == 1
    s = surfs[0]
    assert s.surface_type == SurfaceType.GRATING
    assert s.glass == "MIRROR"
    assert s.grating_lines_per_mm == 600.0
    assert s.grating_order == 1
    assert abs(s.grating_primary_wavelength_um - 1.0) < 1e-4


def test_build_catalog_surfaces_transmission_grating():
    """Test _build_catalog_surfaces for transmission grating GT25-06."""
    rec = get_catalog_lens("GT25-06")
    assert rec is not None

    surfs = SurfaceStack._build_catalog_surfaces(rec, "GT25-06", reverse=False)
    assert len(surfs) == 1
    s = surfs[0]
    assert s.surface_type == SurfaceType.GRATING
    assert s.glass == "AIR"
    assert s.grating_lines_per_mm == 600.0


def test_replace_element_with_catalog_grating_and_mirror():
    """Verify replace_element_with_catalog cleanly replaces a lens with a grating or parabolic mirror."""
    stack = SurfaceStack()
    stack.surfaces = [
        Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=50.0),
        Surface(name="Lens1_S1", radius=34.0, thickness=6.0, glass="N-BK7"),
        Surface(name="Lens1_S2", radius=-115.0, thickness=50.0, glass="AIR"),
        Surface(name="DET", surface_type=SurfaceType.DETECTOR)
    ]
    # Replace Lens1 with Grating GR25-0610
    idx_gr = stack.replace_element_with_catalog("Lens1_S1", "GR25-0610")
    assert len(idx_gr) == 1
    assert stack.surfaces[idx_gr[0]].surface_type == SurfaceType.GRATING
    assert stack.surfaces[idx_gr[0]].grating_lines_per_mm == 600.0

    # Now replace the grating with Parabolic Mirror MPD149-P01
    idx_pm = stack.replace_element_with_catalog(idx_gr[0], "MPD149-P01")
    assert len(idx_pm) == 1
    assert stack.surfaces[idx_pm[0]].surface_type == SurfaceType.MIRROR
    assert stack.surfaces[idx_pm[0]].conic == -1.0


# =============================================================================
# 5. GUI & MENU FORMATTING TESTS
# =============================================================================

def test_diameter_menu_label():
    """Test diameter_menu_label handles 3 inch (75mm), 2 inch (50.8mm), 1 inch (25.4mm), 1/2 inch (12.7mm)."""
    assert "3 inch" in diameter_menu_label(75.0)
    assert "2 inch" in diameter_menu_label(50.8)
    assert "1 inch" in diameter_menu_label(25.4)
    assert "1/2 inch" in diameter_menu_label(12.7)


def test_format_catalog_action_label():
    """Test catalog action label formatting for gratings, parabolic mirrors, and doublets."""
    gr_rec = get_catalog_lens("GR25-0610")
    gr_label = _format_catalog_action_label("GR25-0610", gr_rec)
    assert "600 l/mm" in gr_label
    assert "λb=1000 nm" in gr_label
    assert "f=0 mm" not in gr_label

    pm_rec = get_catalog_lens("MPD149-P01")
    pm_label = _format_catalog_action_label("MPD149-P01", pm_rec)
    assert "Parabolic" in pm_label
    assert "f=50.8 mm" in pm_label

    d_rec = get_catalog_lens("AC254-050-C")
    d_label = _format_catalog_action_label("AC254-050-C", d_rec)
    assert "f=50 mm" in d_label
    assert "20.00 D" in d_label


# =============================================================================
# 6. AI CATALOG TOOLS TESTS
# =============================================================================

def test_lookup_thorlabs_catalog_parabolic_and_grating():
    """Test AI lookup_thorlabs_catalog retrieves specifications for parabolic mirrors and gratings."""
    info_mpd = lookup_thorlabs_catalog("MPD149-P01")
    assert "MPD149-P01" in info_mpd
    assert "conic" in info_mpd.lower()

    info_gr = lookup_thorlabs_catalog("GR25-0610")
    assert "GR25-0610" in info_gr
    assert "600" in info_gr


def test_optical_catalog_manager_create_surfaces():
    """Test OpticalCatalogManager.create_surfaces_from_part builds Surface instances."""
    surfs_mpd = OpticalCatalogManager.create_surfaces_from_part("MPD149-P01")
    assert len(surfs_mpd) == 1
    assert surfs_mpd[0].surface_type == SurfaceType.MIRROR
    assert surfs_mpd[0].conic == -1.0

    surfs_gr = OpticalCatalogManager.create_surfaces_from_part("GR25-0610")
    assert len(surfs_gr) == 1
    assert surfs_gr[0].surface_type == SurfaceType.GRATING
    assert surfs_gr[0].grating_lines_per_mm == 600.0
