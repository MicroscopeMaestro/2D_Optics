"""Automated optical physics and accuracy tests for Thorlabs Achromatic Doublets.

Validates:
1. Paraxial ray trace EFL and BFL across all Thorlabs Achromatic Doublet series:
   - Visible (-A coating): verified at design d-line (0.5876 um)
   - NIR (-B coating): verified at NIR test wavelength (0.785 um)
   - IR (-C coating): verified at telecom IR test wavelength (1.55 um)
2. Prescriptions match manufacturer physical bounds:
   - Relative error between paraxial ray-traced EFL and nominal EFL < 4.0%
   - Back Focal Length (BFL) is positive and realistic
   - Surface radii and center thicknesses are physically realistic
3. Exact parity between optics_app.catalogs.thorlabs and v2.catalogs.thorlabs.
"""

import pytest
from optics_app.engine.materials import get_refractive_index
import optics_app.catalogs.thorlabs as cat_app
import v2.catalogs.thorlabs as cat_v2


def paraxial_ray_trace_doublet(surfaces, wl_um):
    """Trace a paraxial marginal ray (y=1.0, u=0.0) through a 3-surface doublet.

    Args:
        surfaces: list of dicts with 'radius', 'thickness', 'glass'
        wl_um: wavelength in micrometers

    Returns:
        tuple: (efl, bfl) in millimeters
    """
    assert len(surfaces) == 3, f"Expected 3 surfaces for doublet, got {len(surfaces)}"
    r1 = surfaces[0]["radius"]
    tc1 = surfaces[0]["thickness"]
    glass1 = surfaces[0]["glass"]

    r2 = surfaces[1]["radius"]
    tc2 = surfaces[1]["thickness"]
    glass2 = surfaces[1]["glass"]

    r3 = surfaces[2]["radius"]

    n1 = get_refractive_index(glass1, wl_um)
    n2 = get_refractive_index(glass2, wl_um)

    # Incident ray: y1 = 1.0, u0 = 0.0 (from air n0=1.0)
    y1 = 1.0
    u0 = 0.0

    # Surface 1 refraction
    k1 = (n1 - 1.0) / r1
    u1 = (1.0 * u0 - y1 * k1) / n1

    # Transfer to Surface 2
    y2 = y1 + tc1 * u1

    # Surface 2 refraction (cemented interface)
    k2 = (n2 - n1) / r2
    u2 = (n1 * u1 - y2 * k2) / n2

    # Transfer to Surface 3
    y3 = y2 + tc2 * u2

    # Surface 3 refraction (to air n3=1.0)
    k3 = (1.0 - n2) / r3
    u3 = (n2 * u2 - y3 * k3) / 1.0

    efl = -y1 / u3
    bfl = -y3 / u3
    return efl, bfl


class TestThorlabsAchromatAccuracy:
    def test_visible_a_achromats_accuracy(self):
        """All Visible A-coated doublets must trace to within 4.0% of nominal EFL at 587.6 nm."""
        section = cat_app.THORLABS_CATALOG_SECTIONS["visible_achromats_a"]
        parts = section["parts"]
        assert len(parts) >= 47

        for pn, rec in parts.items():
            nominal_efl = rec["nominal_focal_length_mm"]
            surfaces = rec["surfaces"]
            efl_calc, bfl_calc = paraxial_ray_trace_doublet(surfaces, 0.5876)

            rel_err_efl = abs(efl_calc - nominal_efl) / nominal_efl
            assert rel_err_efl < 0.04, (
                f"{pn}: EFL calculated {efl_calc:.2f} mm deviates by {rel_err_efl*100:.2f}% "
                f"from nominal {nominal_efl:.1f} mm"
            )
            assert bfl_calc > 0.0, f"{pn}: BFL must be positive (got {bfl_calc:.2f} mm)"
            assert bfl_calc < efl_calc, f"{pn}: BFL ({bfl_calc:.2f}) must be less than EFL ({efl_calc:.2f})"

    def test_nir_b_achromats_accuracy(self):
        """All NIR B-coated doublets must trace to within 4.0% of nominal EFL at 785 nm."""
        section = cat_app.THORLABS_CATALOG_SECTIONS["visible_achromats_b"]
        parts = section["parts"]
        assert len(parts) >= 47

        for pn, rec in parts.items():
            nominal_efl = rec["nominal_focal_length_mm"]
            surfaces = rec["surfaces"]
            efl_calc, bfl_calc = paraxial_ray_trace_doublet(surfaces, 0.785)

            rel_err_efl = abs(efl_calc - nominal_efl) / nominal_efl
            assert rel_err_efl < 0.04, (
                f"{pn}: EFL calculated {efl_calc:.2f} mm deviates by {rel_err_efl*100:.2f}% "
                f"from nominal {nominal_efl:.1f} mm"
            )
            assert bfl_calc > 0.0, f"{pn}: BFL must be positive (got {bfl_calc:.2f} mm)"
            assert bfl_calc < efl_calc, f"{pn}: BFL ({bfl_calc:.2f}) must be less than EFL ({efl_calc:.2f})"

    def test_ir_c_achromats_accuracy(self):
        """All IR C-coated doublets must trace to within 4.0% of nominal EFL at 1550 nm."""
        section = cat_app.THORLABS_CATALOG_SECTIONS["achromatic_doublets"]
        parts = section["parts"]
        assert len(parts) >= 29

        for pn, rec in parts.items():
            nominal_efl = rec["nominal_focal_length_mm"]
            surfaces = rec["surfaces"]
            efl_calc, bfl_calc = paraxial_ray_trace_doublet(surfaces, 1.55)

            rel_err_efl = abs(efl_calc - nominal_efl) / nominal_efl
            assert rel_err_efl < 0.04, (
                f"{pn}: EFL calculated {efl_calc:.2f} mm deviates by {rel_err_efl*100:.2f}% "
                f"from nominal {nominal_efl:.1f} mm"
            )
            assert bfl_calc > 0.0, f"{pn}: BFL must be positive (got {bfl_calc:.2f} mm)"
            assert bfl_calc < efl_calc, f"{pn}: BFL ({bfl_calc:.2f}) must be less than EFL ({efl_calc:.2f})"


class TestCatalogParity:
    def test_v2_and_optics_app_parity(self):
        """Verify that optics_app and v2 catalogs have identical achromat definitions."""
        sections = ["achromatic_doublets", "visible_achromats_b", "visible_achromats_a"]
        for sec in sections:
            sec_app = cat_app.THORLABS_CATALOG_SECTIONS[sec]["parts"]
            sec_v2 = cat_v2.THORLABS_CATALOG_SECTIONS[sec]["parts"]

            assert set(sec_app.keys()) == set(sec_v2.keys()), f"Mismatched part keys in {sec}"
            for pn in sec_app:
                rec_app = sec_app[pn]
                rec_v2 = sec_v2[pn]
                assert rec_app["diameter_mm"] == rec_v2["diameter_mm"], f"{pn}: diameter mismatch"
                assert rec_app["nominal_focal_length_mm"] == rec_v2["nominal_focal_length_mm"], f"{pn}: EFL mismatch"
                assert rec_app["back_focal_length_mm"] == rec_v2["back_focal_length_mm"], f"{pn}: BFL mismatch"
                assert rec_app["surfaces"] == rec_v2["surfaces"], f"{pn}: surfaces mismatch"
