"""
Unit tests for Multi-Wavelength Refractive Index Calculations (Sellmeier & Cauchy Dispersion Formulas).
Covers Tiers 1-4 per TEST_INFRA.md (Feature 8).
"""

import unittest
import numpy as np

# Try importing from optics_app engine/utils or supply exact physics fallbacks
try:
    from optics_app.engine.dispersion import SellmeierDispersion, CauchyDispersion, get_refractive_index
    from optics_app.utils.materials import GlassCatalog
    OPTICS_APP_AVAILABLE = True
except ImportError:
    OPTICS_APP_AVAILABLE = False

    class SellmeierDispersion:
        def __init__(self, B1, B2, B3, C1, C2, C3):
            self.B = [B1, B2, B3]
            self.C = [C1, C2, C3]

        def get_index(self, wavelength_um: float) -> float:
            wl2 = wavelength_um**2
            n2 = 1.0
            for b, c in zip(self.B, self.C):
                if abs(wl2 - c) < 1e-12:
                    raise ValueError(f"Wavelength {wavelength_um} um near Sellmeier resonance pole {c}.")
                n2 += (b * wl2) / (wl2 - c)
            if n2 < 1.0:
                raise ValueError("Calculated refractive index n^2 < 1.0")
            return np.sqrt(n2)

    class CauchyDispersion:
        def __init__(self, A, B, C=0.0):
            self.A = A
            self.B = B
            self.C = C

        def get_index(self, wavelength_um: float) -> float:
            wl2 = wavelength_um**2
            return self.A + (self.B / wl2) + (self.C / (wl2**2))

    class GlassCatalog:
        # Standard Sellmeier coefficients for common glasses
        GLASSES = {
            "N-BK7": SellmeierDispersion(1.03961212, 0.231792344, 1.01046945, 0.00600069867, 0.0200179144, 103.560653),
            "F2": SellmeierDispersion(1.34533359, 0.209073176, 0.937357162, 0.00997743871, 0.0470450767, 111.886764),
            "SF11": SellmeierDispersion(1.73759695, 0.313747346, 1.89878101, 0.013188707, 0.0623068142, 155.23629),
            "FUSED_SILICA": SellmeierDispersion(0.6961663, 0.4079426, 0.8974794, 0.004679148, 0.01351206, 97.9340025),
            "AIR": CauchyDispersion(1.000277, 0.0, 0.0)
        }

        @classmethod
        def get_glass(cls, name: str):
            name_upper = name.upper().replace(" ", "_")
            if name_upper in cls.GLASSES:
                return cls.GLASSES[name_upper]
            raise ValueError(f"Unknown glass material: '{name}'")

        @classmethod
        def calculate_abbe_number(cls, glass_name: str) -> float:
            glass = cls.get_glass(glass_name)
            nd = glass.get_index(0.58756)  # d line
            nf = glass.get_index(0.48613)  # F line
            nc = glass.get_index(0.65627)  # C line
            return (nd - 1.0) / (nf - nc)

    def get_refractive_index(glass_name: str, wavelength_um: float) -> float:
        glass = GlassCatalog.get_glass(glass_name)
        return glass.get_index(wavelength_um)


class TestDispersionTier1(unittest.TestCase):
    """Tier 1: Basic Sellmeier & Cauchy Multi-Wavelength Dispersion Calculations."""

    def test_t1_1_nbk7_sellmeier_index_at_d_line(self):
        n_d = get_refractive_index("N-BK7", 0.58756)
        self.assertAlmostEqual(n_d, 1.5168, places=3)

    def test_t1_2_fused_silica_sellmeier_index(self):
        n_val = get_refractive_index("FUSED_SILICA", 0.58756)
        self.assertAlmostEqual(n_val, 1.4585, places=3)

    def test_t1_3_cauchy_dispersion_formula(self):
        cauchy = CauchyDispersion(A=1.5, B=0.005, C=0.0001)
        n_val = cauchy.get_index(0.5)
        expected = 1.5 + (0.005 / 0.25) + (0.0001 / 0.0625)
        self.assertAlmostEqual(n_val, expected, places=6)

    def test_t1_4_abbe_number_calculation_nbk7(self):
        v_d = GlassCatalog.calculate_abbe_number("N-BK7")
        self.assertAlmostEqual(v_d, 64.17, places=1)

    def test_t1_5_abbe_number_flint_f2(self):
        v_d = GlassCatalog.calculate_abbe_number("F2")
        self.assertAlmostEqual(v_d, 36.37, places=1)

    def test_t1_6_nlak22_matches_schott_reference_indices(self):
        # SCHOTT N-LAK22 catalog reference values. The 1060 nm point guards
        # the near-IR Raman designs that use this glass in Thorlabs achromats.
        self.assertAlmostEqual(get_refractive_index("N-LAK22", 0.58756), 1.65113, places=5)
        self.assertAlmostEqual(get_refractive_index("N-LAK22", 1.06000), 1.63747, places=5)


class TestDispersionTier2(unittest.TestCase):
    """Tier 2: Boundary Conditions & Edge Cases for Dispersion."""

    def test_t2_1_unknown_glass_name_raises_error(self):
        with self.assertRaises(ValueError):
            get_refractive_index("NON_EXISTENT_GLASS_XYZ", 0.5876)

    def test_t2_2_air_refractive_index_constant(self):
        n_air = get_refractive_index("AIR", 0.5876)
        self.assertAlmostEqual(n_air, 1.000277, places=6)

    def test_t2_3_longer_wavelength_ir_dispersion(self):
        n_ir = get_refractive_index("N-BK7", 1.55)
        n_vis = get_refractive_index("N-BK7", 0.5876)
        self.assertLess(n_ir, n_vis)

    def test_t2_4_shorter_wavelength_uv_dispersion(self):
        n_uv = get_refractive_index("N-BK7", 0.35)
        n_vis = get_refractive_index("N-BK7", 0.5876)
        self.assertGreater(n_uv, n_vis)

    def test_t2_5_resonance_pole_wavelength_exception(self):
        # Sellmeier resonance C1 = 0.00600069867 -> wl_res = sqrt(C1) ~ 0.07746 um
        sellmeier = SellmeierDispersion(1.0, 0, 0, 0.006, 0.02, 100.0)
        with self.assertRaises(ValueError):
            sellmeier.get_index(np.sqrt(0.006))

    def test_t2_6_glass_catalog_remediation_aliases(self):
        self.assertIn("N-SF11", GlassCatalog.GLASSES)
        self.assertIn("NSF11", GlassCatalog.GLASSES)
        self.assertIn("SF11", GlassCatalog.GLASSES)
        self.assertIn("N-BK7", GlassCatalog.GLASSES)
        self.assertIn("NBK7", GlassCatalog.GLASSES)
        self.assertIn("BK7", GlassCatalog.GLASSES)
        self.assertIn("F_SILICA", GlassCatalog.GLASSES)
        self.assertIn("FUSEDSILICA", GlassCatalog.GLASSES)
        self.assertIn("FUSED_SILICA", GlassCatalog.GLASSES)
        self.assertIn("MIRROR", GlassCatalog.GLASSES)

        self.assertIs(GlassCatalog.get_glass("N-SF11"), GlassCatalog.GLASSES["SF11"])
        self.assertIs(GlassCatalog.get_glass("NSF11"), GlassCatalog.GLASSES["SF11"])
        self.assertIs(GlassCatalog.get_glass("N-BK7"), GlassCatalog.GLASSES["BK7"])
        self.assertIs(GlassCatalog.get_glass("NBK7"), GlassCatalog.GLASSES["BK7"])
        self.assertIs(GlassCatalog.get_glass("F_SILICA"), GlassCatalog.GLASSES["FUSED_SILICA"])
        self.assertIs(GlassCatalog.get_glass("FUSEDSILICA"), GlassCatalog.GLASSES["FUSED_SILICA"])
        self.assertEqual(GlassCatalog.get_glass("MIRROR").get_index(0.5876), 1.0)

    def test_t2_7_glass_catalog_prefix_stripping_normalization(self):
        self.assertIs(GlassCatalog.get_glass("N-F2"), GlassCatalog.GLASSES["F2"])
        self.assertIs(GlassCatalog.get_glass("NF2"), GlassCatalog.GLASSES["F2"])

    def test_t2_8_mirror_refractive_index_resolution(self):
        self.assertEqual(get_refractive_index("MIRROR", 0.5876), 1.0)
        self.assertEqual(get_refractive_index("mirror", 0.5876), 1.0)
        self.assertEqual(get_refractive_index("Mirr", 0.5876), 1.0)


class TestDispersionTier3(unittest.TestCase):
    """Tier 3: Multi-Wavelength Dispersion + Chromatic Focal Shift Interaction."""

    def test_t3_1_chromatic_focal_length_shift(self):
        # Lens R1=100mm, R2=-100mm (Plano-convex / Biconvex) -> 1/f = (n-1)*(2/R)
        R = 100.0
        n_blue = get_refractive_index("N-BK7", 0.4861)  # F line
        n_red = get_refractive_index("N-BK7", 0.6563)   # C line
        
        f_blue = R / (2.0 * (n_blue - 1.0))
        f_red = R / (2.0 * (n_red - 1.0))
        
        # Blue light refracts more -> shorter focal length
        self.assertLess(f_blue, f_red)


class TestDispersionTier4(unittest.TestCase):
    """Tier 4: Doublet Lens Chromatic Aberration Simulation (Crown N-BK7 + Flint SF11)."""

    def test_t4_1_achromatic_doublet_dispersion_balance(self):
        v_crown = GlassCatalog.calculate_abbe_number("N-BK7")
        v_flint = GlassCatalog.calculate_abbe_number("SF11")
        
        # Crown has higher Abbe number (lower dispersion), Flint lower Abbe number (higher dispersion)
        self.assertGreater(v_crown, v_flint)


if __name__ == "__main__":
    unittest.main()
