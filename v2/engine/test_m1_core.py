"""
Comprehensive Unit Test & Verification Suite for Milestone M1:
Core Models, ZMX Importer & Dynamic Optical Element Management.
"""
import unittest
import os
import tempfile
import numpy as np

from v2.engine.materials import (
    GlassCatalog,
    SellmeierDispersion,
    CauchyDispersion,
    get_refractive_index,
    get_complex_refractive_index
)
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.models import (
    SpectrometerSystem,
    WavelengthConfig,
    SourceConfig,
    FastTraceResult,
    FullTraceResult
)
from v2.engine.zmx_loader import (
    parse_zmx_string,
    parse_zmx_file,
    export_to_zmx_string,
    export_to_zmx_file,
    import_zmx
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPEC_JSON_PATH = os.path.join(PROJECT_ROOT, "dmd4500_nir_fingerprint_0p8mm_spectrometer.json")


class TestM1SpectrometerModelIngestion(unittest.TestCase):
    """Requirement 1: Spectrometer Model Ingestion without crash or loss of precision."""

    def test_load_dmd4500_nir_spectrometer_json(self):
        """Verify SpectrometerSystem loads dmd4500_nir_fingerprint_0p8mm_spectrometer.json completely."""
        self.assertTrue(os.path.exists(SPEC_JSON_PATH), f"File not found: {SPEC_JSON_PATH}")
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)

        # 1. Surface count
        self.assertEqual(len(spec.surfaces), 17)
        self.assertEqual(len(spec), 17)

        # 2. Spectral configuration
        self.assertEqual(len(spec.wavelengths), 3)
        np.testing.assert_allclose(spec.wavelengths, [1.100, 1.225, 1.350], atol=1e-5)
        self.assertAlmostEqual(spec.primary_wavelength, 1.225, places=4)

        # 3. Component identification
        self.assertEqual(spec.grating_surface_idx, 3)
        self.assertEqual(spec.detector_surface_idx, 16)
        self.assertAlmostEqual(spec.detector_pixel_size_um, 800.0, places=1)

        # 4. Critical surface verification
        # Surface 0: Slit
        s0 = spec.surfaces[0]
        self.assertEqual(s0.surface_type, SurfaceType.OBJECT)
        self.assertAlmostEqual(s0.thickness, 50.0)

        # Surface 2: M1 Collimator Mirror
        s2 = spec.surfaces[2]
        self.assertEqual(s2.glass.upper(), "MIRROR")
        self.assertAlmostEqual(s2.radius, -100.0)
        self.assertAlmostEqual(s2.tilt_x, -8.0)

        # Surface 3: Diffraction Grating
        s3 = spec.surfaces[3]
        self.assertEqual(s3.surface_type, SurfaceType.GRATING)
        self.assertAlmostEqual(s3.grating_lines_per_mm, 600.0)
        self.assertEqual(s3.grating_order, 1)

        # Surface 4-6: AC254-050-C Focuser Doublet
        s4, s5, s6 = spec.surfaces[4], spec.surfaces[5], spec.surfaces[6]
        self.assertEqual(s4.glass.upper(), "N-LAK22")
        self.assertAlmostEqual(s4.radius, 28.3, places=2)
        self.assertAlmostEqual(s4.thickness, 6.0, places=2)
        self.assertEqual(s5.glass.upper(), "N-SF6HT")
        self.assertAlmostEqual(s5.radius, -21.6, places=2)
        self.assertAlmostEqual(s5.thickness, 2.0, places=2)
        self.assertEqual(s6.glass.upper(), "AIR")
        self.assertAlmostEqual(s6.radius, -105.0, places=2)

        # Surface 12: A230-C Molded Asphere
        s12 = spec.surfaces[12]
        self.assertEqual(s12.surface_type, SurfaceType.ASPHERIC)
        self.assertIn("S-NPH1", s12.glass.upper())
        self.assertAlmostEqual(s12.radius, 3.478709, places=5)
        self.assertAlmostEqual(s12.thickness, 2.94, places=2)
        self.assertAlmostEqual(s12.conic, -0.1263039, places=5)
        self.assertGreaterEqual(len(s12.asphere_coeffs), 4)
        self.assertAlmostEqual(s12.asphere_coeffs[1], -0.0012606, places=6)

        # Surface 16: Detector (0.8mm clear aperture, semi_diameter = 0.4mm)
        s16 = spec.surfaces[16]
        self.assertEqual(s16.surface_type, SurfaceType.DETECTOR)
        self.assertAlmostEqual(s16.semi_diameter, 0.4, places=4)

    def test_spectrometer_roundtrip_serialization(self):
        """Verify save_to_json and reload preserves all 17 surfaces and metadata."""
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            tmp_path = tmp.name

        try:
            spec.save_to_json(tmp_path)
            reloaded = SpectrometerSystem.load_from_json(tmp_path)

            self.assertEqual(len(reloaded.surfaces), 17)
            self.assertEqual(reloaded.grating_surface_idx, spec.grating_surface_idx)
            self.assertEqual(reloaded.detector_surface_idx, spec.detector_surface_idx)
            self.assertAlmostEqual(reloaded.primary_wavelength, spec.primary_wavelength)

            for i in range(17):
                orig_s = spec.surfaces[i]
                rel_s = reloaded.surfaces[i]
                self.assertEqual(orig_s.name, rel_s.name)
                self.assertEqual(orig_s.surface_type, rel_s.surface_type)
                self.assertAlmostEqual(orig_s.radius, rel_s.radius, places=5)
                self.assertAlmostEqual(orig_s.thickness, rel_s.thickness, places=5)
                self.assertEqual(orig_s.glass.upper(), rel_s.glass.upper())
                self.assertAlmostEqual(orig_s.semi_diameter, rel_s.semi_diameter, places=5)
                self.assertAlmostEqual(orig_s.tilt_x, rel_s.tilt_x, places=5)
                self.assertAlmostEqual(orig_s.tilt_y, rel_s.tilt_y, places=5)
                self.assertAlmostEqual(orig_s.tilt_z, rel_s.tilt_z, places=5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_dataclasses_wavelength_source_and_trace_results(self):
        """Verify WavelengthConfig, SourceConfig, FastTraceResult, and FullTraceResult."""
        wl_cfg = WavelengthConfig(wavelength_um=1.225, weight=1.0, is_primary=True, color="#FF0000")
        d = wl_cfg.to_dict()
        self.assertEqual(d["wavelength_um"], 1.225)
        self.assertTrue(d["is_primary"])
        wl_restored = WavelengthConfig.from_dict(d)
        self.assertEqual(wl_restored.wavelength_um, 1.225)

        src_cfg = SourceConfig(source_type="Collimated", beam_radius=5.0, num_rays=21)
        src_dict = src_cfg.to_dict()
        self.assertEqual(src_dict["num_rays"], 21)
        src_restored = SourceConfig.from_dict(src_dict)
        self.assertEqual(src_restored.num_rays, 21)

        fast_res = FastTraceResult(
            ray_paths=[np.zeros((3, 3))],
            spot_coordinates={1.225: np.zeros((10, 2))},
            elapsed_ms=12.5,
            wavelengths=[1.225]
        )
        self.assertLess(fast_res.elapsed_ms, 50.0)

        full_res = FullTraceResult(
            ray_paths=[np.zeros((3, 3))],
            spot_coordinates={1.225: np.zeros((100, 2))},
            rms_radius=15.2,
            geo_radius=22.4,
            airy_radius=3.1,
            dispersion_spread=12.0,
            elapsed_ms=120.0
        )
        self.assertGreater(full_res.rms_radius, 0.0)


class TestM1MaterialAndDispersionCatalog(unittest.TestCase):
    """Requirement 2: Material & Dispersion Catalog with Ohara S-NPH1 and no circular imports."""

    def test_catalog_glasses_resolution(self):
        """Verify all required catalog glasses resolve with valid refractive indices."""
        required_glasses = [
            "S-NPH1_MOLD", "S-NPH11_MOLD", "S-NPH1", "S-NPH11",
            "N-LAK22", "N-SF6HT", "N-SF6", "N-BAF10", "B270", "BK7", "N-BK7",
            "FUSED_SILICA", "CAF2", "AIR", "MIRROR", "SF11", "SF5", "SK16"
        ]

        test_wavelengths = [0.5876, 1.100, 1.225, 1.350]
        for g in required_glasses:
            for wl in test_wavelengths:
                n = get_refractive_index(g, wl)
                self.assertIsInstance(n, float)
                if g in ("AIR", "VACUUM", "MIRROR"):
                    self.assertAlmostEqual(n, 1.000277 if g == "AIR" else 1.0, places=4)
                else:
                    self.assertGreater(n, 1.3)
                    self.assertLess(n, 2.5)

    def test_ohara_snph1_properties(self):
        """Verify Ohara S-NPH1 index at visible and NIR wavelengths."""
        n_vis = get_refractive_index("S-NPH1", 0.58756)
        n_nir = get_refractive_index("S-NPH1_MOLD", 1.225)
        # S-NPH1 is a high-index optical glass (n ~ 1.77 - 1.81)
        self.assertGreater(n_vis, 1.75)
        self.assertGreater(n_nir, 1.70)
        self.assertLess(n_nir, n_vis)  # Normal dispersion dn/dlambda < 0

    def test_abbe_number_calculation(self):
        """Verify Abbe number calculation for standard glasses."""
        vd_bk7 = GlassCatalog.calculate_abbe_number("N-BK7")
        self.assertAlmostEqual(vd_bk7, 64.17, places=0)  # Standard N-BK7 Vd ~ 64.17

    def test_complex_refractive_index_metals(self):
        """Verify complex refractive index calculation for dielectric and metallic media."""
        n_al = get_complex_refractive_index("AL", 0.5876)
        self.assertIsInstance(n_al, complex)
        self.assertGreater(n_al.imag, 0.0)

        n_glass = get_complex_refractive_index("N-BK7", 0.5876, extinction_coeff=1e-5)
        self.assertAlmostEqual(n_glass.real, 1.5168, places=2)
        self.assertAlmostEqual(n_glass.imag, 1e-5)


class TestM1ZemaxParser(unittest.TestCase):
    """Requirement 3: Full sequential Zemax token parsing and multi-encoding handling."""

    def test_parse_zmx_tokens_singlet(self):
        """Verify parsing standard sequential tokens on a singlet lens."""
        zmx = """\
VERS 120524 0 0
NAME Thorlabs Singlet Test
UNIT MM
WAVM 1 0.5876 1.0
WAVM 2 1.2250 1.0
SURF 0
  TYPE STANDARD
  THIC 50.0
  DIAM 10.0
SURF 1
  COMM Front_Surface
  TYPE STANDARD
  CURV 0.02
  THIC 6.0
  GLAS N-BK7
  DIAM 12.7
SURF 2
  COMM Back_Surface
  TYPE STANDARD
  CURV -0.02
  THIC 45.0
  GLAS AIR
  DIAM 12.7
SURF 3
  COMM Detector
  TYPE DETECTOR
  THIC 0.0
  DIAM 5.0
"""
        stack = parse_zmx_string(zmx)
        self.assertEqual(len(stack.surfaces), 4)
        self.assertAlmostEqual(stack.surfaces[1].radius, 50.0)
        self.assertEqual(stack.surfaces[1].glass, "N-BK7")
        self.assertEqual(stack.surfaces[1].name, "Front_Surface")
        self.assertAlmostEqual(stack.surfaces[2].radius, -50.0)
        self.assertEqual(stack.surfaces[3].surface_type, SurfaceType.DETECTOR)
        self.assertEqual(len(stack.wavelengths), 2)

    def test_parse_zmx_aspheric_and_grating_tokens(self):
        """Verify parsing EVENASPH, DIFFRACT/GRATING, and coordinate breaks."""
        zmx = """\
SURF 1
  TYPE EVENASPH
  CURV 0.287463
  THIC 2.94
  GLAS S-NPH1_MOLD
  CONI -0.1263039
  PARM 1 0.0
  PARM 2 -0.0012606
  PARM 3 -0.000109
  PARM 4 3.2255631e-07
  PARM 5 -7.8343862e-07
  DIAM 3.17
SURF 2
  TYPE GRATING
  CURV 0.0
  THIC 50.0
  GLAS MIRROR
  PARM 1 600.0
  PARM 2 1
  PARM 3 1.225
SURF 3
  TYPE COORDBRK
  DISP 2.5 -1.5 5.0 -3.0 0.0
"""
        stack = parse_zmx_string(zmx)
        self.assertEqual(len(stack.surfaces), 3)

        # Asphere
        s1 = stack.surfaces[0]
        self.assertEqual(s1.surface_type, SurfaceType.ASPHERIC)
        self.assertAlmostEqual(s1.conic, -0.1263039)
        self.assertAlmostEqual(s1.asphere_coeffs[1], -0.0012606)

        # Grating
        s2 = stack.surfaces[1]
        self.assertEqual(s2.surface_type, SurfaceType.GRATING)
        self.assertAlmostEqual(s2.grating_lines_per_mm, 600.0)
        self.assertEqual(s2.grating_order, 1)

        # Coordinate Break
        s3 = stack.surfaces[2]
        self.assertAlmostEqual(s3.decenter_x, 2.5)
        self.assertAlmostEqual(s3.decenter_y, -1.5)
        self.assertAlmostEqual(s3.tilt_x, 5.0)

    def test_zmx_multi_encoding_auto_detection(self):
        """Verify parse_zmx_file handles utf-8, utf-16, and latin-1 encodings."""
        zmx_content = "VERS 120524\nSURF 0\nTHIC 50.0\nSURF 1\nCURV 0.05\nTHIC 5.0\nGLAS N-BK7\n"

        for enc in ["utf-8", "utf-16", "latin-1"]:
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".zmx") as tmp:
                tmp.write(zmx_content.encode(enc))
                tmp_path = tmp.name

            try:
                stack = parse_zmx_file(tmp_path)
                self.assertEqual(len(stack.surfaces), 2)
                self.assertAlmostEqual(stack.surfaces[1].radius, 20.0)
                self.assertEqual(stack.surfaces[1].glass, "N-BK7")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def test_export_to_zmx_roundtrip(self):
        """Verify export_to_zmx_string and re-import preserves prescription parameters."""
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        zmx_str = export_to_zmx_string(spec.surface_stack, title="Exported NIR Spectrometer")
        self.assertIn("SURF 0", zmx_str)
        self.assertIn("SURF 16", zmx_str)
        self.assertIn("EVENASPH", zmx_str)
        self.assertIn("DIFFRACT", zmx_str)

        reloaded_stack = parse_zmx_string(zmx_str)
        self.assertEqual(len(reloaded_stack.surfaces), 17)
        self.assertAlmostEqual(reloaded_stack.surfaces[3].grating_lines_per_mm, 600.0)


class TestM1DynamicElementManagement(unittest.TestCase):
    """Requirement 4: Dynamic Optical Element Manipulations (insert, delete, flip, rotate, replace)."""

    def setUp(self):
        self.spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)

    def test_insert_element_and_insert_catalog_lens(self):
        """Verify insert_catalog_lens inserts doublet into active sequence."""
        initial_count = len(self.spec.surfaces)
        self.assertEqual(initial_count, 17)

        # Insert Thorlabs AC254-050-C doublet after surface 6
        inserted = self.spec.insert_catalog_lens(6, "AC254-050-C", air_gap_after=25.0)
        self.assertEqual(len(inserted), 3)
        self.assertEqual(inserted, [7, 8, 9])
        self.assertEqual(len(self.spec.surfaces), 20)

        # Check inserted doublet properties
        s_front = self.spec.surfaces[7]
        s_mid = self.spec.surfaces[8]
        s_back = self.spec.surfaces[9]

        self.assertEqual(s_front.glass.upper(), "N-LAK22")
        self.assertAlmostEqual(s_front.thickness, 6.0)
        self.assertEqual(s_mid.glass.upper(), "N-SF6HT")
        self.assertAlmostEqual(s_mid.thickness, 2.0)
        self.assertEqual(s_back.glass.upper(), "AIR")
        self.assertAlmostEqual(s_back.thickness, 25.0)

    def test_delete_element_with_gap_merging(self):
        """Verify delete_element removes all lens surfaces and adds cumulative thickness into preceding air gap."""
        # Insert a singlet after surface 6
        inserted = self.spec.insert_catalog_lens(6, "AC254-050-C", air_gap_after=20.0)
        self.assertEqual(len(self.spec.surfaces), 20)

        preceding_surf_idx = 6
        old_preceding_thick = self.spec.surfaces[preceding_surf_idx].thickness
        elem_total_thick = sum(self.spec.surfaces[k].thickness for k in inserted)  # 6.0 + 2.0 + 20.0 = 28.0

        deleted_indices = self.spec.delete_element(inserted[0])
        self.assertEqual(deleted_indices, [7, 8, 9])
        self.assertEqual(len(self.spec.surfaces), 17)

        # Preceding surface thickness must be increased by exactly elem_total_thick
        new_preceding_thick = self.spec.surfaces[preceding_surf_idx].thickness
        self.assertAlmostEqual(new_preceding_thick, old_preceding_thick + elem_total_thick, places=4)

    def test_flip_element_curvature_and_glass_shift(self):
        """
        Verify flip_element:
        - Inverts curvatures (R -> -R)
        - Shifts glass thicknesses and materials forward (T_{new, k} = T_{old, n-2-k}, G_{new, k} = G_{old, n-2-k})
        - Assigns AIR to the rear with original trailing air gap.
        """
        # Insert AC254-050-C: R=[28.3, -21.6, -105.0], T=[6.0, 2.0, 30.0], G=['N-LAK22', 'N-SF6HT', 'AIR']
        inserted = self.spec.insert_catalog_lens(6, "AC254-050-C", air_gap_after=30.0)

        # Flip the inserted doublet
        flipped = self.spec.flip_element(inserted[0])
        self.assertEqual(flipped, [7, 8, 9])

        s0, s1, s2 = self.spec.surfaces[7], self.spec.surfaces[8], self.spec.surfaces[9]

        # Inverted curvatures: -(-105.0) = 105.0, -(-21.6) = 21.6, -(28.3) = -28.3
        self.assertAlmostEqual(s0.radius, 105.0, places=2)
        self.assertAlmostEqual(s1.radius, 21.6, places=2)
        self.assertAlmostEqual(s2.radius, -28.3, places=2)

        # Shifted thicknesses & glasses
        self.assertAlmostEqual(s0.thickness, 2.0, places=2)
        self.assertEqual(s0.glass.upper(), "N-SF6HT")
        self.assertAlmostEqual(s1.thickness, 6.0, places=2)
        self.assertEqual(s1.glass.upper(), "N-LAK22")
        self.assertAlmostEqual(s2.thickness, 30.0, places=2)  # Trailing air gap preserved
        self.assertEqual(s2.glass.upper(), "AIR")

    def test_rotate_element_front_vertex_pivot(self):
        """Verify 3D rigid body rotation around front_vertex leaves front vertex position stationary."""
        # Focuser doublet is surfaces 4, 5, 6
        self.spec.rotate_element(4, tilt_x=8.0, tilt_y=-4.0, tilt_z=0.0, pivot="front_vertex")

        s4, s5, s6 = self.spec.surfaces[4], self.spec.surfaces[5], self.spec.surfaces[6]
        self.assertAlmostEqual(s4.tilt_x, 8.0)
        self.assertAlmostEqual(s5.tilt_x, 8.0)
        self.assertAlmostEqual(s6.tilt_x, 8.0)
        self.assertAlmostEqual(s4.tilt_y, -4.0)
        self.assertAlmostEqual(s5.tilt_y, -4.0)
        self.assertAlmostEqual(s6.tilt_y, -4.0)

        # Front vertex stays at [0, 0, 0] in local frame
        self.assertAlmostEqual(s4.decenter_x, 0.0, places=5)
        self.assertAlmostEqual(s4.decenter_y, 0.0, places=5)

        # Center distance between front and rear vertex is strictly preserved (8.0 mm glass)
        dist = np.linalg.norm(s6.global_origin - s4.global_origin)
        self.assertAlmostEqual(dist, 8.0, places=4)

    def test_rotate_element_center_of_mass_pivot(self):
        """Verify 3D rigid body rotation around center_of_mass produces symmetric opposing decenters."""
        self.spec.rotate_element(4, tilt_x=10.0, tilt_y=0.0, tilt_z=0.0, pivot="center_of_mass")

        s4, s5, s6 = self.spec.surfaces[4], self.spec.surfaces[5], self.spec.surfaces[6]
        self.assertAlmostEqual(s4.tilt_x, 10.0)
        self.assertAlmostEqual(s6.tilt_x, 10.0)

        # Front and rear vertices displace in opposite directions along Y
        self.assertAlmostEqual(s4.decenter_y, -s6.decenter_y, places=4)

    def test_rotate_element_rear_vertex_pivot(self):
        """Verify 3D rigid body rotation around rear_vertex leaves rear vertex stationary."""
        self.spec.rotate_element(4, tilt_x=5.0, tilt_y=5.0, tilt_z=0.0, pivot="rear_vertex")

        s4, s5, s6 = self.spec.surfaces[4], self.spec.surfaces[5], self.spec.surfaces[6]
        self.assertAlmostEqual(s6.decenter_x, 0.0, places=5)
        self.assertAlmostEqual(s6.decenter_y, 0.0, places=5)

    def test_replace_element(self):
        """Verify replacing focuser doublet with Thorlabs AC254-075-C doublet."""
        replaced = self.spec.surface_stack.replace_lens_from_catalog(4, "AC254-075-C")
        self.assertEqual(len(replaced), 3)
        self.assertEqual(self.spec.surfaces[4].group_name, "AC254-075-C")
        self.assertAlmostEqual(self.spec.surfaces[4].radius, 45.6, places=1)


if __name__ == "__main__":
    unittest.main()
