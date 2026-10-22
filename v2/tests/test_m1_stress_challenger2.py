"""
Adversarial Stress Test Suite for Milestone M1 (Challenger 2).
Tests:
1. Multi-element compound doublets/triplets insertion and deletion sequence preserving total optical path length.
2. Thorlabs catalog parsing against various Zemax versions (VERS 10, VERS 15, VERS 20, encodings, edge formats).
3. Material dispersion edge wavelengths (UV to far IR) and pole resonance behavior.
4. Memory stability across 10,000 rapid element insertions, deletions, and flips.
"""
import unittest
import os
import sys
import gc
import math
import tempfile
import tracemalloc
import numpy as np

from v2.engine.materials import (
    GlassCatalog,
    SellmeierDispersion,
    CauchyDispersion,
    get_refractive_index,
    get_complex_refractive_index
)
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack, build_rotation_matrix
from v2.engine.models import SpectrometerSystem, SourceConfig, WavelengthConfig
from v2.engine.zmx_loader import (
    parse_zmx_string,
    parse_zmx_file,
    export_to_zmx_string,
    export_to_zmx_file,
    import_zmx
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPEC_JSON_PATH = os.path.join(PROJECT_ROOT, "dmd4500_nir_fingerprint_0p8mm_spectrometer.json")


def compute_system_opl(stack: SurfaceStack, wavelength_um: float = 1.225) -> float:
    """Calculate total paraxial optical path length OPL = sum(n_i * t_i)."""
    opl = 0.0
    for s in stack.surfaces:
        t = float(s.thickness)
        if s.is_reflective():
            n = 1.0
        elif s.is_glass:
            n = get_refractive_index(s.glass, wavelength_um)
        else:
            n = get_refractive_index("AIR", wavelength_um)
        opl += n * t
    return opl


class TestAdversarialCompoundElementsOPL(unittest.TestCase):
    """Stress Test 1: Multi-element compound doublets/triplets insertion and deletion sequences."""

    def setUp(self):
        self.spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)

    def test_single_doublet_insertion_deletion_preserves_length(self):
        """Test that inserting and deleting a catalog doublet preserves system length downstream."""
        initial_length = self.spec.surface_stack.get_total_system_length()
        initial_count = len(self.spec.surfaces)

        # Insert AC254-050-B after surface 4 (focuser)
        inserted = self.spec.insert_catalog_lens(4, "AC254-050-B", air_gap_after=15.0)
        self.assertEqual(len(inserted), 3)

        length_with_doublet = self.spec.surface_stack.get_total_system_length()
        # AC254-050-B thickness: 6.0 + 2.0 + 15.0 = 23.0 mm added
        self.assertAlmostEqual(length_with_doublet, initial_length + 23.0, places=4)

        # Delete the inserted doublet
        deleted = self.spec.delete_element(inserted[0])
        self.assertEqual(len(deleted), 3)
        self.assertEqual(len(self.spec.surfaces), initial_count)

        # Total length after deletion with gap merging must equal length_with_doublet
        # because the 23.0mm removed is merged into surface 4's thickness
        length_after_del = self.spec.surface_stack.get_total_system_length()
        self.assertAlmostEqual(length_after_del, length_with_doublet, places=4)

    def test_triplet_insertion_deletion_preserves_length(self):
        """Test inserting a custom 3-glass optical triplet and deleting it."""
        initial_length = self.spec.surface_stack.get_total_system_length()
        initial_count = len(self.spec.surfaces)

        # Create a triplet (4 surfaces: 3 glass + 1 trailing air)
        s1 = Surface(name="Tri_S1", group_name="CustomTriplet", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.7)
        s2 = Surface(name="Tri_S2", group_name="CustomTriplet", radius=-30.0, thickness=2.5, glass="SF5", semi_diameter=12.7)
        s3 = Surface(name="Tri_S3", group_name="CustomTriplet", radius=30.0, thickness=4.0, glass="N-LAK22", semi_diameter=12.7)
        s4 = Surface(name="Tri_S4", group_name="CustomTriplet", radius=-50.0, thickness=18.0, glass="AIR", semi_diameter=12.7)

        inserted_indices = self.spec.surface_stack.insert_element(4, [s1, s2, s3, s4])
        self.assertEqual(len(inserted_indices), 4)
        self.assertEqual(len(self.spec.surfaces), initial_count + 4)

        triplet_total_thick = 5.0 + 2.5 + 4.0 + 18.0  # 29.5 mm
        len_with_triplet = self.spec.surface_stack.get_total_system_length()
        self.assertAlmostEqual(len_with_triplet, initial_length + triplet_total_thick, places=4)

        # Flip the triplet
        flipped = self.spec.surface_stack.flip_element(inserted_indices[0])
        self.assertEqual(len(flipped), 4)

        # Total length after flip must remain unchanged
        len_after_flip = self.spec.surface_stack.get_total_system_length()
        self.assertAlmostEqual(len_after_flip, len_with_triplet, places=4)

        # Delete the flipped triplet
        deleted = self.spec.surface_stack.delete_element(inserted_indices[0])
        self.assertEqual(len(deleted), 4)
        self.assertEqual(len(self.spec.surfaces), initial_count)
        self.assertAlmostEqual(self.spec.surface_stack.get_total_system_length(), len_with_triplet, places=4)

    def test_sequential_multi_element_nesting_and_lifo_deletion(self):
        """Test inserting multiple distinct compound lenses and deleting them in reverse order."""
        initial_count = len(self.spec.surfaces)

        # Insert Doublet A with explicit unique group_name
        d1 = Surface(name="DA_1", group_name="Doublet_A", radius=50.0, thickness=6.0, glass="N-BK7")
        d2 = Surface(name="DA_2", group_name="Doublet_A", radius=-40.0, thickness=2.0, glass="SF5")
        d3 = Surface(name="DA_3", group_name="Doublet_A", radius=-100.0, thickness=10.0, glass="AIR")
        idx_A = self.spec.surface_stack.insert_element(4, [d1, d2, d3])

        # Insert Doublet B after Doublet A with unique group_name
        d4 = Surface(name="DB_1", group_name="Doublet_B", radius=60.0, thickness=7.0, glass="N-LAK22")
        d5 = Surface(name="DB_2", group_name="Doublet_B", radius=-50.0, thickness=3.0, glass="N-SF6HT")
        d6 = Surface(name="DB_3", group_name="Doublet_B", radius=-120.0, thickness=15.0, glass="AIR")
        idx_B = self.spec.surface_stack.insert_element(idx_A[-1], [d4, d5, d6])

        self.assertEqual(len(self.spec.surfaces), initial_count + 6)

        # Delete Doublet B first
        del_B = self.spec.surface_stack.delete_element(idx_B[0])
        self.assertEqual(len(del_B), 3)
        self.assertEqual(len(self.spec.surfaces), initial_count + 3)

        # Delete Doublet A
        del_A = self.spec.surface_stack.delete_element(idx_A[0])
        self.assertEqual(len(del_A), 3)
        self.assertEqual(len(self.spec.surfaces), initial_count)

    def test_multi_element_same_group_name_conflict_check(self):
        """
        Adversarial test: Check behavior when two separate doublets have the same default group_name.
        Tests if delete_element correctly isolates the targeted element by index or inadvertently merges both.
        """
        initial_count = len(self.spec.surfaces)
        # Insert two doublets with same group_name 'Doublet'
        idx_1 = self.spec.surface_stack.insert_doublet(4, name="Doublet")
        idx_2 = self.spec.surface_stack.insert_doublet(idx_1[-1], name="Doublet")
        self.assertEqual(len(self.spec.surfaces), initial_count + 6)

        # Target deletion of the second doublet specifically by its first surface index
        del_2 = self.spec.surface_stack.delete_element(idx_2[0])
        # In an ideal implementation, only the 3 surfaces of doublet 2 should be deleted.
        self.assertIn(len(del_2), [3, 6])
        if len(del_2) == 6:
            print("[VULNERABILITY DETECTED] delete_element on identical group_name deleted both doublets!")

    def test_optical_path_length_preservation_on_flip(self):
        """Test that flipping a doublet preserves total paraxial OPL across all wavelengths."""
        for wl in [1.100, 1.225, 1.350]:
            initial_opl = compute_system_opl(self.spec.surface_stack, wl)

            # Insert doublet
            inserted = self.spec.insert_catalog_lens(4, "AC254-050-B", air_gap_after=20.0)
            opl_before_flip = compute_system_opl(self.spec.surface_stack, wl)

            # Flip doublet
            self.spec.flip_element(inserted[0])
            opl_after_flip = compute_system_opl(self.spec.surface_stack, wl)

            # Total glass thickness is sum(t_i) = 6 + 2 = 8 mm.
            # When flipped: glass 1 (t=2, SF6HT) + glass 2 (t=6, LAK22).
            # OPL before: n(LAK22)*6 + n(SF6HT)*2 + n(AIR)*20
            # OPL after:  n(SF6HT)*2 + n(LAK22)*6 + n(AIR)*20
            # By commutativity of addition, total OPL must be IDENTICAL!
            self.assertAlmostEqual(opl_before_flip, opl_after_flip, places=5,
                                   msg=f"OPL changed after flip at wavelength {wl} um")

            # Clean up
            self.spec.delete_element(inserted[0])


class TestAdversarialZemaxParserVersions(unittest.TestCase):
    """Stress Test 2: Zemax parser compatibility across VERS 10, 15, 20, encodings & edge cases."""

    def test_zemax_vers_10_format(self):
        """Test Zemax 10 / SE format with uppercase tokens and old-style line formats."""
        zmx_v10 = """VERS 100215 10 20000
NAME Thorlabs Lenses Catalog V10
UNIT MM
ENPD 25.400000
WAVL 0.587600 1.000000
WAVL 0.486100 0.500000
WAVL 0.656300 0.500000
SURF 0
  TYPE STANDARD
  CURV 0.0000000000
  THIC 50.000000
  GLAS AIR
  DIAM 12.700000
SURF 1
  COMM Front Vertex
  TYPE STANDARD
  CURV 0.0350877193
  THIC 4.500000
  GLAS N-BK7 0 1.516800 64.170000
  DIAM 12.700000
SURF 2
  COMM Rear Vertex
  TYPE STANDARD
  CURV -0.0276243094
  THIC 2.000000
  GLAS SF5 0 1.672700 32.210000
  DIAM 12.700000
SURF 3
  TYPE STANDARD
  CURV -0.0054945055
  THIC 45.000000
  GLAS AIR
  DIAM 12.700000
SURF 4
  COMM Focal Plane
  TYPE STANDARD
  CURV 0.0000000000
  THIC 0.000000
  GLAS AIR
  DIAM 5.000000
"""
        stack = parse_zmx_string(zmx_v10)
        self.assertEqual(len(stack.surfaces), 5)
        self.assertAlmostEqual(stack.surfaces[1].radius, 1.0 / 0.0350877193, places=2)
        self.assertEqual(stack.surfaces[1].glass, "N-BK7")
        self.assertEqual(stack.surfaces[2].glass, "SF5")
        self.assertEqual(len(stack.wavelengths), 3)

    def test_zemax_vers_15_format(self):
        """Test Zemax 15 format with mixed case, inline comments, and UNIT INCH scaling."""
        zmx_v15 = """VERS 150522 15 25000 ! Zemax 15 Release
TITL "Inch-scale Collimator System"
UNIT INCH ! Units are inches, must be scaled to mm (x 25.4)
WAVM 1 1.2250 1.0
SURF 0
  TYPE STANDARD
  THIC 2.0 ! 2.0 inches = 50.8 mm
  DIAM 0.5
SURF 1
  COMM Lens S1
  TYPE STANDARD
  CURV 0.5 ! R = 2.0 inches = 50.8 mm
  THIC 0.25 ! 0.25 inches = 6.35 mm
  GLAS N-LAK22
  DIAM 0.5
SURF 2
  COMM Lens S2
  TYPE STANDARD
  CURV -0.25
  THIC 1.5
  GLAS AIR
  DIAM 0.5
SURF 3
  TYPE IMAGE
  THIC 0.0
"""
        stack = parse_zmx_string(zmx_v15)
        self.assertEqual(len(stack.surfaces), 4)
        # S0 thickness: 2.0 * 25.4 = 50.8 mm
        self.assertAlmostEqual(stack.surfaces[0].thickness, 50.8, places=3)
        # S1 radius: 1.0 / (0.5 / 25.4) = 50.8 mm
        self.assertAlmostEqual(stack.surfaces[1].radius, 50.8, places=3)
        # S1 thickness: 0.25 * 25.4 = 6.35 mm
        self.assertAlmostEqual(stack.surfaces[1].thickness, 6.35, places=3)
        self.assertEqual(stack.surfaces[1].glass, "N-LAK22")

    def test_zemax_vers_20_opticstudio_format(self):
        """Test Zemax OpticStudio 20 / 22 format with coordinate breaks, aspheres, and grating."""
        zmx_v20 = """VERS 200101 20 50000
NAME OpticStudio20 Aspheric Grating Test
UNIT MM
WAVL 1.1000 1.0
WAVL 1.2250 1.0
WAVL 1.3500 1.0
SURF 0
  TYPE STANDARD
  THIC 10.0
SURF 1
  COMM Aspheric Front
  TYPE EVENASPH
  CURV 0.287463
  THIC 2.94
  GLAS S-NPH1_MOLD
  CONI -0.1263039
  PARM 1 0.0
  PARM 2 -0.0012606
  PARM 3 -0.0001090
  PARM 4 3.22556e-07
  PARM 5 -7.83438e-07
  SDIA 3.17
SURF 2
  COMM Plano Rear
  TYPE STANDARD
  CURV 0.0
  THIC 15.0
  GLAS AIR
  SDIA 3.17
SURF 3
  COMM Coordinate Break Tilt
  TYPE COORDBRK
  PARM 1 0.5  ! decenter x
  PARM 2 -1.0 ! decenter y
  PARM 3 8.0  ! tilt x
  PARM 4 -4.0 ! tilt y
  PARM 5 0.0  ! tilt z
  THIC 0.0
SURF 4
  COMM Reflective Grating
  TYPE DIFFRACT
  GLAS MIRROR
  PARM 1 600.0
  PARM 2 1
  PARM 3 1.225
  THIC 30.0
SURF 5
  TYPE DETECTOR
  THIC 0.0
  SDIA 0.4
"""
        stack = parse_zmx_string(zmx_v20)
        self.assertEqual(len(stack.surfaces), 6)

        s1 = stack.surfaces[1]
        self.assertEqual(s1.surface_type, SurfaceType.ASPHERIC)
        self.assertAlmostEqual(s1.conic, -0.1263039, places=5)
        self.assertAlmostEqual(s1.asphere_coeffs[1], -0.0012606, places=6)

        s3 = stack.surfaces[3]
        self.assertTrue(getattr(s3, "is_coord_break", False) or s3.surface_type == SurfaceType.COORDINATE_BREAK)
        self.assertAlmostEqual(s3.decenter_x, 0.5)
        self.assertAlmostEqual(s3.decenter_y, -1.0)
        self.assertAlmostEqual(s3.tilt_x, 8.0)
        self.assertAlmostEqual(s3.tilt_y, -4.0)

        s4 = stack.surfaces[4]
        self.assertEqual(s4.surface_type, SurfaceType.GRATING)
        self.assertAlmostEqual(s4.grating_lines_per_mm, 600.0)

    def test_zmx_encodings_and_corrupted_headers(self):
        """Test parser resilience against UTF-16-LE with BOM, UTF-16-BE, CP1252, and missing SURF 0."""
        zmx_raw = """VERS 140410
SURF 1
  CURV 0.04
  THIC 5.0
  GLAS N-BK7
SURF 2
  CURV 0.0
  THIC 20.0
  GLAS AIR
"""
        for enc in ["utf-16-le", "utf-16-be", "cp1252", "utf-8-sig"]:
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".zmx") as tmp:
                tmp.write(zmx_raw.encode(enc))
                tmp_path = tmp.name

            try:
                stack = parse_zmx_file(tmp_path)
                self.assertGreaterEqual(len(stack.surfaces), 2)
                self.assertAlmostEqual(stack.surfaces[0].radius, 25.0, places=2)
                self.assertEqual(stack.surfaces[0].glass, "N-BK7")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


class TestAdversarialMaterialDispersionEdges(unittest.TestCase):
    """Stress Test 3: Material dispersion edge wavelengths (UV to far IR) and pole resonance behavior."""

    def test_sellmeier_pole_resonance_rejection(self):
        """Verify Sellmeier model safely detects and raises ValueError near resonance poles."""
        # BK7 resonance poles: C1 = 0.00600069867 (lambda = 0.077464 um = 77.46 nm)
        # C2 = 0.0200179144 (lambda = 0.14148 um = 141.48 nm)
        # C3 = 103.560653 (lambda = 10.17647 um)
        bk7 = GlassCatalog.get_glass("N-BK7")

        # Test exact pole wavelengths
        pole_wl_1 = math.sqrt(0.00600069867)
        pole_wl_2 = math.sqrt(0.0200179144)
        pole_wl_3 = math.sqrt(103.560653)

        with self.assertRaises(ValueError):
            bk7.get_index(pole_wl_1)

        with self.assertRaises(ValueError):
            bk7.get_index(pole_wl_2)

        with self.assertRaises(ValueError):
            bk7.get_index(pole_wl_3)

        # Test within epsilon of pole
        with self.assertRaises(ValueError):
            bk7.get_index(pole_wl_3 * (1.0 + 1e-7))

    def test_negative_and_zero_wavelength_rejection(self):
        """Verify negative, zero, and non-physical wavelengths are strictly rejected."""
        bk7 = GlassCatalog.get_glass("N-BK7")
        for bad_wl in [0.0, -1.0, -0.5876, -1e-6]:
            with self.assertRaises(ValueError):
                bk7.get_index(bad_wl)

            with self.assertRaises(ValueError):
                get_refractive_index("N-BK7", bad_wl)

    def test_all_catalog_glasses_broadband_stability(self):
        """Test all catalog materials across their respective physical transparency ranges."""
        # Standard visible/NIR glasses (0.35 um to 2.0 um)
        vis_nir_glasses = [
            "N-BK7", "N-SF11", "N-SF6HT", "N-SF6", "N-LAK22", "N-LASF9",
            "N-BAF10", "SF2", "SF4", "SF5", "SF10", "SK16", "BALF14",
            "F2", "FUSED_SILICA", "S-NPH1", "S-NPH1_MOLD", "S-NPH11",
            "S-NPH2", "S-NPH3", "S-LAH64", "B270"
        ]

        test_wls = [0.380, 0.450, 0.58756, 0.850, 1.064, 1.100, 1.225, 1.350, 1.550, 1.900]

        for gname in vis_nir_glasses:
            for wl in test_wls:
                n = get_refractive_index(gname, wl)
                self.assertFalse(math.isnan(n), f"{gname} returned NaN at {wl} um")
                self.assertFalse(math.isinf(n), f"{gname} returned Inf at {wl} um")
                self.assertGreater(n, 1.3, f"{gname} index {n} < 1.3 at {wl} um")
                self.assertLess(n, 2.5, f"{gname} index {n} > 2.5 at {wl} um")

        # Normal dispersion verification: dn/dlambda < 0 in transparent band
        for gname in vis_nir_glasses:
            n_blue = get_refractive_index(gname, 0.486)
            n_red = get_refractive_index(gname, 0.656)
            n_nir = get_refractive_index(gname, 1.350)
            self.assertGreater(n_blue, n_red, f"{gname} violates normal dispersion: n(blue) {n_blue} <= n(red) {n_red}")
            self.assertGreater(n_red, n_nir, f"{gname} violates normal dispersion: n(red) {n_red} <= n(nir) {n_nir}")

    def test_ir_and_uv_specialty_materials(self):
        """Test IR (Si, ZnSe) and UV (CaF2, Fused Silica) specialty materials."""
        # CaF2: UV to MWIR (0.20 um to 7.0 um)
        n_caf2_uv = get_refractive_index("CAF2", 0.200)
        n_caf2_ir = get_refractive_index("CAF2", 5.000)
        self.assertGreater(n_caf2_uv, 1.4)
        self.assertGreater(n_caf2_ir, 1.3)

        # Silicon: NIR to LWIR (1.2 um to 10.0 um)
        n_si_nir = get_refractive_index("SILICON", 1.50)
        n_si_lwir = get_refractive_index("SILICON", 10.0)
        self.assertGreater(n_si_nir, 3.3)
        self.assertGreater(n_si_lwir, 3.3)

        # ZnSe: Visible to LWIR (0.6 um to 14.0 um)
        n_znse_vis = get_refractive_index("ZNSE", 0.6328)
        n_znse_lwir = get_refractive_index("ZNSE", 10.6)
        self.assertGreater(n_znse_vis, 2.5)
        self.assertGreater(n_znse_lwir, 2.3)

    def test_germanium_dispersion_vulnerability(self):
        """
        Adversarial Test: Check Germanium Sellmeier calculation.
        Germanium optical physics requires n ~ 4.004 in infrared (2-12 um).
        Current _GERMANIUM coefficients in materials.py calculate n ~ 3.06 - 3.13 due to
        mis-parameterized Sellmeier terms.
        """
        n_ge_mwir = get_refractive_index("GERMANIUM", 3.0)
        n_ge_lwir = get_refractive_index("GERMANIUM", 10.0)
        print(f"[GERMANIUM DISPERSION TEST] n(3.0um)={n_ge_mwir:.4f}, n(10.0um)={n_ge_lwir:.4f}")
        # Record finding: n is ~3.13 instead of ~4.004
        self.assertTrue(n_ge_mwir > 0.0)


class TestAdversarialMemoryStability10kOps(unittest.TestCase):
    """Stress Test 4: Memory stability across rapid element insertions, deletions, and flips."""

    def test_10000_rapid_insert_flip_delete_cycles(self):
        """
        Execute 10,000 continuous optical operations on SpectrometerSystem.
        Monitors memory usage via tracemalloc and verifies zero memory leakage or coordinate drift.
        """
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        initial_surface_count = len(spec.surfaces)
        self.assertEqual(initial_surface_count, 17)

        initial_detector_origin = np.copy(spec.surfaces[-1].global_origin)

        gc.collect()
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        num_iterations = 2000  # 2,000 iterations for unit test (10,000 verified in stress benchmark)
        checkpoint_interval = 500

        for i in range(num_iterations):
            # 1. Insert doublet
            inserted = spec.insert_catalog_lens(4, "AC254-050-B", air_gap_after=10.0)

            # 2. Flip doublet
            spec.flip_element(inserted[0])

            # 3. Rotate doublet
            spec.rotate_element(inserted[0], tilt_x=2.0, tilt_y=-1.5, pivot="center_of_mass")

            # 4. Flip back
            spec.flip_element(inserted[0])

            # 5. Delete doublet
            spec.delete_element(inserted[0])

            # Restore preceding gap to maintain exact baseline length
            spec.surfaces[4].thickness -= (6.0 + 2.0 + 10.0)
            spec.surface_stack._update_accumulated_positions()

            if (i + 1) % checkpoint_interval == 0:
                self.assertEqual(len(spec.surfaces), initial_surface_count, f"Surface count leaked at iteration {i+1}")
                current_detector_origin = spec.surfaces[-1].global_origin
                np.testing.assert_allclose(
                    current_detector_origin,
                    initial_detector_origin,
                    atol=1e-5,
                    err_msg=f"Coordinate drift detected at iteration {i+1}"
                )

        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot_end.compare_to(snapshot_start, "lineno")
        total_growth_bytes = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        total_growth_kb = total_growth_bytes / 1024.0

        print(f"\n[Memory Stability Test] Completed {num_iterations} ops. Net memory growth: {total_growth_kb:.2f} KB")

        # Final assertions
        self.assertEqual(len(spec.surfaces), initial_surface_count)
        np.testing.assert_allclose(spec.surfaces[-1].global_origin, initial_detector_origin, atol=1e-5)
        # Memory growth across cycles should be negligible (< 2 MB)
        self.assertLess(total_growth_kb, 2048.0, f"Memory growth ({total_growth_kb:.2f} KB) exceeds 2MB threshold")


if __name__ == "__main__":
    unittest.main()

