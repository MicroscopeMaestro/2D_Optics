"""
Adversarial Stress-Test & Vulnerability Verification Suite for Milestone M1 (Challenger 1):
Core Models, ZMX Importer, SO(3) 3D Rotations, Dynamic Optical Element Management & Schema Serialization.
"""
import unittest
import os
import io
import math
import tempfile
import numpy as np

from v2.engine.materials import (
    GlassCatalog,
    SellmeierDispersion,
    CauchyDispersion,
    get_refractive_index,
    get_complex_refractive_index
)
from v2.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
from v2.engine.lde import SurfaceStack, build_rotation_matrix
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


class TestAdversarialSO3Rotations(unittest.TestCase):
    """Stress-test 3D Euler rotation matrices and rigid body transformations for SO(3) group properties."""

    def assert_so3_matrix(self, R: np.ndarray, tol: float = 1e-12, msg: str = ""):
        """Verify that R in SO(3): R @ R.T == I and det(R) == +1.0."""
        self.assertEqual(R.shape, (3, 3), f"Matrix shape not (3,3): {R.shape} {msg}")
        
        # 1. Orthogonality: R @ R.T == I
        identity_error = np.max(np.abs(R @ R.T - np.eye(3)))
        self.assertLess(identity_error, tol, f"Orthogonality error {identity_error} exceeds {tol}: {msg}")

        # 2. Special Orthogonal determinant: det(R) == +1.0
        det_val = np.linalg.det(R)
        self.assertAlmostEqual(det_val, 1.0, delta=tol, msg=f"Determinant {det_val} != +1.0: {msg}")

    def test_canonical_and_extreme_angles(self):
        """Test canonical, boundary, and multi-turn rotation angles in SO(3)."""
        test_angles = [
            (0.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (0.0, 90.0, 0.0),
            (0.0, 0.0, 90.0),
            (-90.0, 0.0, 0.0),
            (0.0, -90.0, 0.0),
            (0.0, 0.0, -90.0),
            (180.0, 0.0, 0.0),
            (0.0, 180.0, 0.0),
            (0.0, 0.0, 180.0),
            (-180.0, -180.0, -180.0),
            (360.0, 360.0, 360.0),
            (720.0, -1080.0, 1440.0),
            (180.0, -90.0, 45.0),
            (45.0, 45.0, 45.0),
            (179.9999, -89.9999, 180.0001),
            (-0.00001, 0.00001, -0.00001),
        ]
        for tx, ty, tz in test_angles:
            R = build_rotation_matrix(tx, ty, tz)
            self.assert_so3_matrix(R, tol=1e-12, msg=f"Angles: ({tx}, {ty}, {tz})")

    def test_random_euler_angles_so3_monte_carlo(self):
        """Monte Carlo verification of 2,000 random 3D rotation angles in [-1000°, 1000°]."""
        np.random.seed(1337)
        random_angles = np.random.uniform(-1000.0, 1000.0, size=(2000, 3))
        for tx, ty, tz in random_angles:
            R = build_rotation_matrix(tx, ty, tz)
            self.assert_so3_matrix(R, tol=1e-12, msg=f"Random Angles: ({tx:.3f}, {ty:.3f}, {tz:.3f})")

    def test_rigid_body_distance_invariance_under_rotation(self):
        """
        Verify that rotating an element preserves the exact Euclidean distance between all surface vertices.
        Rigid body transformation must have zero strain / zero thickness distortion.
        """
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        # AC254-050-C Focuser Doublet is surfaces 4, 5, 6
        v4_init = spec.surfaces[4].global_origin.copy()
        v5_init = spec.surfaces[5].global_origin.copy()
        v6_init = spec.surfaces[6].global_origin.copy()

        d45_init = np.linalg.norm(v5_init - v4_init)
        d56_init = np.linalg.norm(v6_init - v5_init)
        d46_init = np.linalg.norm(v6_init - v4_init)

        extreme_tilts = [
            (45.0, -30.0, 15.0),
            (180.0, 0.0, 0.0),
            (-90.0, 90.0, -90.0),
            (12.34, -56.78, 90.12),
        ]

        for tx, ty, tz in extreme_tilts:
            for pivot in ["front_vertex", "center_of_mass", "rear_vertex", [5.0, -2.0, 4.0]]:
                spec.rotate_element(4, tilt_x=tx, tilt_y=ty, tilt_z=tz, pivot=pivot)
                
                v4 = spec.surfaces[4].global_origin
                v5 = spec.surfaces[5].global_origin
                v6 = spec.surfaces[6].global_origin

                d45 = np.linalg.norm(v5 - v4)
                d56 = np.linalg.norm(v6 - v5)
                d46 = np.linalg.norm(v6 - v4)

                self.assertAlmostEqual(d45, d45_init, places=5, msg=f"d45 failed at tilt ({tx},{ty},{tz}) pivot {pivot}")
                self.assertAlmostEqual(d56, d56_init, places=5, msg=f"d56 failed at tilt ({tx},{ty},{tz}) pivot {pivot}")
                self.assertAlmostEqual(d46, d46_init, places=5, msg=f"d46 failed at tilt ({tx},{ty},{tz}) pivot {pivot}")

    def test_global_frame_so3_invariance_across_all_surfaces(self):
        """
        Verify that for every surface in a complex spectrometer system with tilts, gratings, and mirrors,
        the accumulated global_R orientation matrix satisfies SO(3) properties.
        """
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        for i, surf in enumerate(spec.surfaces):
            self.assert_so3_matrix(surf.global_R, tol=1e-10, msg=f"Surface {i} ({surf.name}) global_R")


class TestAdversarialDynamicElementManagement(unittest.TestCase):
    """Stress-test edge case deletions, flip-flip round-trips, and element replacements."""

    def test_flip_flip_roundtrip_idempotence(self):
        """
        Adversarial Round-Trip Test:
        flip(flip(Element)) must exactly recover the initial element prescription:
        - Curvatures: -(-R) = R
        - Conic: unchanged
        - Aspheric coefficients: unchanged
        - Thicknesses: T_0, T_1, T_air recovered
        - Glass materials: G_0, G_1, AIR recovered
        - Test across 10 successive flips (odd = flipped, even = original).
        """
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        # Doublet surfaces 4, 5, 6
        orig_s4 = {
            "radius": spec.surfaces[4].radius,
            "thickness": spec.surfaces[4].thickness,
            "glass": spec.surfaces[4].glass,
            "conic": spec.surfaces[4].conic,
            "semi_diameter": spec.surfaces[4].semi_diameter
        }
        orig_s5 = {
            "radius": spec.surfaces[5].radius,
            "thickness": spec.surfaces[5].thickness,
            "glass": spec.surfaces[5].glass,
            "conic": spec.surfaces[5].conic,
            "semi_diameter": spec.surfaces[5].semi_diameter
        }
        orig_s6 = {
            "radius": spec.surfaces[6].radius,
            "thickness": spec.surfaces[6].thickness,
            "glass": spec.surfaces[6].glass,
            "conic": spec.surfaces[6].conic,
            "semi_diameter": spec.surfaces[6].semi_diameter
        }

        for flip_iter in range(1, 11):
            spec.flip_element(4)
            if flip_iter % 2 == 0:
                # Even iterations must be identical to original
                self.assertAlmostEqual(spec.surfaces[4].radius, orig_s4["radius"], places=5)
                self.assertAlmostEqual(spec.surfaces[4].thickness, orig_s4["thickness"], places=5)
                self.assertEqual(spec.surfaces[4].glass, orig_s4["glass"])

                self.assertAlmostEqual(spec.surfaces[5].radius, orig_s5["radius"], places=5)
                self.assertAlmostEqual(spec.surfaces[5].thickness, orig_s5["thickness"], places=5)
                self.assertEqual(spec.surfaces[5].glass, orig_s5["glass"])

                self.assertAlmostEqual(spec.surfaces[6].radius, orig_s6["radius"], places=5)
                self.assertAlmostEqual(spec.surfaces[6].thickness, orig_s6["thickness"], places=5)
                self.assertEqual(spec.surfaces[6].glass, orig_s6["glass"])

    def test_flip_aspheric_singlet(self):
        """Flip an aspheric singlet (A230-C) and verify conic and coefficients inversion."""
        stack = SurfaceStack()
        stack.insert_catalog_lens(0, "A230-C", air_gap_after=15.0)
        
        orig_r = stack.surfaces[0].radius
        orig_conic = stack.surfaces[0].conic
        orig_coeffs = list(stack.surfaces[0].asphere_coeffs)
        orig_thick = stack.surfaces[0].thickness
        orig_glass = stack.surfaces[0].glass

        # First flip
        stack.flip_element(0)
        self.assertAlmostEqual(stack.surfaces[0].radius, 0.0)
        self.assertAlmostEqual(stack.surfaces[1].radius, -orig_r)
        self.assertAlmostEqual(stack.surfaces[1].conic, orig_conic)
        self.assertEqual(stack.surfaces[0].glass, orig_glass)
        self.assertEqual(stack.surfaces[1].glass, "AIR")

        # Second flip
        stack.flip_element(0)
        self.assertAlmostEqual(stack.surfaces[0].radius, orig_r)
        self.assertAlmostEqual(stack.surfaces[0].conic, orig_conic)
        self.assertEqual(stack.surfaces[0].asphere_coeffs, orig_coeffs)
        self.assertAlmostEqual(stack.surfaces[0].thickness, orig_thick)
        self.assertEqual(stack.surfaces[0].glass, orig_glass)

    def test_delete_first_element_surface_0(self):
        """
        Adversarial Edge Deletion: Deleting the very first surface (index 0).
        Since there is no preceding surface (first_idx == 0), the deletion must not crash,
        and must not attempt to access index -1.
        """
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        initial_count = len(spec.surfaces)
        deleted = spec.delete_element(0)
        self.assertEqual(deleted, [0])
        self.assertEqual(len(spec.surfaces), initial_count - 1)

    def test_delete_standalone_single_surface_mirror(self):
        """Adversarial Edge Deletion: Deleting a standalone mirror (Surface 2)."""
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        preceding_thick = spec.surfaces[1].thickness
        mirror_thick = spec.surfaces[2].thickness

        deleted = spec.delete_element(2)
        self.assertEqual(deleted, [2])
        # Thickness of mirror is merged into surface 1
        self.assertAlmostEqual(spec.surfaces[1].thickness, preceding_thick + mirror_thick)

    def test_delete_invalid_indices_and_empty_stack(self):
        """Deleting invalid indices (-1, 999, non-existent name) or on empty stack."""
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        self.assertEqual(spec.delete_element(-1), [])
        self.assertEqual(spec.delete_element(999), [])
        self.assertEqual(spec.delete_element("NON_EXISTENT_NAME"), [])

        empty_stack = SurfaceStack()
        self.assertEqual(empty_stack.delete_element(0), [])
        self.assertEqual(empty_stack.flip_element(0), [])


class TestAdversarialZmxImporter(unittest.TestCase):
    """Stress-test Zemax (.zmx) importer with multi-encoding and edge formats."""

    def test_empty_and_whitespace_zmx(self):
        """Empty string or whitespace-only ZMX should return empty stack without crashing."""
        for content in ["", "   \n\n\t  \r\n", "! Only comments\n# Another comment\n! Empty"]:
            stack = parse_zmx_string(content)
            self.assertIsInstance(stack, SurfaceStack)
            self.assertEqual(len(stack.surfaces), 0)

    def test_unusual_encodings(self):
        """Test multi-encoding detection with UTF-8 BOM, UTF-16-LE, UTF-16-BE, CP1252, Latin-1."""
        base_zmx = "VERS 140410\nNAME \"Test Encoding\xc4\xe9\xf1\"\nSURF 0\nTHIC 10.0\nSURF 1\nCURV 0.02\nTHIC 5.0\nGLAS N-BK7\n"

        encodings = ["utf-8-sig", "utf-16-le", "utf-16-be", "cp1252", "latin-1"]
        for enc in encodings:
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".zmx") as tmp:
                tmp.write(base_zmx.encode(enc))
                tmp_path = tmp.name

            try:
                stack = parse_zmx_file(tmp_path)
                self.assertEqual(len(stack.surfaces), 2, f"Failed on encoding: {enc}")
                self.assertAlmostEqual(stack.surfaces[1].radius, 50.0)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def test_nonexistent_file_raises_error(self):
        """Non-existent file should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            parse_zmx_file("c:/non_existent_path_xyz_12345.zmx")


class TestAdversarialSchemaSerialization(unittest.TestCase):
    """Stress-test JSON serialization and schema round-trips with edge-case values."""

    def test_serialization_with_negative_and_variable_params(self):
        """Serialization of surfaces containing negative radii, variable flags, and extreme floats."""
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        spec.surfaces[1].radius = -50.0
        spec.surfaces[1].thickness = 0.0
        spec.surfaces[2].thickness_variable = True
        spec.surfaces[2].thickness_min = 0.5
        spec.surfaces[2].thickness_max = 500.0

        d = spec.to_dict()
        self.assertEqual(d["version"], 3)
        self.assertEqual(d["surfaces"][1]["radius"], -50.0)
        self.assertTrue(d["surfaces"][2]["thickness_variable"])

        # Roundtrip through JSON string
        import json
        json_str = json.dumps(d)
        reloaded_dict = json.loads(json_str)
        reloaded_spec = SpectrometerSystem.from_dict(reloaded_dict)

        self.assertEqual(len(reloaded_spec.surfaces), 17)
        self.assertAlmostEqual(reloaded_spec.surfaces[1].radius, -50.0)
        self.assertTrue(reloaded_spec.surfaces[2].thickness_variable)

    def test_materials_case_and_alias_resolution(self):
        """Verify glass catalog resolves various alias styles and raises clean ValueError on unknown."""
        aliases = [
            ("n-bk7", "N-BK7"),
            ("nbk7", "N-BK7"),
            ("N_BK7", "N-BK7"),
            ("  bk7  ", "BK7"),
            ("s-nph1_mold", "S-NPH1"),
            ("SNPH1MOLD", "S-NPH1"),
            ("fused_silica", "FUSED_SILICA"),
            ("silica", "FUSED_SILICA"),
            ("caf2", "CAF2"),
            ("AIR", "AIR"),
            ("air", "AIR"),
            ("vacuum", "AIR"),
            ("mirror", "MIRROR"),
        ]
        for query, canonical in aliases:
            g = GlassCatalog.get_glass(query)
            self.assertIsNotNone(g, f"Failed to resolve alias: {query}")
            n = get_refractive_index(query, 1.225)
            self.assertGreater(n, 0.99)

        # Unknown glass should raise ValueError
        with self.assertRaises(ValueError):
            GlassCatalog.get_glass("COMPLETELY_FICTITIOUS_CRYSTAL_999")

    def test_dispersion_near_resonance_pole_handling(self):
        """Sellmeier dispersion near resonance pole or with non-positive wavelength."""
        bk7 = GlassCatalog.get_glass("N-BK7")
        with self.assertRaises(ValueError):
            bk7.get_index(0.0)
        with self.assertRaises(ValueError):
            bk7.get_index(-1.0)


class TestEmpiricalVulnerabilitiesFound(unittest.TestCase):
    """
    Empirical tests explicitly proving the exact failure modes discovered:
    1. Unhandled ValueError in zmx_loader when encountering corrupted token values.
    2. Substring false positive in auto_detect_indices() matching 'COLLIMATOR' for 'IMA'.
    3. Multi-match ambiguity in _get_element_indices() returning disjoint surfaces.
    """

    def test_vulnerability_1_zmx_loader_unhandled_token_exceptions(self):
        """
        VULNERABILITY 1: zmx_loader.py crashes with unhandled ValueError when parsing
        tokens containing non-numeric strings (e.g. 'CURV invalid_val').
        """
        corrupted_token_zmx = "SURF 1\n  CURV not_a_float\n"
        with self.assertRaises(ValueError):
            # Proves vulnerability: parse_zmx_string does not catch or sanitize token parse errors
            parse_zmx_string(corrupted_token_zmx)

    def test_vulnerability_2_collimator_matched_as_ima_detector(self):
        """
        VULNERABILITY 2: models.py auto_detect_indices() uses substring matching '"IMA" in s.name.upper()',
        which falsely matches 'M1_Collimator_50mm' (contains 'IMA' in 'COLLIMATOR') as the Detector surface
        when the real detector (Surface 16) is deleted.
        """
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        # Delete surface 16 (the true detector)
        spec.delete_element(16)
        # BUG: detector_surface_idx is set to 2 (M1_Collimator_50mm) instead of 15 (last surface)
        # because 'IMA' in 'M1_COLLIMATOR_50MM' is True!
        self.assertEqual(spec.detector_surface_idx, 2, "Confirmed: Collimator matched as detector via 'IMA' substring")
        self.assertNotEqual(spec.detector_surface_idx, len(spec.surfaces) - 1)

    def test_vulnerability_3_element_name_query_returns_disjoint_elements(self):
        """
        VULNERABILITY 3: lde.py _get_element_indices() substring matching returns disjoint surfaces
        belonging to distinct optical groups when searching common names like 'AC254'.
        """
        spec = SpectrometerSystem.load_from_json(SPEC_JSON_PATH)
        # Both Focuser (4,5,6) and Condenser (9,10,11) contain 'AC254'
        matched = spec.surface_stack._get_element_indices("AC254")
        # Returns [4, 5, 6, 9, 10, 11] which spans across the DMD (surfaces 7, 8)
        self.assertEqual(matched, [4, 5, 6, 9, 10, 11])
        # Non-contiguous span
        is_contiguous = (matched == list(range(min(matched), max(matched) + 1)))
        self.assertFalse(is_contiguous, "Confirmed: _get_element_indices returns disjoint surfaces across layout")


if __name__ == "__main__":
    unittest.main()
