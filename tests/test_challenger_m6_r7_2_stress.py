"""
Adversarial Stress Test Suite for LensDataTableModel Row Operations & Metadata Preservation.
Milestone: M6 (Resumed Gen 7 Gate Evaluation) - Challenger 2.

Verifies 100% preservation of surface object identity (id(surf)) and complex surface metadata:
- asphere_coeffs
- cad_profile
- jones_matrix
- grating_order
- is_active
- extinction_coeff
- aperture_type & aperture_params
across moveRow, insertRows, removeRows, setData, set_surface_stack, and randomized operation sequences.
"""

import unittest
import random
import numpy as np

from optics_app.gui.lde_table import LensDataTableModel
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack


def make_test_surface(tag: str, idx: int) -> Surface:
    """Construct a Surface with unique complex metadata."""
    return Surface(
        name=f"S_{tag}_{idx}",
        is_active=(idx % 2 == 0),
        surface_type=SurfaceType.ASPHERIC if idx % 2 == 1 else SurfaceType.GRATING,
        radius=100.0 + idx * 10,
        thickness=5.0 + idx,
        glass="N-BK7" if idx % 2 == 0 else "FS",
        semi_diameter=20.0 + idx,
        conic=-0.5 - idx * 0.1,
        asphere_coeffs=[1e-4 * (idx + 1), -2e-6 * (idx + 1), 3e-8 * (idx + 1)],
        decenter_x=0.1 * idx,
        decenter_y=0.2 * idx,
        tilt_x=0.5 * idx,
        tilt_y=0.6 * idx,
        tilt_z=0.7 * idx,
        grating_lines_per_mm=600.0 + idx * 50,
        grating_order=idx - 2,
        cad_profile=f"cad_model_{tag}_{idx}",
        extinction_coeff=0.001 * (idx + 1),
        aperture_type="Spider" if idx % 2 == 0 else "Rectangular",
        aperture_center_radius=3.0 + idx,
        aperture_spider_width=0.5 + idx * 0.1,
        aperture_spider_count=4 + idx,
        aperture_spider_angle=15.0 * idx,
        jones_matrix=np.array([[1.0 + 0.1j * idx, 0.0], [0.0, 0.9 - 0.1j * idx]], dtype=complex)
    )


class TestChallengerM6R72Stress(unittest.TestCase):
    """Adversarial stress tests for LensDataTableModel row operations & surface metadata binding."""

    def assert_surface_metadata_intact(self, surf: Surface, tag: str, idx: int):
        """Assert all complex metadata attributes are accurately retained."""
        self.assertEqual(surf.is_active, (idx % 2 == 0), f"is_active corrupted for {tag}_{idx}")
        self.assertEqual(surf.asphere_coeffs, [1e-4 * (idx + 1), -2e-6 * (idx + 1), 3e-8 * (idx + 1)], f"asphere_coeffs corrupted for {tag}_{idx}")
        self.assertEqual(surf.grating_order, idx - 2, f"grating_order corrupted for {tag}_{idx}")
        self.assertEqual(surf.cad_profile, f"cad_model_{tag}_{idx}", f"cad_profile corrupted for {tag}_{idx}")
        self.assertAlmostEqual(surf.extinction_coeff, 0.001 * (idx + 1), places=9, msg=f"extinction_coeff corrupted for {tag}_{idx}")
        self.assertIsNotNone(surf.jones_matrix, f"jones_matrix is None for {tag}_{idx}")
        expected_jm = np.array([[1.0 + 0.1j * idx, 0.0], [0.0, 0.9 - 0.1j * idx]], dtype=complex)
        np.testing.assert_array_almost_equal(surf.jones_matrix, expected_jm, err_msg=f"jones_matrix values corrupted for {tag}_{idx}")

    def test_move_row_permutations_and_metadata(self):
        """Verify moveRow, move_surface_up, move_surface_down, and move_row preserve identity & metadata."""
        stack = SurfaceStack()
        surfaces = [make_test_surface("P1", i) for i in range(5)]
        for s in surfaces:
            stack.add_surface(s)

        model = LensDataTableModel(stack)
        orig_ids = [id(s) for s in surfaces]

        # 1. moveRow(0 -> 3)
        self.assertTrue(model.moveRow(None, 0, None, 3))
        st = model.get_surface_stack()
        expected_ids = [orig_ids[1], orig_ids[2], orig_ids[0], orig_ids[3], orig_ids[4]]
        self.assertEqual([id(s) for s in st.surfaces], expected_ids)
        self.assert_surface_metadata_intact(st.surfaces[2], "P1", 0)

        # 2. move_surface_up(2) -> index 2 moves to index 1
        self.assertTrue(model.move_surface_up(2))
        st = model.get_surface_stack()
        expected_ids_2 = [orig_ids[1], orig_ids[0], orig_ids[2], orig_ids[3], orig_ids[4]]
        self.assertEqual([id(s) for s in st.surfaces], expected_ids_2)
        self.assert_surface_metadata_intact(st.surfaces[1], "P1", 0)

        # 3. move_surface_down(0) -> index 0 moves to index 1
        self.assertTrue(model.move_surface_down(0))
        st = model.get_surface_stack()
        self.assertEqual([id(s) for s in st.surfaces], orig_ids)
        self.assert_surface_metadata_intact(st.surfaces[0], "P1", 0)

        # 4. move_row(4, 0) -> row 4 moves to row 0
        self.assertTrue(model.move_row(4, 0))
        st = model.get_surface_stack()
        expected_ids_4 = [orig_ids[4], orig_ids[0], orig_ids[1], orig_ids[2], orig_ids[3]]
        self.assertEqual([id(s) for s in st.surfaces], expected_ids_4)
        self.assert_surface_metadata_intact(st.surfaces[0], "P1", 4)

    def test_insert_rows_and_metadata(self):
        """Verify insertRows shifts existing surface objects and preserves metadata."""
        stack = SurfaceStack()
        surfaces = [make_test_surface("INS", i) for i in range(3)]
        for s in surfaces:
            stack.add_surface(s)

        model = LensDataTableModel(stack)
        orig_ids = [id(s) for s in surfaces]

        self.assertTrue(model.insertRows(1, 2))
        st = model.get_surface_stack()
        self.assertEqual(len(st.surfaces), 5)

        # Original surfaces are at indices 0, 3, 4
        actual_ids = [id(s) for s in st.surfaces]
        self.assertEqual(actual_ids[0], orig_ids[0])
        self.assertEqual(actual_ids[3], orig_ids[1])
        self.assertEqual(actual_ids[4], orig_ids[2])

        self.assert_surface_metadata_intact(st.surfaces[0], "INS", 0)
        self.assert_surface_metadata_intact(st.surfaces[3], "INS", 1)
        self.assert_surface_metadata_intact(st.surfaces[4], "INS", 2)

    def test_remove_rows_and_metadata(self):
        """Verify removeRows pops target surfaces and preserves remaining metadata."""
        stack = SurfaceStack()
        surfaces = [make_test_surface("DEL", i) for i in range(6)]
        for s in surfaces:
            stack.add_surface(s)

        model = LensDataTableModel(stack)
        orig_ids = [id(s) for s in surfaces]

        # Remove 2 rows at index 2 (surfaces 2 and 3)
        self.assertTrue(model.removeRows(2, 2))
        st = model.get_surface_stack()
        self.assertEqual(len(st.surfaces), 4)

        expected_ids = [orig_ids[0], orig_ids[1], orig_ids[4], orig_ids[5]]
        self.assertEqual([id(s) for s in st.surfaces], expected_ids)

        self.assert_surface_metadata_intact(st.surfaces[0], "DEL", 0)
        self.assert_surface_metadata_intact(st.surfaces[1], "DEL", 1)
        self.assert_surface_metadata_intact(st.surfaces[2], "DEL", 4)
        self.assert_surface_metadata_intact(st.surfaces[3], "DEL", 5)

    def test_set_data_cell_edit_preserves_metadata(self):
        """Verify cell modifications via setData do not recreate Surface objects or clear metadata."""
        stack = SurfaceStack()
        s0 = make_test_surface("EDIT", 0)
        stack.add_surface(s0)
        model = LensDataTableModel(stack)

        orig_id = id(model.get_surface_stack().surfaces[0])

        self.assertTrue(model.setData(0, 2, 145.0))  # Radius
        self.assertTrue(model.setData(0, 3, 14.2))   # Thickness
        self.assertTrue(model.setData(0, 6, -1.2))   # Conic

        st = model.get_surface_stack()
        surf = st.surfaces[0]
        self.assertEqual(id(surf), orig_id)
        self.assertEqual(surf.radius, 145.0)
        self.assertEqual(surf.thickness, 14.2)
        self.assertEqual(surf.conic, -1.2)
        self.assert_surface_metadata_intact(surf, "EDIT", 0)

    def test_randomized_high_volume_stress_sequence(self):
        """Randomized stress test performing 100 sequential table operations."""
        random.seed(12345)
        model = LensDataTableModel()

        tracked_surfaces = []
        stack = SurfaceStack()
        for i in range(12):
            s = make_test_surface("RND", i)
            tracked_surfaces.append(s)
            stack.add_surface(s)
        model.set_surface_stack(stack)

        for step in range(100):
            n = len(tracked_surfaces)
            op = random.choice(["move", "move_up", "move_down", "insert", "remove", "edit"])

            if op == "move" and n > 1:
                src = random.randint(0, n - 1)
                dst = random.randint(0, n)
                if dst != src and dst != src + 1:
                    elem = tracked_surfaces.pop(src)
                    actual_dst = dst if dst < src else dst - 1
                    tracked_surfaces.insert(actual_dst, elem)
                self.assertTrue(model.moveRow(None, src, None, dst))

            elif op == "move_up" and n > 1:
                row = random.randint(1, n - 1)
                elem = tracked_surfaces.pop(row)
                tracked_surfaces.insert(row - 1, elem)
                self.assertTrue(model.move_surface_up(row))

            elif op == "move_down" and n > 1:
                row = random.randint(0, n - 2)
                elem = tracked_surfaces.pop(row)
                tracked_surfaces.insert(row + 1, elem)
                self.assertTrue(model.move_surface_down(row))

            elif op == "insert":
                pos = random.randint(0, n)
                self.assertTrue(model.insertRows(pos, 1))
                inserted_surf = model._surfaces[pos]
                tracked_surfaces.insert(pos, inserted_surf)

            elif op == "remove" and n > 3:
                pos = random.randint(0, n - 1)
                tracked_surfaces.pop(pos)
                self.assertTrue(model.removeRows(pos, 1))

            elif op == "edit" and n > 0:
                row = random.randint(0, n - 1)
                self.assertTrue(model.setData(row, 2, float(random.randint(10, 300))))

            st = model.get_surface_stack()
            actual = list(st.surfaces)
            self.assertEqual(len(actual), len(tracked_surfaces), f"Length mismatch at step {step}")
            for r_idx, (act, exp) in enumerate(zip(actual, tracked_surfaces)):
                self.assertIs(act, exp, f"Object identity mismatch at step {step}, row {r_idx}")
                if hasattr(exp, 'cad_profile') and str(exp.cad_profile).startswith("cad_model_RND_"):
                    orig_i = int(str(exp.cad_profile).split("_")[-1])
                    self.assert_surface_metadata_intact(act, "RND", orig_i)

    def test_out_of_bounds_and_edge_cases(self):
        """Verify invalid row move/remove operations fail gracefully without altering model."""
        model = LensDataTableModel()
        initial_count = model.rowCount()

        self.assertFalse(model.moveRow(None, -1, None, 1))
        self.assertFalse(model.moveRow(None, 0, None, 99))
        self.assertFalse(model.removeRows(-1, 1))
        self.assertFalse(model.removeRows(99, 1))
        self.assertFalse(model.move_surface_up(0))
        self.assertFalse(model.move_surface_down(initial_count - 1))

        self.assertEqual(model.rowCount(), initial_count)


if __name__ == "__main__":
    unittest.main()
