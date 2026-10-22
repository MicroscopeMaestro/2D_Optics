"""Tests for Thorlabs Multi-Diameter Achromat Catalog and Element Replacement/Insertion.

Verifies:
1. Availability of standard Thorlabs diameters:
   - Miniature (5.0, 6.0, 6.35, 8.0 mm)
   - 1/2 inch (12.7 mm)
   - 1 inch (25.4 mm)
   - 30.0 mm
   - 2 inch (50.8 mm)
   - 3 inch (75.0 mm)
2. Availability across Visible (-A), NIR (-B), and IR (-C) coatings.
3. Element replacement and insertion with multi-diameter lenses in SurfaceStack.
4. CatalogLensDialog 7-column table, diameter filter, coating filter, and count badge.
"""

import pytest
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.catalogs.thorlabs import (
    get_catalog_sections, get_lens_catalog, get_catalog_lens, query_catalog
)
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.lde_table import LensDataTableModel
from optics_app.gui.catalog_lens_dialog import CatalogLensDialog


@pytest.fixture(autouse=True)
def ensure_qt_app():
    if QT_AVAILABLE:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
    yield


class TestMultiDiameterCatalogCoverage:
    def test_all_standard_diameters_present(self):
        cat = get_lens_catalog()
        diameters = {rec["diameter_mm"] for rec in cat.values()}

        # Verify presence of all standard commercial Thorlabs diameters
        assert 5.0 in diameters
        assert 6.0 in diameters
        assert 6.35 in diameters
        assert 8.0 in diameters
        assert 12.7 in diameters
        assert 25.4 in diameters
        assert 30.0 in diameters
        assert 50.8 in diameters
        assert 75.0 in diameters

    def test_coating_series_coverage(self):
        # A-coating (400-700 nm Visible)
        a_lenses = query_catalog(coating_code="A")
        assert len(a_lenses) >= 30
        a_diams = {rec["diameter_mm"] for rec in a_lenses}
        assert {12.7, 25.4, 30.0, 50.8, 75.0}.issubset(a_diams)

        # B-coating (650-1050 nm NIR)
        b_lenses = query_catalog(coating_code="B")
        assert len(b_lenses) >= 30
        b_diams = {rec["diameter_mm"] for rec in b_lenses}
        assert {12.7, 25.4, 30.0, 50.8, 75.0}.issubset(b_diams)

        # C-coating (1050-1700 nm IR)
        c_lenses = query_catalog(coating_code="C")
        assert len(c_lenses) >= 29
        c_diams = {rec["diameter_mm"] for rec in c_lenses}
        assert {5.0, 6.0, 6.35, 8.0, 12.7, 25.4, 30.0, 50.8}.issubset(c_diams)

    def test_specific_multi_diameter_part_numbers(self):
        parts_to_verify = [
            ("AC050-010-B", 5.0, "B"),
            ("AC064-013-A", 6.35, "A"),
            ("AC080-016-C", 8.0, "C"),
            ("AC127-025-A", 12.7, "A"),
            ("AC127-050-B", 12.7, "B"),
            ("AC127-075-C", 12.7, "C"),
            ("AC254-050-A", 25.4, "A"),
            ("AC254-100-B", 25.4, "B"),
            ("AC254-200-C", 25.4, "C"),
            ("AC300-050-A", 30.0, "A"),
            ("AC300-080-B", 30.0, "B"),
            ("AC300-100-C", 30.0, "C"),
            ("AC508-100-A", 50.8, "A"),
            ("AC508-150-B", 50.8, "B"),
            ("AC508-200-C", 50.8, "C"),
            ("AC750-200-A", 75.0, "A"),
            ("AC750-300-B", 75.0, "B"),
        ]
        for pn, expected_diam, expected_coat in parts_to_verify:
            rec = get_catalog_lens(pn)
            assert rec is not None, f"Part {pn} should exist in catalog"
            assert pytest.approx(rec["diameter_mm"], abs=0.01) == expected_diam
            assert rec["coating_code"] == expected_coat


class TestMultiDiameterInsertAndReplace:
    def test_insert_different_diameter_lenses(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=30.0, glass="AIR"))

        # 1. Insert 1/2" doublet (AC127-025-A)
        indices_127 = stack.insert_catalog_lens(0, "AC127-025-A", air_gap_after=15.0)
        assert len(indices_127) == 3
        assert pytest.approx(stack.surfaces[indices_127[0]].semi_diameter) == 6.35
        assert stack.surfaces[indices_127[2]].thickness == 15.0

        # 2. Insert 2" doublet (AC508-150-B) after the first lens
        indices_508 = stack.insert_catalog_lens(indices_127[-1], "AC508-150-B", air_gap_after=25.0)
        assert len(indices_508) == 3
        assert pytest.approx(stack.surfaces[indices_508[0]].semi_diameter) == 25.4
        assert stack.surfaces[indices_508[2]].thickness == 25.0

        # 3. Insert 30 mm doublet (AC300-080-A) after the second lens
        indices_300 = stack.insert_catalog_lens(indices_508[-1], "AC300-080-A", air_gap_after=40.0)
        assert len(indices_300) == 3
        assert pytest.approx(stack.surfaces[indices_300[0]].semi_diameter) == 15.0

    def test_replace_with_different_diameter_lenses(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=25.0, glass="AIR"))
        # Initial 1" lens
        initial_indices = stack.insert_catalog_lens(0, "AC254-050-C", air_gap_after=35.0)
        stack.add_surface(Surface(name="DET", thickness=0.0, glass="AIR"))

        # Replace 1" lens with 1/2" lens (AC127-030-B)
        rep1 = stack.replace_lens_from_catalog(
            target=initial_indices[0],
            part_number="AC127-030-B",
            air_gap_after=None,  # preserve existing air gap (35.0)
        )
        assert len(rep1) == 3
        assert pytest.approx(stack.surfaces[rep1[0]].semi_diameter) == 6.35
        assert stack.surfaces[rep1[2]].thickness == 35.0

        # Replace 1/2" lens with 2" lens (AC508-100-A) in reverse orientation
        rep2 = stack.replace_lens_from_catalog(
            target=rep1[0],
            part_number="AC508-100-A",
            reverse=True,
            air_gap_after=42.0,
        )
        assert len(rep2) == 3
        assert pytest.approx(stack.surfaces[rep2[0]].semi_diameter) == 25.4
        assert stack.surfaces[rep2[2]].thickness == 42.0

        # Replace with 3" lens (AC750-200-B)
        rep3 = stack.replace_lens_from_catalog(
            target=rep2[0],
            part_number="AC750-200-B",
            reverse=False,
            air_gap_after=50.0,
        )
        assert len(rep3) == 3
        assert pytest.approx(stack.surfaces[rep3[0]].semi_diameter) == 37.5
        assert stack.surfaces[rep3[2]].thickness == 50.0


class TestCatalogLensDialogUI:
    def test_dialog_table_columns_and_filters(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=20.0, glass="AIR"))
        stack.insert_catalog_lens(0, "AC254-050-C", air_gap_after=20.0)
        model = LensDataTableModel(stack)

        dialog = CatalogLensDialog(model=model, mode="replace")

        # Verify 7 columns
        assert dialog.lens_table.columnCount() == 7
        headers = [dialog.lens_table.horizontalHeaderItem(c).text() for c in range(7)]
        assert headers == ["Part Number", "Type / Family", "Coating", "Ø (mm)", "EFL (mm)", "BFL (mm)", "Power (D)"]

        # Verify coating combobox exists
        assert hasattr(dialog, "filter_coating")
        assert dialog.filter_coating.count() >= 5

        # Verify item count status label exists and shows non-zero count
        assert hasattr(dialog, "lbl_catalog_count")
        assert "Showing" in dialog.lbl_catalog_count.text()

        # Test filtering by coating A
        coat_idx = dialog.filter_coating.findData("A")
        assert coat_idx >= 0
        dialog.filter_coating.setCurrentIndex(coat_idx)
        assert dialog.lens_table.rowCount() >= 30

        # Test diameter filter for 2 inch
        diam_idx = dialog.filter_diameter.findData(50.8)
        assert diam_idx >= 0
        dialog.filter_diameter.setCurrentIndex(diam_idx)
        assert dialog.lens_table.rowCount() >= 10

    def test_dialog_autofocus_radio_button_and_execution(self):
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0, glass="AIR"))
        init_lens = stack.insert_catalog_lens(0, "AC254-050-C", air_gap_after=43.3)
        stack.add_surface(Surface(name="DET", thickness=0.0, glass="AIR", surface_type=SurfaceType.DETECTOR))
        model = LensDataTableModel(stack)

        dialog = CatalogLensDialog(model=model, mode="replace")
        assert hasattr(dialog, "radio_gap_autofocus")
        assert "Auto-correct" in dialog.radio_gap_autofocus.text()

        # Find AC254-100-C in the table
        match_row = -1
        for r in range(dialog.lens_table.rowCount()):
            if dialog.lens_table.item(r, 0).text() == "AC254-100-C":
                match_row = r
                break
        assert match_row >= 0
        dialog.lens_table.selectRow(match_row)
        dialog._on_table_row_selected()

        # The autofocus radio button should dynamically display the new BFL (90.4 mm)
        assert "90.4" in dialog.radio_gap_autofocus.text()

        # Apply replacement with autofocus selected
        dialog.radio_gap_autofocus.setChecked(True)
        dialog._on_apply()

        # Verify that trailing thickness on rear lens surface was updated
        updated_stack = model.get_surface_stack()
        rear_surf = updated_stack.surfaces[3]  # rear of 3-surface doublet
        assert pytest.approx(rear_surf.thickness, rel=1e-2) == 90.4

    def test_autofocus_correction_physics_and_ray_trace(self):
        """Verify that replacing a lens with autofocus=True accurately focuses rays onto the detector."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=50.0, glass="AIR"))
        init_indices = stack.insert_catalog_lens(0, "AC254-050-C", air_gap_after=43.3)
        stack.add_surface(Surface(name="DET", thickness=0.0, glass="AIR", surface_type=SurfaceType.DETECTOR))

        # Replace AC254-050-C with AC254-100-C using autofocus
        rep_indices = stack.replace_lens_from_catalog(
            target=init_indices[0],
            part_number="AC254-100-C",
            auto_adjust_focus=True,
        )
        assert len(rep_indices) == 3
        # AC254-100-C catalog BFL is 90.4 mm
        rear_thickness = stack.surfaces[rep_indices[-1]].thickness
        assert pytest.approx(rear_thickness, rel=1e-2) == 90.4

        # Ray trace collimated parallel bundle across input aperture through the system
        import numpy as np
        from optics_app.engine.tracer import OpticalSystem
        from optics_app.engine.ray import Ray
        opt_sys = OpticalSystem(stack)
        
        y_spots = []
        for y_in in [-4.0, -2.0, 0.0, 2.0, 4.0]:
            r_in = Ray(pos=np.array([0.0, y_in, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.064)
            r_out = opt_sys.trace_ray(r_in)
            assert not r_out.vignetted
            assert len(r_out.history) > 0
            y_det = r_out.history[-1]["y"]
            y_spots.append(y_det)

        # Verify rays converge to a tight spot at the detector plane
        spot_size = max(y_spots) - min(y_spots)
        # Spot diameter should be well-focused (< 0.25 mm geometric spread for this f/4 doublet)
        assert spot_size < 0.25

        # Test relay case: not immediately followed by detector, trailing gap expands by ΔEFL
        relay_stack = SurfaceStack()
        relay_stack.add_surface(Surface(name="OBJ", thickness=50.0, glass="AIR"))
        r_init = relay_stack.insert_catalog_lens(0, "AC254-050-C", air_gap_after=60.0)
        relay_stack.add_surface(Surface(name="RELAY_APERTURE", thickness=20.0, glass="AIR"))

        relay_stack.replace_lens_from_catalog(
            target=r_init[0],
            part_number="AC254-100-C",
            auto_adjust_focus=True,
        )
        # ΔEFL = 100 - 50 = +50 mm, so 60.0 + 50.0 = 110.0 mm
        assert pytest.approx(relay_stack.surfaces[3].thickness, rel=1e-2) == 110.0
