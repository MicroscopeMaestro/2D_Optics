"""Tests for Thorlabs Achromat & Aspheric Catalog and Parameterized Replacement Tool.

Verifies:
1. Catalog queries, sections, filtering, and exact Thorlabs specifications.
2. SurfaceStack.replace_lens_from_catalog parameter choices:
   - Normal forward orientation (R1 -> R2 -> R3)
   - Reversed/flipped orientation (-R3 -> -R2 -> -R1, element thickness/glass swap)
   - Air gap options: preserve existing, set to catalog BFL, custom distance
   - Aperture semi-diameter override vs preservation
   - 3D pose preservation
3. LensDataTableModel catalog replacement and insertion bindings.
4. Interactive CatalogLensDialog Qt workflow with parameter selection.
"""

import pytest
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.catalogs.thorlabs import (
    get_catalog_sections, get_lens_catalog, get_catalog_lens, query_catalog
)
from optics_app.gui.qt_compat import QT_AVAILABLE, QApplication
from optics_app.gui.lde_table import LensDataTableModel


def create_collimator_spectrometer_stack() -> SurfaceStack:
    """Create a sample optical system with an object slit, a collimator doublet, and a grating."""
    stack = SurfaceStack()
    # S0: Slit / Object
    stack.add_surface(Surface(name="Slit", surface_type=SurfaceType.OBJECT, thickness=50.0))
    # S1, S2, S3: AC254-050-C Collimator Doublet (standard Thorlabs prescription)
    stack.add_surface(Surface(name="L1_Front", radius=28.3, thickness=7.0, glass="N-LAK22", semi_diameter=12.7))
    stack.add_surface(Surface(name="L1_Cemented", radius=-22.6, thickness=2.0, glass="N-SF6HT", semi_diameter=12.7))
    stack.add_surface(Surface(name="L1_Rear", radius=-104.7, thickness=30.0, glass="AIR", semi_diameter=12.7))
    # S4: Grating
    stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=50.0, grating_lines_per_mm=600.0))
    # S5: Detector
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0))
    return stack


class TestCatalogPrescriptionsAndQueries:
    def test_catalog_sections_and_completeness(self):
        sections = get_catalog_sections()
        assert "achromatic_doublets" in sections
        assert "aspheric_condensers" in sections
        assert "visible_achromats_b" in sections

        cat = get_lens_catalog()
        assert "AC254-050-C" in cat
        assert "AC254-030-C" in cat
        assert "AC254-100-C" in cat
        assert "ACL25416U" in cat
        assert len(cat) >= 29

        full_cat = SurfaceStack._get_lens_catalog()
        assert "ACL25416U" in full_cat

    def test_catalog_query_filtering(self):
        # Query 1-inch doublets in C coating
        c_lenses = query_catalog(section="achromatic_doublets", diameter_mm=25.4)
        assert len(c_lenses) >= 8
        part_numbers = [m["part_number"] for m in c_lenses]
        assert "AC254-050-C" in part_numbers
        assert "AC254-030-C" in part_numbers

        # Query search text
        matches = query_catalog(search_text="AC254-050")
        assert any(m["part_number"] == "AC254-050-C" for m in matches)

    def test_catalog_exact_parameters(self):
        lens = get_catalog_lens("AC254-050-C")
        assert lens is not None
        assert lens["focal_length_mm"] == pytest.approx(50.0, abs=0.1)
        assert lens["diameter_mm"] == pytest.approx(25.4, abs=0.01)
        assert lens["back_focal_length_mm"] == pytest.approx(41.2, abs=0.5)
        assert len(lens["surfaces"]) == 3
        # Thorlabs official glasses for AC254-050-C: N-LAK22 and N-SF6HT
        assert lens["surfaces"][0]["glass"].upper() == "N-LAK22"
        assert lens["surfaces"][1]["glass"].upper() == "N-SF6HT"


class TestSurfaceStackReplaceFromCatalog:
    def test_replace_normal_orientation_preserve_gap(self):
        stack = create_collimator_spectrometer_stack()
        initial_trailing_gap = stack.surfaces[3].thickness  # 30.0 mm
        
        # Replace the collimator with AC254-030-C in normal orientation, preserving gap
        indices = stack.replace_lens_from_catalog(
            target=1,
            part_number="AC254-030-C",
            reverse=False,
            air_gap_after=None,
            use_catalog_bfl=False
        )
        assert len(indices) == 3
        s1, s2, s3 = stack.surfaces[indices[0]], stack.surfaces[indices[1]], stack.surfaces[indices[2]]
        assert s1.glass == "N-BAF10"
        assert s2.glass == "N-SF6HT"
        assert s3.glass == "AIR"
        # Preserved trailing air gap:
        assert s3.thickness == pytest.approx(initial_trailing_gap, abs=1e-5)
        # S1 radius is positive for AC254-030-C in normal orientation
        assert s1.radius > 0

    def test_replace_reversed_orientation(self):
        stack = create_collimator_spectrometer_stack()
        ac30 = get_catalog_lens("AC254-030-C")
        r1 = float(ac30["surfaces"][0]["radius"])
        r2 = float(ac30["surfaces"][1]["radius"])
        r3 = float(ac30["surfaces"][2]["radius"])
        t1 = float(ac30["surfaces"][0]["thickness"])
        t2 = float(ac30["surfaces"][1]["thickness"])
        g1 = str(ac30["surfaces"][0]["glass"])
        g2 = str(ac30["surfaces"][1]["glass"])

        # Replace with reversed orientation (collimator mode)
        indices = stack.replace_lens_from_catalog(
            target=1,
            part_number="AC254-030-C",
            reverse=True,
            air_gap_after=None
        )
        s1, s2, s3 = stack.surfaces[indices[0]], stack.surfaces[indices[1]], stack.surfaces[indices[2]]

        # Reversed doublet physics:
        # S1: -R3, thickness=t2, glass=g2
        assert s1.radius == pytest.approx(-r3, abs=1e-5)
        assert s1.thickness == pytest.approx(t2, abs=1e-5)
        assert s1.glass == g2

        # S2: -R2, thickness=t1, glass=g1
        assert s2.radius == pytest.approx(-r2, abs=1e-5)
        assert s2.thickness == pytest.approx(t1, abs=1e-5)
        assert s2.glass == g1

        # S3: -R1, thickness=trailing gap, glass=AIR
        assert s3.radius == pytest.approx(-r1, abs=1e-5)
        assert s3.glass == "AIR"

    def test_replace_with_catalog_bfl(self):
        stack = create_collimator_spectrometer_stack()
        ac50 = get_catalog_lens("AC254-050-C")
        bfl = float(ac50["back_focal_length_mm"])

        indices = stack.replace_lens_from_catalog(
            target=1,
            part_number="AC254-050-C",
            use_catalog_bfl=True
        )
        s3 = stack.surfaces[indices[-1]]
        assert s3.thickness == pytest.approx(bfl, abs=0.1)

    def test_replace_with_custom_air_gap(self):
        stack = create_collimator_spectrometer_stack()
        custom_gap = 42.125

        indices = stack.replace_lens_from_catalog(
            target=1,
            part_number="AC254-050-C",
            air_gap_after=custom_gap
        )
        s3 = stack.surfaces[indices[-1]]
        assert s3.thickness == pytest.approx(custom_gap, abs=1e-5)

    def test_replace_preserve_aperture_and_pose(self):
        stack = create_collimator_spectrometer_stack()
        # Set custom tilt and aperture on front surface
        stack.surfaces[1].tilt_x = 2.5
        stack.surfaces[1].tilt_y = -1.0
        stack.surfaces[1].semi_diameter = 15.0

        indices = stack.replace_lens_from_catalog(
            target=1,
            part_number="AC254-050-C",
            preserve_aperture=True
        )
        s1 = stack.surfaces[indices[0]]
        assert s1.tilt_x == pytest.approx(2.5, abs=1e-5)
        assert s1.tilt_y == pytest.approx(-1.0, abs=1e-5)
        assert s1.semi_diameter == pytest.approx(15.0, abs=1e-5)


class TestLensDataTableModelCatalogActions:
    def test_model_replace_lens_from_catalog(self):
        stack = create_collimator_spectrometer_stack()
        model = LensDataTableModel(stack)
        initial_signals = model.update_signals_emitted

        success = model.replace_lens_from_catalog(
            target=1,
            part_number="AC254-030-C",
            reverse=True,
            air_gap_after=18.5
        )
        assert success is True
        assert model.update_signals_emitted > initial_signals
        
        updated_stack = model.get_surface_stack()
        # Find replaced lens
        assert updated_stack.surfaces[3].thickness == pytest.approx(18.5, abs=1e-5)

    def test_model_insert_catalog_lens_after(self):
        stack = create_collimator_spectrometer_stack()
        model = LensDataTableModel(stack)
        orig_count = len(model._surfaces)

        inserted_indices = model.insert_catalog_lens_after(
            row=3,
            part_number="AC254-050-C",
            air_gap_after=15.0,
            reverse=False
        )
        assert len(inserted_indices) == 3
        assert len(model._surfaces) == orig_count + 3


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt not available")
class TestCatalogLensDialogInteractive:
    @pytest.fixture(autouse=True)
    def ensure_qapp(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        self.app = app

    def test_dialog_init_and_filtering(self):
        from optics_app.gui.catalog_lens_dialog import CatalogLensDialog
        stack = create_collimator_spectrometer_stack()
        model = LensDataTableModel(stack)

        dlg = CatalogLensDialog(model=model, target=1, mode="replace")
        assert dlg.target_combo.count() >= 1

        # Check section filtering
        sec_idx = dlg.filter_section.findData("achromatic_doublets")
        if sec_idx >= 0:
            dlg.filter_section.setCurrentIndex(sec_idx)
        dlg._apply_filters()
        assert dlg.lens_table.rowCount() >= 10

        # Check search filtering
        dlg.search_edit.setText("AC254-050-C")
        dlg._apply_filters()
        assert dlg.lens_table.rowCount() >= 1
        assert dlg._selected_lens_record is not None
        assert dlg._selected_lens_record["part_number"] == "AC254-050-C"

    def test_dialog_apply_normal_and_custom_gap(self):
        from optics_app.gui.catalog_lens_dialog import CatalogLensDialog
        stack = create_collimator_spectrometer_stack()
        model = LensDataTableModel(stack)

        dlg = CatalogLensDialog(model=model, target=1, mode="replace")
        dlg.search_edit.setText("AC254-030-C")
        dlg._apply_filters()
        dlg.lens_table.selectRow(0)

        # Parameter choices: Normal orientation, Custom gap 27.5 mm
        dlg.radio_normal.setChecked(True)
        dlg.radio_gap_custom.setChecked(True)
        dlg.spin_custom_gap.setValue(27.5)

        dlg._on_apply()

        updated_stack = model.get_surface_stack()
        # S1, S2, S3 is replaced doublet
        assert updated_stack.surfaces[3].thickness == pytest.approx(27.5, abs=1e-5)
        assert updated_stack.surfaces[1].glass == "N-BAF10"

    def test_dialog_apply_reversed_orientation_and_bfl(self):
        from optics_app.gui.catalog_lens_dialog import CatalogLensDialog
        stack = create_collimator_spectrometer_stack()
        model = LensDataTableModel(stack)

        dlg = CatalogLensDialog(model=model, target=1, mode="replace")
        dlg.search_edit.setText("AC254-050-C")
        dlg._apply_filters()
        dlg.lens_table.selectRow(0)

        # Parameter choices: Flipped / Reversed, Catalog BFL
        dlg.radio_reverse.setChecked(True)
        dlg.radio_gap_bfl.setChecked(True)

        dlg._on_apply()

        updated_stack = model.get_surface_stack()
        ac50 = get_catalog_lens("AC254-050-C")
        # In reversed orientation, front glass is N-SF6HT
        assert updated_stack.surfaces[1].glass == "N-SF6HT"
        # Trailing gap is catalog BFL (~43.3 mm)
        assert updated_stack.surfaces[3].thickness == pytest.approx(ac50["back_focal_length_mm"], abs=0.5)
