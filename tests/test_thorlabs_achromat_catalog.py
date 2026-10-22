"""Validation for the sectioned Thorlabs achromatic-doublet catalog."""

import numpy as np
import pytest

from optics_app.catalogs.thorlabs import get_catalog_sections, get_lens_catalog, query_catalog
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.gui.catalog_menu import diameter_menu_label
from optics_app.utils.materials import get_refractive_index


def _cardinal_points(record, wavelength_um=1.330):
    surfaces = record["surfaces"]

    def refraction(n_before, n_after, radius):
        return np.array([
            [1.0, 0.0],
            [-(n_after - n_before) / (radius * n_after), n_before / n_after],
        ])

    def translation(distance):
        return np.array([[1.0, distance], [0.0, 1.0]])

    n1 = get_refractive_index(surfaces[0]["glass"], wavelength_um)
    n2 = get_refractive_index(surfaces[1]["glass"], wavelength_um)
    matrix = (
        refraction(n2, 1.0, surfaces[2]["radius"])
        @ translation(surfaces[1]["thickness"])
        @ refraction(n1, n2, surfaces[1]["radius"])
        @ translation(surfaces[0]["thickness"])
        @ refraction(1.0, n1, surfaces[0]["radius"])
    )
    return -1.0 / matrix[1, 0], -matrix[0, 0] / matrix[1, 0]


def test_catalog_has_achromat_section_and_multiple_diameters_and_powers():
    sections = get_catalog_sections()
    assert "achromatic_doublets" in sections

    records = list(sections["achromatic_doublets"]["parts"].values())
    assert len(records) == 29
    assert {record["diameter_mm"] for record in records} == {
        5.0, 6.0, 6.35, 8.0, 12.7, 25.4, 30.0, 50.8
    }
    assert min(record["optical_power_diopters"] for record in records) < 6.0
    assert max(record["optical_power_diopters"] for record in records) > 50.0


def test_catalog_filters_by_diameter_and_nominal_focal_length():
    matches = query_catalog(
        section="achromatic_doublets",
        diameter_mm=25.4,
        focal_length_mm=100.0,
        coating_code="C",
    )
    assert [record["part_number"] for record in matches] == ["AC254-100-C"]


def test_every_achromat_matches_catalog_efl_and_bfl_at_design_wavelength():
    for part_number, record in get_catalog_sections()["achromatic_doublets"]["parts"].items():
        efl, bfl = _cardinal_points(record)
        # Catalog values are nominal and the in-app Schott dispersion model is
        # intentionally compact; miniature high-power doublets are most sensitive.
        assert efl == pytest.approx(record["focal_length_mm"], rel=0.025), part_number
        assert bfl == pytest.approx(record["back_focal_length_mm"], rel=0.025), part_number


def test_catalog_lenses_can_be_inserted_and_replaced_while_preserving_air_gap():
    stack = SurfaceStack()
    stack.surfaces.append(Surface(name="OBJ", thickness=20.0, glass="AIR"))

    inserted = stack.insert_catalog_lens(0, "AC127-030-C", air_gap_after=17.0)
    assert inserted == [1, 2, 3]
    assert stack.surfaces[1].semi_diameter == 6.35
    assert stack.surfaces[3].thickness == 17.0

    replaced = stack.replace_lens_from_catalog(1, "AC508-100-C")
    assert replaced == [1, 2, 3]
    assert stack.surfaces[1].semi_diameter == 25.4
    assert stack.surfaces[1].glass == "N-BAF10"
    assert stack.surfaces[2].glass == "N-SF6HT"
    assert stack.surfaces[3].thickness == 17.0


def test_catalog_diameter_labels_expose_imperial_and_miniature_sizes():
    assert diameter_menu_label(25.4) == "1 inch (Ø25.4 mm)"
    assert diameter_menu_label(12.7) == "1/2 inch (Ø12.7 mm)"
    assert "Miniature ≤10 mm" in diameter_menu_label(8.0)


def test_replacement_is_atomic_and_only_changes_selected_duplicate_lens():
    stack = SurfaceStack()
    stack.surfaces = [Surface(name="OBJ", thickness=20.0, glass="AIR")]
    stack.insert_catalog_lens(0, "AC127-030-C", air_gap_after=31.0)
    stack.insert_catalog_lens(3, "AC127-030-C", air_gap_after=47.0)
    stack.surfaces.append(Surface(name="Detector", glass="AIR", thickness=0.0))
    stack._update_accumulated_positions()

    untouched_prefix = stack.surfaces[0]
    first_lens_objects = tuple(stack.surfaces[1:4])
    untouched_detector = stack.surfaces[-1]
    first_lens_values = [(s.radius, s.thickness, s.glass) for s in first_lens_objects]

    replaced = stack.replace_lens_from_catalog(4, "AC050-010-C")

    assert replaced == [4, 5, 6]
    assert stack.surfaces[0] is untouched_prefix
    assert tuple(stack.surfaces[1:4]) == first_lens_objects
    assert [(s.radius, s.thickness, s.glass) for s in stack.surfaces[1:4]] == first_lens_values
    assert stack.surfaces[-1] is untouched_detector
    assert [s.glass for s in stack.surfaces[4:7]] == ["N-LAK22", "N-SF6", "AIR"]
    assert stack.surfaces[6].thickness == 47.0


@pytest.mark.parametrize("selected_surface", [1, 2, 3])
def test_quick_replace_resolves_every_selected_surface_to_the_whole_lens(selected_surface):
    """Front, cemented, and rear-boundary selections replace the same complete lens."""
    stack = SurfaceStack()
    object_surface = Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=20.0, glass="AIR")
    detector_surface = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR")
    stack.surfaces = [object_surface]
    stack.insert_catalog_lens(0, "AC254-050-C", air_gap_after=41.0)
    old_lens_surfaces = tuple(stack.surfaces[1:4])
    stack.surfaces.append(detector_surface)
    stack._update_accumulated_positions()

    replaced = stack.replace_with_catalog_lens(selected_surface, "AC254-030-C")

    assert replaced == [1, 2, 3]
    assert len(stack.surfaces) == 5
    assert stack.surfaces[0] is object_surface
    assert stack.surfaces[4] is detector_surface
    assert not any(surface in stack.surfaces for surface in old_lens_surfaces)
    assert [surface.group_name for surface in stack.surfaces[1:4]] == ["AC254-030-C"] * 3
    assert [surface.glass for surface in stack.surfaces[1:4]] == ["N-BAF10", "N-SF6HT", "AIR"]
    assert stack.surfaces[3].thickness == pytest.approx(41.0)


def test_catalog_replace_rejects_standalone_air_surface_without_mutating_system():
    stack = SurfaceStack()
    stack.surfaces = [
        Surface(name="OBJ", surface_type=SurfaceType.OBJECT, thickness=20.0, glass="AIR"),
        Surface(name="Free Space", thickness=35.0, glass="AIR"),
        Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR"),
    ]
    before = list(stack.surfaces)

    with pytest.raises(ValueError, match="physical lens"):
        stack.replace_lens_from_catalog(1, "AC254-030-C")

    assert stack.surfaces == before


def test_ambiguous_lens_name_is_rejected_without_mutating_stack():
    stack = SurfaceStack()
    stack.surfaces = [Surface(name="OBJ", thickness=20.0, glass="AIR")]
    stack.insert_catalog_lens(0, "AC127-030-C", air_gap_after=20.0)
    stack.insert_catalog_lens(3, "AC127-030-C", air_gap_after=30.0)
    before = [(id(s), s.radius, s.thickness, s.glass) for s in stack.surfaces]

    with pytest.raises(ValueError, match="ambiguous"):
        stack.replace_lens_from_catalog("AC127-030-C", "AC050-010-C")

    assert [(id(s), s.radius, s.thickness, s.glass) for s in stack.surfaces] == before
