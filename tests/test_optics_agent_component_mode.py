"""Natural-language catalog replacement and merged LDE component-name view."""

import json

from optics_app.ai.nemotron_client import parse_optical_command_heuristics
from optics_app.ai.tools import (
    get_default_tool_registry,
    interpret_and_execute_command,
    optimize_focus_at_surface,
    replace_optical_element,
    replace_lens_from_thorlabs_query,
)
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.models import SpectrometerSystem
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.gui.lde_table import LensDataEditorWidget, LensDataTableModel
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets


_APP = None
if QT_AVAILABLE and hasattr(QtWidgets, "QApplication"):
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _named_catalog_lens_stack(name="Collimator"):
    stack = SurfaceStack()
    stack.surfaces = [Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=20.0, glass="AIR")]
    stack.insert_catalog_lens(0, "AC254-050-C", air_gap_after=41.0)
    for surface in stack.surfaces[1:4]:
        surface.group_name = name
    detector = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR")
    stack.surfaces.append(detector)
    stack._update_accumulated_positions()
    return stack, detector


def test_agent_understands_named_30_mm_thorlabs_achromat_request():
    call = parse_optical_command_heuristics(
        'replace the lens "Collimator" for a 30 mm achormatic lens from Thorlabs'
    )
    assert call == {
        "name": "replace_lens_from_thorlabs_query",
        "arguments": {"element": "Collimator", "focal_length_mm": 30.0},
    }


def test_agent_distinguishes_explicit_diameter_from_focal_length():
    call = parse_optical_command_heuristics(
        'replace lens "Camera" with a Thorlabs achromat, focal length 50 mm, diameter 30 mm'
    )
    assert call["name"] == "replace_lens_from_thorlabs_query"
    assert call["arguments"]["focal_length_mm"] == 50.0
    assert call["arguments"]["diameter_mm"] == 30.0


def test_agent_selects_catalog_by_focal_length_and_current_lens_diameter():
    stack, detector = _named_catalog_lens_stack()
    system = SpectrometerSystem(surface_stack=stack)

    result = json.loads(
        replace_lens_from_thorlabs_query("Collimator", 30.0, system=system)
    )

    assert result["status"] == "success"
    assert result["selected_part"] == "AC254-030-C"
    assert result["surface_indices"] == [1, 2, 3]
    assert [surface.group_name for surface in stack.surfaces[1:4]] == ["AC254-030-C"] * 3
    assert [surface.glass for surface in stack.surfaces[1:4]] == ["N-BAF10", "N-SF6HT", "AIR"]
    assert stack.surfaces[3].thickness == 41.0
    assert stack.surfaces[4] is detector


def test_generic_ai_replace_tool_replaces_whole_lens_from_internal_surface():
    stack, detector = _named_catalog_lens_stack("Camera Lens")
    system = SpectrometerSystem(surface_stack=stack)
    old_surfaces = tuple(stack.surfaces[1:4])

    message = replace_optical_element(2, "AC254-030-C", system=system)

    assert message.startswith("Successfully replaced")
    assert len(stack.surfaces) == 5
    assert not any(surface in stack.surfaces for surface in old_surfaces)
    assert [surface.group_name for surface in stack.surfaces[1:4]] == ["AC254-030-C"] * 3
    assert [surface.glass for surface in stack.surfaces[1:4]] == ["N-BAF10", "N-SF6HT", "AIR"]
    assert stack.surfaces[3].thickness == 41.0
    assert stack.surfaces[4] is detector


def test_full_ai_command_fallback_executes_catalog_replacement():
    class NoToolCallClient:
        def chat_completion(self, **kwargs):
            return {"content": "", "tool_calls": []}

    stack, _ = _named_catalog_lens_stack("Camera Lens")
    system = SpectrometerSystem(surface_stack=stack)
    result = interpret_and_execute_command(
        'replace lens "Camera Lens" for a 30 mm achromatic lens from Thorlabs',
        system=system,
        client=NoToolCallClient(),
    )

    assert result["success"] is True
    assert result["executed_results"][0]["tool"] == "replace_lens_from_thorlabs_query"
    tool_result = json.loads(result["executed_results"][0]["result"])
    assert tool_result["selected_part"] == "AC254-030-C"
    assert [surface.group_name for surface in stack.surfaces[1:4]] == ["AC254-030-C"] * 3


def _defocused_singlet_system():
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=1000.0))
    stack.add_surface(Surface(
        name="Focus Lens Front", radius=51.68, thickness=5.0,
        glass="N-BK7", semi_diameter=15.0,
    ))
    stack.add_surface(Surface(
        name="Focus Lens Rear Gap", radius=0.0, thickness=30.0,
        glass="AIR", semi_diameter=15.0,
    ))
    stack.add_surface(Surface(
        name="Detector", surface_type=SurfaceType.DETECTOR,
        thickness=0.0, glass="AIR", semi_diameter=20.0,
    ))
    return SpectrometerSystem(surface_stack=stack, wavelengths=[0.5876])


def test_agent_routes_surface_focus_request_to_autofocus_tool():
    call = parse_optical_command_heuristics(
        'optimize focus at surface "Detector" by adjusting surface "Focus Lens Rear Gap"'
    )
    assert call == {
        "name": "optimize_focus_at_surface",
        "arguments": {
            "target_surface": "Detector",
            "adjust_surface": "Focus Lens Rear Gap",
            "max_iterations": 80,
        },
    }
    assert parse_optical_command_heuristics("optimize focus at the Detector surface")["arguments"]["target_surface"] == "Detector"


def test_surface_autofocus_tool_is_registered_and_changes_only_requested_gap():
    assert "optimize_focus_at_surface" in get_default_tool_registry().tools
    system = _defocused_singlet_system()
    before = [(s.radius, s.thickness, s.glass) for s in system.surfaces]

    result = json.loads(optimize_focus_at_surface(
        target_surface="Detector",
        adjust_surface="Focus Lens Rear Gap",
        min_gap_mm=10.0,
        max_gap_mm=200.0,
        max_iterations=100,
        system=system,
    ))

    assert result["status"] == "success"
    assert result["final_rms_mm"] < result["initial_rms_mm"]
    assert result["improvement_percent"] >= 50.0
    assert result["changed_parameters"] == ["surface[2].thickness"]
    assert system.surfaces[2].thickness != before[2][1]
    for index in (0, 1, 3):
        assert (system.surfaces[index].radius, system.surfaces[index].thickness, system.surfaces[index].glass) == before[index]


def test_full_ai_command_fallback_executes_surface_autofocus():
    class NoToolCallClient:
        def chat_completion(self, **kwargs):
            return {"content": "", "tool_calls": []}

    system = _defocused_singlet_system()
    result = interpret_and_execute_command(
        "optimize focus at surface Detector",
        system=system,
        client=NoToolCallClient(),
    )

    assert result["success"] is True
    assert result["executed_results"][0]["tool"] == "optimize_focus_at_surface"
    tool_result = json.loads(result["executed_results"][0]["result"])
    assert tool_result["status"] == "success"
    assert tool_result["final_rms_mm"] < tool_result["initial_rms_mm"]


def test_lde_can_merge_and_split_only_the_surface_name_column():
    stack, _ = _named_catalog_lens_stack()
    model = LensDataTableModel(stack)

    spans = model.component_name_spans()
    assert spans == [{
        "start": 1,
        "count": 3,
        "end": 3,
        "name": "Collimator",
        "label": "◉ Collimator  (3 surfaces)",
    }]
    # Default is False: surfaces are displayed individually, not merged
    assert model.component_name_mode() is False
    assert model.data(1, 0) == stack.surfaces[1].name
    assert model.component_rows_for_row(2) == [2]

    # Non-name prescription cells remain surface-level and unchanged.
    assert model.data(1, 2) == stack.surfaces[1].radius
    assert model.data(2, 4) == stack.surfaces[2].glass

    # Enabling component mode merges first column names
    model.set_component_name_mode(True)
    assert model.component_name_mode() is True
    assert model.data(1, 0) == "◉ Collimator  (3 surfaces)"
    assert model.component_rows_for_row(2) == [1, 2, 3]

    model.set_component_name_mode(False)
    assert model.component_name_mode() is False
    assert model.data(1, 0) == stack.surfaces[1].name
    assert model.component_rows_for_row(2) == [2]


def test_merged_view_keeps_repeated_identical_lenses_as_two_components():
    stack, _ = _named_catalog_lens_stack("Relay")
    stack.surfaces.pop()  # detector
    stack.insert_catalog_lens(3, "AC254-050-C", air_gap_after=22.0)
    for surface in stack.surfaces[4:7]:
        surface.group_name = "Relay"
    model = LensDataTableModel(stack)

    assert [(span["start"], span["count"]) for span in model.component_name_spans()] == [(1, 3), (4, 3)]
    assert model.set_lens_locked(1, True) is True
    assert model.set_lens_locked(4, True) is True
    assert model.set_lens_locked(4, False) is True
    assert all(surface.is_locked for surface in model._surfaces[1:4])
    assert all(not surface.is_locked for surface in model._surfaces[4:7])

    # An index identifies one physical instance even when both lenses share the
    # exact same catalog/common name.
    assert model.delete_lens_element(4) is True
    assert len(model._surfaces) == 4
    assert [(span["start"], span["count"]) for span in model.component_name_spans()] == [(1, 3)]


def test_qt_lde_merge_and_split_controls_apply_real_table_spans():
    if not QT_AVAILABLE:
        return
    stack, _ = _named_catalog_lens_stack()
    widget = LensDataEditorWidget(LensDataTableModel(stack))

    # By default, surfaces are not merged: rowSpan is 1
    assert widget.table_view.frozen_table_view.rowSpan(1, 0) == 1
    widget._set_component_name_mode(True)
    assert widget.table_view.frozen_table_view.rowSpan(1, 0) == 3
    widget._set_component_name_mode(False)
    assert widget.table_view.frozen_table_view.rowSpan(1, 0) == 1
    widget.close()


def test_desktop_main_window_exposes_component_controls_and_root_optics_agent():
    if not QT_AVAILABLE:
        return
    from optics_app.gui.main_window import MainWindow

    window = MainWindow()
    assert window.ai_dock.__class__.__module__ == "optics_app.gui.ai_dock"
    assert window.btn_merge_lde_components is None
    assert window.btn_split_lde_components is None
    assert window.model.component_name_mode() is False
    stack, _ = _named_catalog_lens_stack()
    window.model.set_surface_stack(stack)
    # Individual surface selection without merged component expansion
    assert window._select_complete_lens(2) == 2
    assert window.component_editor_dock.group_rotation_check.isChecked() is True
    window.close()
