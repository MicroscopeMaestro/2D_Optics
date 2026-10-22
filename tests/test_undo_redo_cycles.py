"""Exercise the actual desktop Undo/Redo actions over repeated edits."""
import copy

from optics_app.gui.qt_compat import QtWidgets
from optics_app.gui.main_window import MainWindow
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType


def test_repeated_desktop_undo_redo_and_branch(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Exercise model/dock restoration without expensive tracing on every edit.
    monkeypatch.setattr(MainWindow, "on_trace_rays", lambda *args: True)
    window = MainWindow()
    try:
        stack = SurfaceStack()
        for surface in (
            Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=20),
            Surface(name="Lens Front", radius=35, thickness=6, glass="N-BK7"),
            Surface(name="Lens Rear", radius=-70, thickness=30),
            Surface(name="Detector", surface_type=SurfaceType.DETECTOR),
        ):
            stack.add_surface(surface)
        window.model.set_surface_stack(stack)
        window.undo_manager.clear()

        def snapshot():
            return copy.deepcopy([(s.radius, s.thickness, s.conic, s.glass)
                                  for s in window.model._surfaces])

        states = [snapshot()]
        assert window.model.setData(1, 6, -0.5)
        states.append(snapshot())
        assert window.model.setData(1, 6, -1.5)
        states.append(snapshot())
        assert window.model.flip_lens_element(2)
        states.append(snapshot())
        assert window.model.setData(3, 5, 12.0)
        # Include aperture in comparisons too, so every action is observable.
        assert window.on_undo()
        assert window.model._surfaces[3].semi_diameter == 25.0

        for _ in range(3):
            for expected in reversed(states[:-1]):
                assert window.act_undo.isEnabled()
                window.act_undo.trigger()
                assert snapshot() == expected
            assert not window.act_undo.isEnabled()
            for expected in states[1:]:
                assert window.act_redo.isEnabled()
                window.act_redo.trigger()
                assert snapshot() == expected

        window.act_undo.trigger()
        assert window.act_redo.isEnabled()
        assert window.model.setData(1, 2, 48.0)
        assert not window.act_redo.isEnabled()
        window.act_undo.trigger()
        assert snapshot() == states[-2]
        window.act_redo.trigger()
        assert window.model._surfaces[1].radius == 48.0
    finally:
        window.close()


def test_whole_lens_catalog_replacement_from_internal_surface_is_one_undo_step(monkeypatch):
    """A click on a cemented surface replaces/undoes/redoes the complete assembly."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(MainWindow, "on_trace_rays", lambda *args: True)
    window = MainWindow()
    try:
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=20.0))
        stack.insert_catalog_lens(0, "AC254-050-C", air_gap_after=41.0)
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR))
        window.model.set_surface_stack(stack)
        window.undo_manager.clear()

        original = [(s.group_name, s.radius, s.thickness, s.glass) for s in window.model._surfaces]
        assert window.model.replace_lens_from_catalog(2, "AC254-030-C")
        replaced = [(s.group_name, s.radius, s.thickness, s.glass) for s in window.model._surfaces]
        assert [s.group_name for s in window.model._surfaces[1:4]] == ["AC254-030-C"] * 3
        assert len(window.model._surfaces) == len(original)

        for _ in range(2):
            assert window.on_undo()
            assert [(s.group_name, s.radius, s.thickness, s.glass) for s in window.model._surfaces] == original
            assert window.on_redo()
            assert [(s.group_name, s.radius, s.thickness, s.glass) for s in window.model._surfaces] == replaced
    finally:
        window.close()
