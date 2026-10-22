"""
Non-Intrusive Undo & Redo (Memento Pattern) & Design Modification History Management for 2D_Optics.
Captures snapshots of SurfaceStack and SystemConfig states with timestamps and detailed action descriptions.
Supports Ctrl+Z (Undo), Ctrl+Shift+Z / Ctrl+Y (Redo), and direct time-travel to any historical revision.
"""
import copy
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface


@dataclass
class OpticalSystemState:
    stack_surfaces: List[Surface]
    system_config_dict: Optional[Dict[str, Any]] = None
    stack_auxiliary_dict: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    step_number: int = 1

    @property
    def num_surfaces(self) -> int:
        return len(self.stack_surfaces)


class UndoManager:
    """
    Manages undo/redo stacks and full modification revision timeline for optical designs.
    Deep-copies optical surfaces and configuration dictionaries.
    """

    def __init__(self, max_history: int = 150):
        self.max_history: int = max_history
        self._undo_stack: List[OpticalSystemState] = []
        self._redo_stack: List[OpticalSystemState] = []
        self._timeline: List[OpticalSystemState] = []
        self._current_step_counter: int = 0
        self._is_undoing_or_redoing: bool = False
        self._is_blocked: bool = False
        self._on_history_changed_callbacks: List[Callable[[], None]] = []

    @staticmethod
    def _capture_auxiliary(stack: Optional[SurfaceStack]) -> Dict[str, Any]:
        auxiliary: Dict[str, Any] = {}
        if stack is None:
            return auxiliary
        for attr in ("optical_routes", "two_photon_config", "nonlinear_imaging_config", "spectrometer_calibration", "analysis_config", "source_config", "optimizer_config", "viewer_settings"):
            value = getattr(stack, attr, None)
            if value is not None:
                auxiliary[attr] = copy.deepcopy(value.to_dict() if hasattr(value, "to_dict") else value)
        return auxiliary

    def add_history_listener(self, callback: Callable[[], None]) -> None:
        """Register a callback to be invoked whenever modification history changes."""
        if callback not in self._on_history_changed_callbacks:
            self._on_history_changed_callbacks.append(callback)

    def _notify_history_changed(self) -> None:
        for cb in list(self._on_history_changed_callbacks):
            try:
                cb()
            except Exception:
                pass

    def push_state(self, stack: SurfaceStack, system_config: Optional[Any] = None, description: str = "") -> bool:
        """Capture and push current system state onto the undo stack and history timeline."""
        if self._is_undoing_or_redoing or self._is_blocked:
            return False

        if stack is None or not hasattr(stack, 'surfaces'):
            return False

        # Clone surfaces safely
        cloned_surfaces = []
        for s in stack.surfaces:
            cloned_surfaces.append(copy.deepcopy(s))

        # Clone system config if available
        cfg_dict = None
        if system_config is not None:
            if hasattr(system_config, 'to_dict'):
                try:
                    cfg_dict = copy.deepcopy(system_config.to_dict())
                except Exception:
                    cfg_dict = None
            elif isinstance(system_config, dict):
                cfg_dict = copy.deepcopy(system_config)

        auxiliary = self._capture_auxiliary(stack)

        self._current_step_counter += 1
        state = OpticalSystemState(
            stack_surfaces=cloned_surfaces,
            system_config_dict=cfg_dict,
            stack_auxiliary_dict=auxiliary,
            description=description or "Modification",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            step_number=self._current_step_counter
        )

        # A new edit invalidates redo even if its pre-edit snapshot matches
        # the last checkpoint.
        self._redo_stack.clear()

        # Avoid redundant duplicate consecutive states
        if self._undo_stack:
            top = self._undo_stack[-1]
            if self._states_equal(top, state):
                self._current_step_counter -= 1
                self._notify_history_changed()
                return False

        self._undo_stack.append(state)
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)

        # Append to chronological timeline
        self._timeline.append(state)
        if len(self._timeline) > self.max_history * 2:
            self._timeline.pop(0)

        # Any new user mutation invalidates the redo branch
        self._redo_stack.clear()
        self._notify_history_changed()
        return True

    def undo(self, current_stack: SurfaceStack, current_system_config: Optional[Any] = None) -> Optional[OpticalSystemState]:
        """
        Pop the latest state from undo stack, push current state to redo stack,
        and return the state to restore.
        """
        if not self.can_undo():
            return None

        # Capture current state before reverting
        current_cloned = [copy.deepcopy(s) for s in current_stack.surfaces] if current_stack else []
        curr_cfg_dict = None
        if current_system_config is not None and hasattr(current_system_config, 'to_dict'):
            try:
                curr_cfg_dict = copy.deepcopy(current_system_config.to_dict())
            except Exception:
                curr_cfg_dict = None

        state_to_restore = self._undo_stack.pop()

        current_state = OpticalSystemState(
            stack_surfaces=current_cloned,
            system_config_dict=curr_cfg_dict,
            stack_auxiliary_dict=self._capture_auxiliary(current_stack),
            description=state_to_restore.description,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            step_number=state_to_restore.step_number
        )
        self._redo_stack.append(current_state)
        if len(self._redo_stack) > self.max_history:
            self._redo_stack.pop(0)

        self._notify_history_changed()
        return state_to_restore

    def redo(self, current_stack: SurfaceStack, current_system_config: Optional[Any] = None) -> Optional[OpticalSystemState]:
        """
        Pop the latest state from redo stack, push current state to undo stack,
        and return the state to restore.
        """
        if not self.can_redo():
            return None

        current_cloned = [copy.deepcopy(s) for s in current_stack.surfaces] if current_stack else []
        curr_cfg_dict = None
        if current_system_config is not None and hasattr(current_system_config, 'to_dict'):
            try:
                curr_cfg_dict = copy.deepcopy(current_system_config.to_dict())
            except Exception:
                curr_cfg_dict = None

        state_to_restore = self._redo_stack.pop()

        current_state = OpticalSystemState(
            stack_surfaces=current_cloned,
            system_config_dict=curr_cfg_dict,
            stack_auxiliary_dict=self._capture_auxiliary(current_stack),
            description=state_to_restore.description,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            step_number=state_to_restore.step_number
        )
        self._undo_stack.append(current_state)
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)

        self._notify_history_changed()
        return state_to_restore

    def jump_to_state_by_step(self, step_number: int, current_stack: SurfaceStack, current_system_config: Optional[Any] = None) -> Optional[OpticalSystemState]:
        """
        Allows jumping directly to any historical revision step in the timeline.
        """
        target_state = None
        for s in reversed(self._timeline):
            if s.step_number == step_number:
                target_state = s
                break

        if target_state is None:
            return None

        if current_stack:
            self.push_state(current_stack, current_system_config, f"Jump to Step #{step_number}")

        return target_state

    def get_history_timeline(self) -> List[Dict[str, Any]]:
        """
        Returns a list of human-readable modification history items for UI presentation.
        """
        items = []
        for i, s in enumerate(self._undo_stack):
            items.append({
                "step": s.step_number,
                "timestamp": s.timestamp,
                "description": s.description,
                "num_surfaces": s.num_surfaces,
                "is_current": (i == len(self._undo_stack) - 1 and len(self._redo_stack) == 0),
                "is_undo": True,
                "state": s
            })
        for i, s in enumerate(self._redo_stack):
            items.append({
                "step": s.step_number,
                "timestamp": s.timestamp,
                "description": f"[Redo] {s.description}",
                "num_surfaces": s.num_surfaces,
                "is_current": False,
                "is_undo": False,
                "state": s
            })
        return items

    def can_undo(self) -> bool:
        """Check if any undo operations are available."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if any redo operations are available."""
        return len(self._redo_stack) > 0

    def clear(self):
        """Clear all undo, redo, and timeline history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._timeline.clear()
        self._current_step_counter = 0
        self._notify_history_changed()

    @property
    def undo_count(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        return len(self._redo_stack)

    def _states_equal(self, s1: OpticalSystemState, s2: OpticalSystemState) -> bool:
        """Check equality between two snapshots to prevent duplicate stack entries."""
        if len(s1.stack_surfaces) != len(s2.stack_surfaces):
            return False

        for a, b in zip(s1.stack_surfaces, s2.stack_surfaces):
            # Surface.__eq__ and the legacy comparisons below cover only a
            # subset of editable parameters. Compare all stored prescription
            # fields, including conics, aspheres, apertures and solve flags.
            import numpy as np

            def equal_value(left, right):
                if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                    return np.array_equal(left, right, equal_nan=True)
                if isinstance(left, dict) and isinstance(right, dict):
                    return left.keys() == right.keys() and all(
                        equal_value(left[key], right[key]) for key in left
                    )
                if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
                    return len(left) == len(right) and all(
                        equal_value(x, y) for x, y in zip(left, right)
                    )
                if hasattr(left, '__dict__') and hasattr(right, '__dict__'):
                    return type(left) is type(right) and equal_value(vars(left), vars(right))
                return left == right

            ignored = {'global_origin', 'global_R'}  # Derived placement cache.
            if not equal_value(
                {key: val for key, val in vars(a).items() if key not in ignored},
                {key: val for key, val in vars(b).items() if key not in ignored},
            ):
                return False
            if a.name != b.name:
                return False
            if a.surface_type != b.surface_type:
                return False
            if abs(a.radius - b.radius) > 1e-9:
                return False
            if abs(a.thickness - b.thickness) > 1e-9:
                return False
            if str(a.glass).upper() != str(b.glass).upper():
                return False
            if abs(a.semi_diameter - b.semi_diameter) > 1e-9:
                return False
            if abs(a.tilt_x - b.tilt_x) > 1e-9 or abs(a.tilt_y - b.tilt_y) > 1e-9 or abs(a.tilt_z - b.tilt_z) > 1e-9:
                return False
            if abs(a.decenter_x - b.decenter_x) > 1e-9 or abs(a.decenter_y - b.decenter_y) > 1e-9:
                return False

        if s1.system_config_dict != s2.system_config_dict:
            return False
        if s1.stack_auxiliary_dict != s2.stack_auxiliary_dict:
            return False

        return True
