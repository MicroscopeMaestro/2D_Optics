"""Reusable mechanical-state support for actuated optical components.

The first built-in mechanism is the TI DLP4500 DMD.  Existing layouts are
treated as the aligned ON state; OFF rotates the micromirror by the full 24°
state-to-state travel without moving the DMD package or downstream optics.
The same state schema can represent filter-wheel slots, shutters, flip mounts,
and other future mechanisms.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


DMD4500_STATES: Dict[str, Dict[str, Any]] = {
    "ON": {
        "label": "ON (+12°) → detector path",
        "optical_tilt_offset_deg": [0.0, 0.0, 0.0],
        "route": "detector",
        "tilt_angle_deg": 12.0,
        "micromirror_tilt_deg": 12.0,
        "input_angle_deg": 24.0,
        "output_angle_deg": 0.0,
        "description": "Nominal aligned state; reflected light follows the designed detector arm.",
    },
    "OFF": {
        "label": "OFF (−12°) → beam dump",
        "optical_tilt_offset_deg": [-24.0, 0.0, 0.0],
        "route": "beam dump",
        "tilt_angle_deg": -12.0,
        "micromirror_tilt_deg": -12.0,
        "input_angle_deg": 24.0,
        "output_angle_deg": -48.0,
        "description": "Micromirror is 24° from ON, producing a 48° change in reflected-ray direction.",
    },
    "FLAT": {
        "label": "FLAT (0°) → parking / specular",
        "optical_tilt_offset_deg": [-12.0, 0.0, 0.0],
        "route": "parking",
        "tilt_angle_deg": 0.0,
        "micromirror_tilt_deg": 0.0,
        "input_angle_deg": 24.0,
        "output_angle_deg": -24.0,
        "description": "Unpowered flat micromirror parking position (0° tilt); specular reflection.",
    },
}


try:
    from optics_app.engine.plm import is_plm, is_dmd, get_plm_mechanical_states, ensure_plm_config
except ImportError:
    from v2.engine.plm import is_plm, is_dmd, get_plm_mechanical_states, ensure_plm_config



def dmd4500_reflection_angles(input_angle_deg: float = 24.0, state: str = "ON") -> Dict[str, Any]:
    """
    Calculate reflection angles according to Texas Instruments DLP4500 DMD specifications:
    - Micromirror tilt: +12° in ON, -12° in OFF, 0° in FLAT.
    - Output angle relative to substrate normal: theta_out = 2 * alpha_tilt - theta_in.
    """
    st = str(state).upper()
    tilt_deg = 12.0 if st == "ON" else (-12.0 if st == "OFF" else 0.0)
    theta_out = 2.0 * tilt_deg - float(input_angle_deg)
    route = DMD4500_STATES.get(st, {}).get("route", "unspecified")
    return {
        "state": st,
        "micromirror_tilt_deg": tilt_deg,
        "tilt_angle_deg": tilt_deg,
        "input_angle_deg": float(input_angle_deg),
        "output_angle_deg": theta_out,
        "deflection_from_on_deg": 2.0 * (tilt_deg - 12.0),
        "route": route,
    }


def available_states(surface: Any) -> Dict[str, Dict[str, Any]]:
    """Return explicit states, or inferred safe defaults for known hardware (DMD or TI-PLM)."""
    if is_plm(surface):
        plm_cfg = ensure_plm_config(surface)
        plm_states = get_plm_mechanical_states(
            plm_cfg.design_wavelength_um,
            modulator_type=getattr(plm_cfg, "modulator_type", "PLM"),
            tilt_angle_deg=getattr(plm_cfg, "tilt_angle_deg", 12.0),
        )
        explicit = getattr(surface, "mechanical_states", None)
        if isinstance(explicit, dict) and explicit:
            merged = deepcopy(plm_states)
            merged.update(explicit)
            return merged
        return plm_states
    explicit = getattr(surface, "mechanical_states", None)
    if isinstance(explicit, dict) and explicit:
        return explicit
    if is_dmd(surface):
        states = deepcopy(DMD4500_STATES)
        # The DLP4500 hinge follows the diamond-pixel diagonal.  Existing
        # layouts conventionally encode that plane with the non-zero base tilt.
        # Pick that axis so older X- and Y-oriented designs both work.
        if abs(float(getattr(surface, "tilt_y", 0.0))) > abs(float(getattr(surface, "tilt_x", 0.0))):
            states["OFF"]["optical_tilt_offset_deg"] = [0.0, -24.0, 0.0]
            states["FLAT"]["optical_tilt_offset_deg"] = [0.0, -12.0, 0.0]
        return states
    return {}


def active_state_name(surface: Any) -> str:
    states = available_states(surface)
    requested = str(getattr(surface, "mechanical_state", "") or "").upper()
    if requested in states:
        return requested
    return "ON" if "ON" in states else (next(iter(states), ""))


def active_state(surface: Any) -> Dict[str, Any]:
    return available_states(surface).get(active_state_name(surface), {})


def _rotation_matrix_xyz(angles_deg: Sequence[float]) -> np.ndarray:
    ax = list(angles_deg) + [0.0, 0.0, 0.0]
    rx, ry, rz = np.radians([float(ax[0]), float(ax[1]), float(ax[2])])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=float)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=float)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def get_effective_pose(surface: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve nominal position + orientation with active mechanism kinematics.
    Handles:
    - Galvo scanner continuous deflection around arbitrary rotation axis
      with off-vertex pivot offset kinematics:
        P_pivot = P_0 + R_mount @ d_pivot
        R_active = R_mount @ R_scan(theta_mech)
        P_active = P_pivot - R_active @ d_pivot
    - Discrete mechanical states (e.g. TI DLP4500 DMD tilt offsets)
    - Arbitrary continuous ActiveMechanism channels (tilts, decenters)
    Returns:
        (effective_origin, effective_R) as (np.ndarray shape (3,), np.ndarray shape (3, 3))
    """
    nom_origin = np.asarray(getattr(surface, "global_origin", [0.0, 0.0, 0.0]), dtype=float).copy()
    nom_R = np.asarray(getattr(surface, "global_R", np.eye(3)), dtype=float).copy()

    # 1. Galvo Scanner Kinematics
    galvo_cfg = getattr(surface, "galvo_config", None)
    if galvo_cfg is not None:
        theta_mech_deg = float(getattr(galvo_cfg, "current_angle_deg", 0.0))
        scan_axis = getattr(galvo_cfg, "scan_axis", "X")
        pivot_offset = np.asarray(getattr(galvo_cfg, "pivot_offset_mm", [0.0, 0.0, 0.0]), dtype=float)

        # Axis definition in local mount frame:
        # Optical deflection in X => mirror tilts around local Y axis [0, 1, 0]
        # Optical deflection in Y => mirror tilts around local X axis [1, 0, 0]
        if isinstance(scan_axis, str):
            ax_up = scan_axis.upper()
            if ax_up == "X":
                axis_vec = np.array([0.0, 1.0, 0.0], dtype=float)
            elif ax_up == "Y":
                axis_vec = np.array([1.0, 0.0, 0.0], dtype=float)
            elif ax_up == "Z":
                axis_vec = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                axis_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        elif isinstance(scan_axis, (list, tuple, np.ndarray)) and len(scan_axis) == 3:
            axis_vec = np.asarray(scan_axis, dtype=float)
            norm_ax = float(np.linalg.norm(axis_vec))
            if norm_ax > 0:
                axis_vec = axis_vec / norm_ax
            else:
                axis_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            axis_vec = np.array([0.0, 1.0, 0.0], dtype=float)

        theta_rad = np.radians(theta_mech_deg)
        K = np.array([
            [0.0, -axis_vec[2], axis_vec[1]],
            [axis_vec[2], 0.0, -axis_vec[0]],
            [-axis_vec[1], axis_vec[0], 0.0]
        ], dtype=float)
        R_scan = np.eye(3, dtype=float) + np.sin(theta_rad) * K + (1.0 - np.cos(theta_rad)) * (K @ K)

        eff_R = nom_R @ R_scan
        p_pivot = nom_origin + nom_R @ pivot_offset
        eff_origin = p_pivot - eff_R @ pivot_offset
        return eff_origin, eff_R

    # 2. Discrete mechanical states (DMD, filter wheels, etc.)
    states = available_states(surface)
    if states:
        offsets = active_state(surface).get("optical_tilt_offset_deg", [0.0, 0.0, 0.0])
        R_offset = _rotation_matrix_xyz(offsets)
        eff_R = nom_R @ R_offset
        return nom_origin, eff_R

    # 3. Arbitrary ActiveMechanism
    mech = getattr(surface, "mechanism", None)
    if mech is not None and hasattr(mech, "channels") and mech.channels:
        tilts = [0.0, 0.0, 0.0]
        decenters = [0.0, 0.0, 0.0]
        for ch in mech.channels.values():
            prop = getattr(ch, "target_property", "")
            val = ch.effective_property_value() if hasattr(ch, "effective_property_value") else getattr(ch, "current_value", 0.0)
            if prop == "tilt_x": tilts[0] += val
            elif prop == "tilt_y": tilts[1] += val
            elif prop == "tilt_z": tilts[2] += val
            elif prop == "decenter_x": decenters[0] += val
            elif prop == "decenter_y": decenters[1] += val
            elif prop == "decenter_z": decenters[2] += val
        eff_R = nom_R @ _rotation_matrix_xyz(tilts) if any(tilts) else nom_R
        eff_origin = nom_origin + nom_R @ np.asarray(decenters, dtype=float)
        return eff_origin, eff_R

    return nom_origin, nom_R


def interaction_rotation(surface: Any) -> np.ndarray:
    """Return the global optical frame for the active state of a surface."""
    return get_effective_pose(surface)[1]


def state_transmission(surface: Any, wavelength_um: float) -> float:
    """Return a state's scalar throughput, including an optional pass band."""
    state = active_state(surface)
    if bool(state.get("blocked", False)):
        return 0.0
    transmission = float(state.get("transmission", 1.0))
    band = state.get("wavelength_range_um")
    if isinstance(band, (list, tuple)) and len(band) == 2:
        if not float(band[0]) <= float(wavelength_um) <= float(band[1]):
            return 0.0
    return max(0.0, min(1.0, transmission))


class MechanicalStateController:
    """Discover and operate stateful components in a surface stack."""

    def __init__(self, stack: Any):
        self.stack = stack

    def mechanisms(self) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        for index, surface in enumerate(getattr(self.stack, "surfaces", [])):
            states = available_states(surface)
            if not states:
                continue
            found.append({
                "surface_index": index,
                "id": str(getattr(surface, "mechanism_id", "") or getattr(surface, "name", f"mechanism_{index}")),
                "type": str(getattr(surface, "mechanism_type", "") or ("PLM" if is_plm(surface) else ("DMD" if is_dmd(surface) else "generic"))),
                "name": str(getattr(surface, "name", f"Surface {index}")),
                "state": active_state_name(surface),
                "states": states,
            })
        return found

    def set_state(self, target: Any, state: str) -> Dict[str, Any]:
        index = self._resolve_target(target)
        surface = self.stack.surfaces[index]
        states = available_states(surface)
        requested = str(state).upper()
        if requested not in states:
            raise ValueError(f"State '{state}' is not available for {surface.name}; choose {sorted(states)}")
        if not getattr(surface, "mechanical_states", None):
            surface.mechanical_states = deepcopy(states)
        surface.mechanism_id = str(getattr(surface, "mechanism_id", "") or surface.name)
        surface.mechanism_type = str(getattr(surface, "mechanism_type", "") or ("PLM" if is_plm(surface) else ("DMD" if is_dmd(surface) else "generic")))
        surface.mechanical_state = requested
        return {
            "surface_index": index,
            "name": surface.name,
            "state": requested,
            "route": states[requested].get("route", "unspecified"),
            "description": states[requested].get("description", ""),
        }

    def _resolve_target(self, target: Any) -> int:
        surfaces = getattr(self.stack, "surfaces", [])
        if isinstance(target, int):
            if 0 <= target < len(surfaces):
                return target
            raise IndexError(f"Surface index {target} is outside the model")
        needle = str(target).strip().lower()
        exact = [i for i, s in enumerate(surfaces) if str(getattr(s, "name", "")).lower() == needle]
        partial = [i for i, s in enumerate(surfaces) if needle in str(getattr(s, "name", "")).lower()]
        matches = exact or partial
        if not matches:
            raise KeyError(f"No mechanical component matches '{target}'")
        return matches[0]


def detector_reached(traced_ray: Any, surfaces: Sequence[Any]) -> bool:
    """True when a traced ray recorded an intercept on a detector surface."""
    detector_indices = {i for i, s in enumerate(surfaces) if str(getattr(s, "surface_type", "")).upper() == "DETECTOR"}
    return any(
        int(hit.get("surface_index", -1)) in detector_indices
        and not bool(hit.get("vignetted", False))
        and float(hit.get("intensity", 0.0)) > 0.0
        for hit in getattr(traced_ray, "history", [])
    )


def simulate_state_throughput(stack: Any, target: Any, rays: Iterable[Any], states: Sequence[str] = ("ON", "OFF")) -> Dict[str, Dict[str, Any]]:
    """Trace the same rays for each mechanism state and report detector reach."""
    from optics_app.engine.tracer import OpticalSystem

    ray_list = list(rays)
    results: Dict[str, Dict[str, Any]] = {}
    for state in states:
        trial = deepcopy(stack)
        controller = MechanicalStateController(trial)
        applied = controller.set_state(target, state)
        traced = [OpticalSystem(trial).trace_ray(ray) for ray in ray_list]
        reached = sum(detector_reached(ray, OpticalSystem(trial).surfaces) for ray in traced)
        results[str(state).upper()] = {
            "launched": len(ray_list),
            "detector_hits": reached,
            "detector_throughput": (float(reached) / len(ray_list)) if ray_list else 0.0,
            "route": applied["route"],
        }
    return results
