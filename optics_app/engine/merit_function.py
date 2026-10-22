"""
Zemax-Style Optical Merit Function Engine (Operands, Multi-Surface Spot Sizes, Weights, Constraints).
Defines MeritOperand, OperandType, and MeritFunction for multi-criteria optical optimization.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from optics_app.engine.ray import Ray
from optics_app.engine.surface import SurfaceType


class OperandType(Enum):
    """Supported Merit Function Operand Types (matching Zemax conventions)."""
    # Spot Size Operands
    RMS_SPOT_RADIUS = "TRAC"      # Total RMS Spot Radius on Surface (mm)
    RMS_SPOT_X = "RSX"            # RMS Spot X Size on Surface (mm)
    RMS_SPOT_Y = "RSY"            # RMS Spot Y Size on Surface (mm)
    GEO_SPOT_RADIUS = "GEOR"      # 100% Geometric Encircled Spot Radius (mm)
    GEO_SPOT_X = "GEOX"           # 100% Geometric Spot X Half-Width (mm)
    GEO_SPOT_Y = "GEOY"           # 100% Geometric Spot Y Half-Width (mm)
    SPOT_TARGET_SIZE = "SPTR"     # Target Spot Size / Diameter (mm)

    # Beam & Ray Path Operands
    SPECTROMETER_SPREAD = "SPRD"  # Spectral Dispersion Spread delta_X (mm)
    ANGULAR_SPREAD = "ANGL"       # Ray Angular Divergence Variance (rad)
    TELECENTRICITY = "TELE"       # Deviation from Normal Incidence (rad)

    # Aberration Operands (Zemax 3rd-Order Seidel & Color)
    SEIDEL_S1 = "SPHA"            # Spherical Aberration S_I (mm)
    SEIDEL_S2 = "COMA"            # Coma S_II (mm)
    SEIDEL_S3 = "ASTI"            # Astigmatism S_III (mm)
    SEIDEL_S4 = "PETZ"            # Petzval Curvature S_IV (mm)
    SEIDEL_S5 = "DIST"            # Distortion S_V (mm)
    AXIAL_COLOR = "AXCL"          # Longitudinal Chromatic Aberration C_L (mm)
    LATERAL_COLOR = "LACL"        # Transverse Chromatic Aberration C_T (mm)
    WAVEFRONT_OPD = "OPDX"        # RMS Optical Path Difference (waves)

    # First-Order & Mechanical Operands
    TOTAL_TRACK = "TOTR"          # Total System Physical Length Surface 1 to Image (mm)
    TOTR = "TOTR"                 # Zemax Total Track length Surface 1 to Image (mm)
    CENTER_THICKNESS = "CTMN"     # Center Thickness of Surface (mm)
    AIR_GAP = "AIRG"              # Air Gap Distance between Surfaces (mm)
    EFFL = "EFFL"                 # Effective Focal Length (mm) via paraxial matrix power Phi = -C
    MNCT = "MNCT"                 # Minimum Center Thickness (mm)
    MXCT = "MXCT"                 # Maximum Center Thickness (mm)
    MNET = "MNET"                 # Minimum Edge Thickness (mm) at clear semi-diameter
    MXET = "MXET"                 # Maximum Edge Thickness (mm) at clear semi-diameter
    MNEA = "MNEA"                 # Minimum Air Edge Clearance (mm) between elements
    EDGE_THICKNESS = "ET"         # Edge Thickness (mm) at clear semi-diameter


@dataclass
class MeritOperand:
    """
    Individual Optimization Operand in Merit Function.
    Evaluates specific optical target against actual ray trace or geometric property.
    """
    operand_type: OperandType
    surface_index: int = -1       # Target surface index (-1 for detector / last surface)
    target: float = 0.0           # Target value in mm, rad, or waves (0.0 for minimization)
    weight: float = 1.0           # Optimization weight / priority (W_i >= 0)
    wavelength: Optional[float] = None  # Specific wavelength in um (None for polychromatic)
    field_index: Optional[int] = None   # Specific field index (None for composite across all fields)
    reference: str = "centroid"         # Reference origin: "centroid" or "chief_ray"
    
    # Live evaluation cache
    value: float = 0.0
    diff: float = 0.0
    contribution: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.operand_type.name,
            "code": self.operand_type.value,
            "surface_index": self.surface_index,
            "target": self.target,
            "weight": self.weight,
            "wavelength": self.wavelength,
            "field_index": self.field_index,
            "reference": self.reference,
            "value": self.value,
            "contribution": self.contribution
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MeritOperand':
        t_name = d.get("type", "RMS_SPOT_RADIUS")
        op_type = OperandType[t_name] if t_name in OperandType.__members__ else OperandType.RMS_SPOT_RADIUS
        return cls(
            operand_type=op_type,
            surface_index=d.get("surface_index", -1),
            target=float(d.get("target", 0.0)),
            weight=float(d.get("weight", 1.0)),
            wavelength=d.get("wavelength", None),
            field_index=d.get("field_index", None),
            reference=d.get("reference", "centroid")
        )


class MeritFunction:
    """
    Zemax-style Multi-Criteria Merit Function Manager.
    Computes root-sum-square weighted error across all operands:
    MF = sqrt( sum( W_i * (Value_i - Target_i)^2 ) / sum(W_i) ) + Penalties
    """
    def __init__(self, operands: Optional[List[MeritOperand]] = None, penalty_weight: float = 1000.0):
        self.operands: List[MeritOperand] = operands or []
        self.penalty_weight = float(penalty_weight)
        self.last_merit: float = 0.0
        self.last_vignetted_fraction: float = 0.0

    def add_operand(
        self,
        operand_type: Union[OperandType, str],
        surface_index: int = -1,
        target: float = 0.0,
        weight: float = 1.0,
        wavelength: Optional[float] = None,
        field_index: Optional[int] = None,
        reference: str = "centroid"
    ) -> MeritOperand:
        """Add a new operand to the merit function."""
        if isinstance(operand_type, str):
            if operand_type in OperandType.__members__:
                op_enum = OperandType[operand_type]
            else:
                # Try code matching (e.g. 'TRAC', 'RSX', 'RSY', 'SPRD')
                match = next((e for e in OperandType if e.value == operand_type.upper()), OperandType.RMS_SPOT_RADIUS)
                op_enum = match
        else:
            op_enum = operand_type

        op = MeritOperand(
            operand_type=op_enum,
            surface_index=surface_index,
            target=target,
            weight=weight,
            wavelength=wavelength,
            field_index=field_index,
            reference=reference
        )
        self.operands.append(op)
        return op

    def clear(self):
        """Remove all operands."""
        self.operands.clear()

    def evaluate(
        self,
        system: Any,
        traced_rays: List[Ray],
        wavelengths: Optional[List[float]] = None,
        system_config: Optional[Any] = None,
    ) -> float:
        """
        Evaluate all operands using ray trace history and system stack.
        Partitions ray hits by (surface_index, field_idx, wavelength) to evaluate
        true per-field spot diagrams and RMS combination across fields.
        Returns total scalar merit function value.
        """
        from optics_app.analysis.spot import SpotDiagram

        if not self.operands:
            # Default fallback: RMS Spot Radius on detector
            det_idx = len(system.stack.surfaces) - 1
            for idx, s in enumerate(system.stack.surfaces):
                if s.surface_type == SurfaceType.DETECTOR:
                    det_idx = idx
                    break
            self.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=det_idx, target=0.0, weight=1.0)

        stack = system.stack
        total_rays = len(traced_rays)
        valid_rays = sum(1 for r in traced_rays if len(r.history) > 0 and not r.vignetted)
        vignetted_frac = (total_rays - valid_rays) / total_rays if total_rays > 0 else 1.0
        self.last_vignetted_fraction = vignetted_frac

        # Organize ray hits by (surface_index, field_idx, wavelength)
        # surface_hits: Dict[Tuple[int, int, float], List[Dict[str, Any]]]
        surface_hits: Dict[Tuple[int, int, float], List[Dict[str, Any]]] = {}
        chief_hits: Dict[Tuple[int, int, float], Tuple[float, float]] = {}
        chief_phases: Dict[Tuple[int, int, float], float] = {}

        for r in traced_rays:
            f_idx = int(getattr(r, "field_idx", 0))
            is_chief = bool(getattr(r, "is_chief", False))
            wl = round(float(r.wavelength), 4)
            p_weight = float(getattr(r, "pupil_weight", 1.0))
            r_phase = float(getattr(r, "phase", 0.0))

            for h in r.history:
                s_idx = h.get("surface_index", -1)
                key = (s_idx, f_idx, wl)
                if key not in surface_hits:
                    surface_hits[key] = []

                xl = float(h.get("x_local", h.get("x", 0.0)))
                yl = float(h.get("y_local", h.get("y", 0.0)))
                dx = float(h.get("dir_x", r.dir[0]))
                dy = float(h.get("dir_y", r.dir[1]))
                dz = float(h.get("dir_z", r.dir[2]))

                hit_entry = {
                    "x": xl, "y": yl,
                    "dx": dx, "dy": dy, "dz": dz,
                    "is_chief": is_chief,
                    "weight": p_weight,
                    "phase": r_phase
                }
                surface_hits[key].append(hit_entry)
                if is_chief and key not in chief_hits:
                    chief_hits[key] = (xl, yl)
                    chief_phases[key] = r_phase

        all_field_indices = sorted(list(set(k[1] for k in surface_hits.keys())))
        if not all_field_indices:
            all_field_indices = [0]

        def get_field_weight(f_idx: int) -> float:
            if system_config is not None and hasattr(system_config, "fields") and system_config.fields:
                if 0 <= f_idx < len(system_config.fields):
                    return float(system_config.fields[f_idx].weight)
            if len(all_field_indices) == 3:
                standard_weights = [1.0, 2.0, 1.0]
                if f_idx < 3:
                    return standard_weights[f_idx]
            return 1.0

        primary_wl = wavelengths[0] if (wavelengths and len(wavelengths) > 0) else 0.5876
        has_stop = False
        if len(stack.surfaces) > 1:
            for idx, s in enumerate(stack.surfaces):
                if idx == 0 and len(stack.surfaces) > 1:
                    continue
                if (
                    getattr(s, "is_stop", False)
                    or getattr(s, "is_aperture_stop", False)
                    or getattr(s, "surface_type", None) == SurfaceType.STOP
                    or "STOP" in str(getattr(s, "surface_type", "")).upper()
                    or "STOP" in str(getattr(s, "name", "")).upper()
                    or "(STO)" in str(getattr(s, "name", "")).upper()
                ):
                    has_stop = True
                    break

        if has_stop:
            try:
                from optics_app.engine.pupil_solve import calculate_entrance_pupil
            except ImportError:
                from v2.engine.pupil_solve import calculate_entrance_pupil
            _, r_ep_init, _ = calculate_entrance_pupil(stack.surfaces, wavelength_um=primary_wl)
            r_p = float(r_ep_init)
        elif system_config is not None and getattr(system_config, "aperture_value", 0) > 0:
            r_p = float(system_config.aperture_value) / 2.0
            if len(stack.surfaces) > 1:
                s1_sd = float(getattr(stack.surfaces[1], "semi_diameter", 0.0))
                if 0 < s1_sd < r_p:
                    r_p = min(r_p, s1_sd * 0.8)
        elif len(stack.surfaces) > 1 and getattr(stack.surfaces[1], "semi_diameter", 0) > 0:
            s1_sd = float(stack.surfaces[1].semi_diameter)
            r_p = min(5.0, s1_sd * 0.8) if s1_sd > 0 else 5.0
        else:
            r_p = 5.0

        working_f_num = 4.0
        try:
            from optics_app.engine.matrix_tracer import MatrixTracer
            tracer = MatrixTracer()
            opt_surfs = stack.surfaces[:-1] if len(stack.surfaces) > 1 else stack.surfaces
            if len(opt_surfs) > 1 and opt_surfs[0].surface_type == SurfaceType.OBJECT and not opt_surfs[0].glass:
                opt_surfs = opt_surfs[1:]
            mat = tracer.build_from_system(opt_surfs, wavelength_um=primary_wl)
            if abs(mat.C) > 1e-12:
                efl = -1.0 / mat.C
                if abs(efl) > 1e-3 and r_p > 0:
                    working_f_num = abs(efl) / (2.0 * r_p)
        except Exception:
            pass

        if working_f_num == 4.0:
            try:
                from optics_app.engine.first_order import calculate_first_order_data
                fo = calculate_first_order_data(stack.surfaces, wavelength_um=primary_wl)
                wf = fo.get("working_f_number", float("inf"))
                if not math.isinf(wf) and wf > 0.1:
                    working_f_num = float(wf)
            except Exception:
                pass

        # Evaluate each operand
        sum_weighted_sq_err = 0.0
        sum_weights = 0.0

        for op in self.operands:
            s_idx = op.surface_index
            if s_idx < 0:
                s_idx = len(stack.surfaces) + s_idx

            val = 0.0
            diff = None

            # First-order & Mechanical / Boundary operands
            if op.operand_type in (OperandType.TOTAL_TRACK, OperandType.TOTR):
                st0 = getattr(stack.surfaces[0], "surface_type", None) if len(stack.surfaces) > 0 else None
                is_obj0 = (st0 == "Object" or getattr(st0, "name", None) == "OBJECT" or getattr(st0, "value", None) == "Object" or str(st0).lower() in ("object", "surfacetype.object"))
                start_s = 1 if len(stack.surfaces) > 1 and is_obj0 else 0
                val = sum(max(0.0, s.thickness) for s in stack.surfaces[start_s:])
                diff = val - op.target if op.target > 0.0 else val
            elif op.operand_type == OperandType.EFFL:
                try:
                    from optics_app.engine.matrix_tracer import MatrixTracer
                    tracer = MatrixTracer()
                    end_opt = s_idx if (0 < s_idx < len(stack.surfaces)) else len(stack.surfaces) - 1
                    opt_surfs = stack.surfaces[:end_opt]
                    if len(opt_surfs) > 1:
                        st0 = getattr(opt_surfs[0], "surface_type", None)
                        is_obj0 = (st0 == "Object" or getattr(st0, "name", None) == "OBJECT" or str(st0).lower() in ("object", "surfacetype.object"))
                        if is_obj0 and not opt_surfs[0].glass:
                            opt_surfs = opt_surfs[1:]
                    mat = tracer.build_from_system(opt_surfs, wavelength_um=primary_wl)
                    val = -1.0 / mat.C if abs(mat.C) > 1e-12 else float("inf")
                    if op.target > 0 and not math.isinf(val):
                        diff = val - op.target
                    elif op.target > 0 and abs(mat.C) <= 1e-12:
                        diff = 1000.0
                    else:
                        diff = val
                except Exception:
                    val = float("inf")
                    diff = 1000.0
            elif op.operand_type in (OperandType.CENTER_THICKNESS, OperandType.MNCT, OperandType.MXCT):
                if 0 <= s_idx < len(stack.surfaces):
                    val = float(stack.surfaces[s_idx].thickness)
                else:
                    glass_surfs = [s for s in stack.surfaces if getattr(s, "is_glass", False)]
                    if not glass_surfs:
                        glass_surfs = [s for s in stack.surfaces if str(getattr(s, "glass", "AIR") or "AIR").upper() not in ("AIR", "MIRROR", "MIRR", "VACUUM", "")]
                    if op.operand_type == OperandType.MNCT:
                        val = min((s.thickness for s in glass_surfs), default=0.0)
                    elif op.operand_type == OperandType.MXCT:
                        val = max((s.thickness for s in glass_surfs), default=0.0)
                    else:
                        val = stack.surfaces[s_idx].thickness if 0 <= s_idx < len(stack.surfaces) else 0.0
                if op.operand_type == OperandType.MNCT:
                    diff = max(0.0, op.target - val)
                elif op.operand_type == OperandType.MXCT:
                    diff = max(0.0, val - op.target)
                else:
                    diff = val - op.target if op.target > 0.0 else val
            elif op.operand_type in (OperandType.MNET, OperandType.MXET, OperandType.EDGE_THICKNESS):
                if 0 <= s_idx < len(stack.surfaces) - 1:
                    s = stack.surfaces[s_idx]
                    s_next = stack.surfaces[s_idx + 1]
                    r_edge = min(s.semi_diameter, s_next.semi_diameter)
                    sag1 = s.sag(r_edge)
                    sag2 = s_next.sag(r_edge)
                    val = float(s.thickness + sag2 - sag1)
                else:
                    ets = []
                    for idx_el, s_el in enumerate(stack.surfaces[:-1]):
                        is_g = getattr(s_el, "is_glass", False) or (str(getattr(s_el, "glass", "AIR") or "AIR").upper() not in ("AIR", "MIRROR", "MIRR", "VACUUM", ""))
                        if is_g:
                            s_nxt = stack.surfaces[idx_el + 1]
                            r_ed = min(s_el.semi_diameter, s_nxt.semi_diameter)
                            ets.append(float(s_el.thickness + s_nxt.sag(r_ed) - s_el.sag(r_ed)))
                    val = min(ets) if (ets and op.operand_type != OperandType.MXET) else (max(ets) if ets else 0.0)
                if op.operand_type == OperandType.MNET:
                    diff = max(0.0, op.target - val)
                elif op.operand_type == OperandType.MXET:
                    diff = max(0.0, val - op.target)
                else:
                    diff = val - op.target if op.target > 0.0 else val
            elif op.operand_type == OperandType.MNEA:
                if 0 <= s_idx < len(stack.surfaces) - 1:
                    s = stack.surfaces[s_idx]
                    s_next = stack.surfaces[s_idx + 1]
                    r_edge = min(s.semi_diameter, s_next.semi_diameter)
                    sag1 = s.sag(r_edge)
                    sag2 = s_next.sag(r_edge)
                    val = float(s.thickness + sag2 - sag1)
                else:
                    et_airs = []
                    for idx_a, s_a in enumerate(stack.surfaces[:-1]):
                        is_g = getattr(s_a, "is_glass", False) or (str(getattr(s_a, "glass", "AIR") or "AIR").upper() not in ("AIR", "MIRROR", "MIRR", "VACUUM", ""))
                        st_a = getattr(s_a, "surface_type", None)
                        is_obj_or_det_a = (st_a == "Object" or getattr(st_a, "name", None) in ("OBJECT", "DETECTOR") or str(st_a).lower() in ("object", "detector"))
                        if not is_g and not is_obj_or_det_a:
                            s_nxt = stack.surfaces[idx_a + 1]
                            st_nxt = getattr(s_nxt, "surface_type", None)
                            is_det_nxt = (st_nxt == "Detector" or getattr(st_nxt, "name", None) == "DETECTOR" or str(st_nxt).lower() == "detector")
                            if not is_det_nxt:
                                r_ed = min(s_a.semi_diameter, s_nxt.semi_diameter)
                                et_airs.append(float(s_a.thickness + s_nxt.sag(r_ed) - s_a.sag(r_ed)))
                    val = min(et_airs) if et_airs else 0.0
                diff = max(0.0, op.target - val)
            elif op.operand_type == OperandType.AIR_GAP:
                val = float(stack.surfaces[s_idx].thickness) if 0 <= s_idx < len(stack.surfaces) else 0.0
                diff = val - op.target if op.target > 0.0 else val
            elif op.operand_type in (
                OperandType.SEIDEL_S1, OperandType.SEIDEL_S2, OperandType.SEIDEL_S3,
                OperandType.SEIDEL_S4, OperandType.SEIDEL_S5, OperandType.AXIAL_COLOR,
                OperandType.LATERAL_COLOR
            ):
                try:
                    from optics_app.analysis.aberration import calculate_seidel_coefficients
                    seidel = calculate_seidel_coefficients(stack, wavelength_um=primary_wl)
                    if op.operand_type == OperandType.SEIDEL_S1:
                        val = seidel.S_I
                    elif op.operand_type == OperandType.SEIDEL_S2:
                        val = seidel.S_II
                    elif op.operand_type == OperandType.SEIDEL_S3:
                        val = seidel.S_III
                    elif op.operand_type == OperandType.SEIDEL_S4:
                        val = seidel.S_IV
                    elif op.operand_type == OperandType.SEIDEL_S5:
                        val = seidel.S_V
                    elif op.operand_type == OperandType.AXIAL_COLOR:
                        val = seidel.C_L
                    elif op.operand_type == OperandType.LATERAL_COLOR:
                        val = seidel.C_T
                except Exception:
                    val = 0.0
            else:
                # Ray trace dependent operands evaluated per-field
                target_f_indices = [op.field_index] if (op.field_index is not None) else all_field_indices
                field_evals = []  # List of (f_weight, wl_weight, metric_val)

                for f_idx in target_f_indices:
                    f_wt = get_field_weight(f_idx)
                    matching_keys = [k for k in surface_hits.keys() if k[0] == s_idx and k[1] == f_idx]
                    if op.wavelength is not None:
                        target_wl = round(float(op.wavelength), 4)
                        matching_keys = [k for k in matching_keys if abs(k[2] - target_wl) < 1e-4]

                    if not matching_keys:
                        field_evals.append((f_wt, 1.0, 100.0))
                        continue

                    for key in matching_keys:
                        wl_val = key[2]
                        hits = surface_hits[key]
                        if not hits:
                            field_evals.append((f_wt, 1.0, 100.0))
                            continue

                        xs = [h["x"] for h in hits]
                        ys = [h["y"] for h in hits]
                        dxs = [h["dx"] for h in hits]
                        dys = [h["dy"] for h in hits]
                        dzs = [h["dz"] for h in hits]

                        chief_coord = chief_hits.get(key, None)
                        ref_mode = getattr(op, "reference", "centroid")

                        spot = SpotDiagram(
                            xs, ys,
                            wavelength_um=wl_val,
                            f_number=working_f_num,
                            reference=ref_mode,
                            chief_ray_coord=chief_coord
                        )

                        if op.operand_type == OperandType.RMS_SPOT_RADIUS:
                            m_val = spot.calculate_rms_radius()
                        elif op.operand_type == OperandType.WAVEFRONT_OPD:
                            if len(hits) >= 2:
                                s_rms = spot.calculate_rms_radius()
                                wl_mm = wl_val * 1e-3 if wl_val > 0.01 else wl_val
                                m_val = s_rms / (8.0 * working_f_num * max(1e-6, wl_mm))
                            else:
                                m_val = 0.0
                        elif op.operand_type == OperandType.RMS_SPOT_X:
                            m_val = spot.calculate_rms_x()
                        elif op.operand_type == OperandType.RMS_SPOT_Y:
                            m_val = spot.calculate_rms_y()
                        elif op.operand_type == OperandType.GEO_SPOT_RADIUS:
                            m_val = spot.calculate_geo_radius()
                        elif op.operand_type == OperandType.GEO_SPOT_X:
                            m_val = spot.calculate_geo_x()
                        elif op.operand_type == OperandType.GEO_SPOT_Y:
                            m_val = spot.calculate_geo_y()
                        elif op.operand_type == OperandType.SPOT_TARGET_SIZE:
                            m_val = spot.calculate_rms_radius()
                        elif op.operand_type == OperandType.SPECTROMETER_SPREAD:
                            m_val = float(max(xs) - min(xs)) if xs else 0.0
                        elif op.operand_type == OperandType.ANGULAR_SPREAD:
                            m_val = float(np.sqrt(np.var(dxs) + np.var(dys) + np.var(dzs))) * 1e4
                        elif op.operand_type == OperandType.TELECENTRICITY:
                            dev_sq = np.mean(np.array(dxs)**2 + np.array(dys)**2 + (np.array(dzs) - 1.0)**2)
                            m_val = float(np.sqrt(dev_sq)) * 1e4
                        else:
                            m_val = spot.calculate_rms_radius()

                        field_evals.append((f_wt, 1.0, m_val))

                    # Special handling for SPECTROMETER_SPREAD across wavelengths for this field:
                    if op.operand_type == OperandType.SPECTROMETER_SPREAD and len(matching_keys) >= 2:
                        wls_sorted = sorted(matching_keys, key=lambda k: k[2])
                        min_hits = surface_hits[wls_sorted[0]]
                        max_hits = surface_hits[wls_sorted[-1]]
                        c_min_x = np.mean([h["x"] for h in min_hits]) if min_hits else 0.0
                        c_max_x = np.mean([h["x"] for h in max_hits]) if max_hits else 0.0
                        spec_spread = float(abs(c_max_x - c_min_x))
                        field_evals = [(f_wt, 1.0, spec_spread)]

                # Combine across fields and wavelengths
                if not field_evals:
                    val = 100.0
                elif op.operand_type in (OperandType.GEO_SPOT_RADIUS, OperandType.GEO_SPOT_X, OperandType.GEO_SPOT_Y):
                    val = max(fe[2] for fe in field_evals)
                else:
                    sum_weighted_sq = sum(fe[0] * fe[1] * (fe[2]**2) for fe in field_evals)
                    sum_comb_weights = sum(fe[0] * fe[1] for fe in field_evals)
                    val = math.sqrt(sum_weighted_sq / sum_comb_weights) if sum_comb_weights > 0 else 0.0

            if diff is None:
                diff = val - op.target
            op.value = float(val)
            op.diff = float(diff)
            weighted_err_sq = op.weight * (op.diff ** 2)
            op.contribution = float(weighted_err_sq)

            sum_weighted_sq_err += weighted_err_sq
            sum_weights += op.weight

        base_merit = math.sqrt(sum_weighted_sq_err / sum_weights) if sum_weights > 0 else 0.0
        
        # Penalties: ray loss & physical collision / interference
        penalty = self.penalty_weight * vignetted_frac
        
        interference_count = sum(1 for r in traced_rays if getattr(r, 'interference', False))
        if interference_count > 0:
            penalty += 5000.0 * interference_count

        total_mf = float(base_merit + penalty)
        self.last_merit = total_mf
        self.last_base_merit = float(base_merit)
        self.last_penalty = float(penalty)
        return total_mf

    def get_breakdown(self) -> List[Dict[str, Any]]:
        """Return structured summary of each operand for table display in GUI."""
        total_contrib = sum(op.contribution for op in self.operands)
        rows = []
        for op in self.operands:
            pct = (op.contribution / total_contrib * 100.0) if total_contrib > 0 else 0.0
            rows.append({
                "type": op.operand_type.name,
                "code": op.operand_type.value,
                "surface": op.surface_index,
                "target": op.target,
                "value": op.value,
                "diff": op.diff,
                "weight": op.weight,
                "contribution_pct": pct
            })
        return rows

    # --- Preset Factory Builders ---

    @classmethod
    def create_rms_spot_default(cls, target_surface: int = -1, target_radius: float = 0.0) -> 'MeritFunction':
        """Standard Zemax default merit function minimizing Total RMS Spot Radius."""
        mf = cls()
        mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=target_surface, target=target_radius, weight=1.0)
        return mf

    @classmethod
    def create_rms_spot_xy_default(
        cls,
        target_surface: int = -1,
        weight_x: float = 1.0,
        weight_y: float = 1.0,
        target_x: float = 0.0,
        target_y: float = 0.0
    ) -> 'MeritFunction':
        """Zemax RSX + RSY dual merit function for astigmatic / spectrometer imaging."""
        mf = cls()
        mf.add_operand(OperandType.RMS_SPOT_X, surface_index=target_surface, target=target_x, weight=weight_x)
        mf.add_operand(OperandType.RMS_SPOT_Y, surface_index=target_surface, target=target_y, weight=weight_y)
        return mf

    @classmethod
    def create_spectrometer_default(
        cls,
        dmd_surface: int = 8,
        target_dmd_spread: float = 5.0,
        det_surface: int = 14,
        target_det_spot: float = 0.0,
        weight_spread: float = 10.0,
        weight_det: float = 1.0
    ) -> 'MeritFunction':
        """Custom spectrometer merit function (DMD dispersion spread + Detector focus)."""
        mf = cls()
        # 1. Enforce linear dispersion on DMD
        mf.add_operand(OperandType.SPECTROMETER_SPREAD, surface_index=dmd_surface, target=target_dmd_spread, weight=weight_spread)
        # 2. Minimize RMS spot on detector
        mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=det_surface, target=target_det_spot, weight=weight_det)
        return mf

    @classmethod
    def create_multi_surface_spot_default(
        cls,
        surface_weights: List[Tuple[int, float]]
    ) -> 'MeritFunction':
        """Merit function evaluating spot sizes across multiple intermediate surfaces."""
        mf = cls()
        for s_idx, wt in surface_weights:
            mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=s_idx, target=0.0, weight=wt)
        return mf

    @classmethod
    def create_seidel_default(cls, weight_s1: float = 1.0, weight_color: float = 1.0) -> 'MeritFunction':
        """Merit function penalizing primary 3rd-order Seidel aberrations and chromatic focal shift."""
        mf = cls()
        mf.add_operand(OperandType.SEIDEL_S1, target=0.0, weight=weight_s1)
        mf.add_operand(OperandType.SEIDEL_S2, target=0.0, weight=0.5 * weight_s1)
        mf.add_operand(OperandType.SEIDEL_S3, target=0.0, weight=0.5 * weight_s1)
        mf.add_operand(OperandType.AXIAL_COLOR, target=0.0, weight=weight_color)
        mf.add_operand(OperandType.RMS_SPOT_RADIUS, target=0.0, weight=1.0)
        return mf

    @classmethod
    def create_wavefront_default(cls, target_surface: int = -1, target_opd: float = 0.0) -> 'MeritFunction':
        """Merit function minimizing RMS Wavefront Error (OPD in waves) on detector."""
        mf = cls()
        mf.add_operand(OperandType.WAVEFRONT_OPD, surface_index=target_surface, target=target_opd, weight=1.0)
        return mf

    @classmethod
    def create_zemax_default(
        cls,
        target_surface: int = -1,
        target_efl: Optional[float] = None,
        min_ct: float = 1.5,
        max_ct: float = 18.0,
        min_et: float = 1.0,
        min_air_edge: float = 0.5,
        max_totr: Optional[float] = None,
        weight_spot: float = 1.0,
        weight_boundary: float = 10.0
    ) -> 'MeritFunction':
        """Standard Zemax Default Merit Function with multi-criteria spot and boundary operands."""
        mf = cls()
        mf.add_operand(OperandType.RMS_SPOT_RADIUS, surface_index=target_surface, target=0.0, weight=weight_spot)
        if target_efl is not None and target_efl > 0:
            mf.add_operand(OperandType.EFFL, target=target_efl, weight=weight_boundary)
        mf.add_operand(OperandType.MNCT, target=min_ct, weight=weight_boundary)
        mf.add_operand(OperandType.MXCT, target=max_ct, weight=weight_boundary)
        mf.add_operand(OperandType.MNET, target=min_et, weight=weight_boundary)
        mf.add_operand(OperandType.MNEA, target=min_air_edge, weight=weight_boundary)
        if max_totr is not None and max_totr > 0:
            mf.add_operand(OperandType.TOTR, target=max_totr, weight=weight_boundary)
        return mf


