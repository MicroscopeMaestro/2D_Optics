import math
"""
Lens Data Editor (LDE) QAbstractTableModel Implementation.
Provides 2-way live binding with SurfaceStack, numerical validation, and Qt signal emissions.
Extended for Requirement R5 with Aperture Type and Aperture Params columns.
"""
from typing import List, Any, Optional, Union, Tuple, Dict

import numpy as np

from optics_app.gui.qt_compat import (
    QAbstractTableModel, QWidget, QTableView, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    Qt, QModelIndex, pyqtSignal, QT_AVAILABLE, DummySignal, QHeaderView, QAbstractItemView
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack


class LensDataTableModel(QAbstractTableModel if QT_AVAILABLE else object):
    if QT_AVAILABLE:
        system_updated = pyqtSignal()
    else:
        system_updated = None

    COLUMNS = [
        "Surface", "Type", "Radius (mm)", "Thickness (mm)",
        "Glass", "Semi-Diameter (mm)", "Conic", "Decenter X",
        "Decenter Y", "Tilt X", "Tilt Y", "Tilt Z", "Grating Lines", "CAD File",
        "Aperture Type", "Aperture Params"
    ]

    def __init__(self, stack: Optional[SurfaceStack] = None, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)
        else:
            self.system_updated = DummySignal()

        self._data: List[List[Any]] = []
        self._surfaces: List[Surface] = []
        self._stack: Optional[SurfaceStack] = None
        self.update_signals_emitted: int = 0
        self.undo_manager = None
        self.system_config = None
        self._unlocked_lens_groups: set = set()
        # Default component name mode is False (standard individual surface-by-surface view)
        self._component_name_mode: bool = False
        self.dark_mode: bool = False

        if stack is not None:
            self.set_surface_stack(stack)
        else:
            self._init_default_system()

    def set_dark_mode(self, enabled: bool) -> None:
        """Toggle dark mode styling for the Lens Data Editor model."""
        self.dark_mode = bool(enabled)
        if QT_AVAILABLE and hasattr(self, "layoutChanged"):
            self.layoutChanged.emit()

    def component_name_mode(self) -> bool:
        return bool(self._component_name_mode)

    def set_component_name_mode(self, merged: bool) -> None:
        """Switch the first LDE column between component and surface names."""
        merged = bool(merged)
        if self._component_name_mode == merged:
            return
        self._component_name_mode = merged
        if QT_AVAILABLE and hasattr(self, "layoutChanged"):
            self.layoutChanged.emit()

    def get_rays(self) -> List[Any]:
        """Return currently cached traced rays for 2D/3D visualizers."""
        return getattr(self, "_cached_rays", [])

    def set_rays(self, rays: List[Any]) -> None:
        """Store traced rays and notify visualizers."""
        self._cached_rays = list(rays) if rays else []

    def component_name_spans(self) -> List[Dict[str, Any]]:
        """Return contiguous physical lenses, including their rear boundary."""
        from optics_app.engine.element_colors import compute_surface_groups

        spans: List[Dict[str, Any]] = []
        groups = compute_surface_groups(self._surfaces)
        i = 0
        while i < len(self._surfaces):
            group = groups[i] if i < len(groups) else None
            if group and group.get("group_type") == "LENS" and self._surfaces[i].is_glass:
                start = i
                last = i
                while last + 1 < len(self._surfaces) and self._surfaces[last].is_glass:
                    last += 1
                group_name = str(
                    group.get("group_name", "")
                    or getattr(self._surfaces[start], "group_name", "")
                    or self._surfaces[start].name
                )
                spans.append({
                    "start": start,
                    "count": last - start + 1,
                    "end": last,
                    "name": group_name,
                    "label": f"\u25c9 {group_name}  ({last - start + 1} surfaces)",
                })
                i = last + 1
            else:
                i += 1
        return spans

    def component_span_for_row(self, row: int) -> Optional[Dict[str, Any]]:
        for span in self.component_name_spans():
            if span["start"] <= row <= span["end"]:
                return span
        return None

    def component_rows_for_row(self, row: int) -> List[int]:
        """Resolve a clicked prescription row to its complete physical lens."""
        span = self.component_span_for_row(row)
        if self._component_name_mode and span is not None:
            return list(range(span["start"], span["end"] + 1))
        return [row] if 0 <= row < len(self._surfaces) else []

    def is_lens_locked(self, row: int) -> bool:
        """
        Check if the surface at row belongs to a locked lens element.
        When locked, internal glass parameters (radius, glass, internal thickness) are protected.
        """
        if not (0 <= row < len(self._surfaces)):
            return False
        surf = self._surfaces[row]
        from optics_app.engine.element_colors import get_surface_element_group
        grp = get_surface_element_group(self._surfaces, row)
        if grp.get("group_type") != "LENS":
            return False
        
        # If surface itself is marked unlocked
        if not getattr(surf, 'is_locked', True):
            return False

        g_name = grp.get("group_name", "")
        e_id = grp.get("element_id", -1)
        if hasattr(self, '_unlocked_lens_groups'):
            if g_name in self._unlocked_lens_groups or e_id in self._unlocked_lens_groups:
                return False
        return True

    def _init_default_system(self):
        """Initialize default 3-surface optical system (Object, Stop, Image)."""
        clean_stack = SurfaceStack.create_clean_system()
        self.set_surface_stack(clean_stack, emit_signal=False)

    def reset_to_clean_system(self, emit_signal: bool = True) -> None:
        """Reset the table model to a clean 3-surface optical system (Object, Stop, Image) like Zemax."""
        clean_stack = SurfaceStack.create_clean_system()
        self.set_surface_stack(clean_stack, emit_signal=emit_signal)

    def _push_undo(self, description: str = ""):
        """Helper to record state in undo manager before modifications."""
        if hasattr(self, 'undo_manager') and self.undo_manager is not None:
            try:
                stack = self.get_surface_stack()
                sys_cfg = getattr(self, 'system_config', None)
                self.undo_manager.push_state(stack, sys_cfg, description=description)
            except Exception:
                pass

    def rowCount(self, parent=None) -> int:
        return len(self._data)

    def columnCount(self, parent=None) -> int:
        return len(self.COLUMNS)

    def headerData(self, section: int, orientation: Any = None, role: int = 0) -> Any:
        if role is None or role in (0, 2, getattr(Qt, 'DisplayRole', 0)):
            if orientation is None or orientation in (1, getattr(Qt, 'Horizontal', 1)):
                if 0 <= section < len(self.COLUMNS):
                    return self.COLUMNS[section]
            elif orientation in (2, getattr(Qt, 'Vertical', 2)):
                if 0 <= section < len(self._data):
                    stype = str(self._data[section][1]).upper() if len(self._data[section]) > 1 else ""
                    if section == 0 or "OBJECT" in stype:
                        return "0 (OBJ)"
                    elif section == len(self._data) - 1 or "IMAGE" in stype or "DETECTOR" in stype:
                        return f"{section} (IMA)"
                    elif "STOP" in stype:
                        return f"{section} (STO)"
                    else:
                        return str(section)
        return None

    def supportedDropActions(self) -> Any:
        """Allow move action for dragging and dropping surface rows."""
        if QT_AVAILABLE and hasattr(Qt, 'MoveAction'):
            return Qt.MoveAction
        return 2

    def flags(self, index: Any = None) -> Any:
        """Provide item flags supporting row dragging, editing, selection, dropping, and lens prescription protection."""
        if QT_AVAILABLE and hasattr(Qt, 'ItemIsSelectable'):
            base = Qt.ItemIsSelectable | Qt.ItemIsEnabled
            if index is not None and hasattr(index, 'isValid') and index.isValid():
                r = index.row()
                c = index.column()
                # Non-refractive objects (Stop, Detector, Object, Mirror, Grating) do not have a glass type to edit
                if c == 4 and 0 <= r < len(self._surfaces):
                    from optics_app.engine.element_colors import get_surface_element_group
                    grp = get_surface_element_group(self._surfaces, r)
                    if grp.get("group_type") in ("STOP", "DETECTOR", "OBJECT", "MIRROR", "GRATING") and not getattr(self._surfaces[r], 'is_glass', False):
                        return base | getattr(Qt, 'ItemIsDragEnabled', 8)
                return base | Qt.ItemIsEditable | getattr(Qt, 'ItemIsDragEnabled', 8)
            return base | getattr(Qt, 'ItemIsDropEnabled', 16)
        return 7

    def mimeTypes(self) -> List[str]:
        return ["application/x-optical-surface-row", "text/plain"]

    def mimeData(self, indexes: Any) -> Any:
        from optics_app.gui.qt_compat import QMimeData
        mime_data = QMimeData()
        if not indexes:
            return mime_data
        
        rows = sorted(list(set(idx.row() for idx in indexes if hasattr(idx, 'row') and idx.row() >= 0)))
        if self._component_name_mode:
            rows = sorted({component_row for row in rows for component_row in self.component_rows_for_row(row)})
        import json
        payload = json.dumps(rows).encode('utf-8')
        mime_data.setData("application/x-optical-surface-row", payload)
        if hasattr(mime_data, 'setText') and rows:
            mime_data.setText(str(rows[0]))
        return mime_data

    def canDropMimeData(self, data: Any, action: Any, row: int, column: int, parent: Any) -> bool:
        if data is None:
            return False
        if hasattr(data, 'hasFormat') and data.hasFormat("application/x-optical-surface-row"):
            return True
        if hasattr(data, 'hasText') and data.hasText():
            return True
        return False

    def dropMimeData(self, data: Any, action: Any, row: int, column: int, parent: Any) -> bool:
        if data is None:
            return False
        
        # Determine target drop row
        target_row = -1
        if row >= 0:
            target_row = row
        elif parent is not None and hasattr(parent, 'isValid') and parent.isValid():
            target_row = parent.row()
        else:
            target_row = len(self._data)

        # Extract source rows from mimeData
        src_rows = []
        if hasattr(data, 'hasFormat') and data.hasFormat("application/x-optical-surface-row"):
            import json
            try:
                raw_bytes = data.data("application/x-optical-surface-row")
                src_rows = json.loads(bytes(raw_bytes).decode('utf-8'))
            except Exception:
                pass
        
        if not src_rows and hasattr(data, 'hasText') and data.hasText():
            try:
                src_rows = [int(data.text().strip())]
            except Exception:
                pass

        if not src_rows:
            return False

        # Move the row(s) sequentially
        for src_row in src_rows:
            if 0 <= src_row < len(self._data):
                self.move_row(src_row, target_row)
                if target_row > src_row:
                    target_row -= 1

        return True

    def data(self, index, role=None, kw_role=0) -> Any:
        """
        Flexible data accessor supporting both Qt's (QModelIndex, role) 
        and custom (row, col) calls.
        """
        if isinstance(index, int) and isinstance(role, int):
            r = index
            c = role
            actual_role = kw_role if kw_role != 0 else 2
        elif hasattr(index, 'row'):
            r = index.row()
            c = index.column()
            actual_role = role if role is not None else 0
        else:
            return None

        if not (0 <= r < len(self._data)) or not (0 <= c < len(self.COLUMNS)):
            return None

        role_val = actual_role.value if hasattr(actual_role, 'value') else (int(actual_role) if isinstance(actual_role, int) else 0)

        if role_val == 8: # BackgroundRole
            try:
                from optics_app.gui.qt_compat import QBrush, QColor
                from optics_app.engine.element_colors import get_surface_element_group
                
                # Check for Zemax Variable / Solve status on specific parameter columns
                if r < len(self._surfaces):
                    surf = self._surfaces[r]
                    is_var = False
                    is_pickup = False
                    is_solve = False

                    if c == 2: # Radius
                        if getattr(surf, 'radius_variable', False):
                            is_var = True
                        elif getattr(surf, 'radius_solve_type', '') == 'Pickup':
                            is_pickup = True
                        elif 'MarginalRay' in getattr(surf, 'radius_solve_type', ''):
                            is_solve = True
                    elif c == 3: # Thickness
                        if getattr(surf, 'thickness_variable', False):
                            is_var = True
                        elif getattr(surf, 'thickness_solve_type', '') == 'Pickup':
                            is_pickup = True
                        elif 'MarginalRay' in getattr(surf, 'thickness_solve_type', ''):
                            is_solve = True
                    elif c == 6: # Conic
                        if getattr(surf, 'conic_variable', False):
                            is_var = True
                    elif c in (7, 8): # Decenters
                        attr = 'decenter_x_variable' if c == 7 else 'decenter_y_variable'
                        if getattr(surf, attr, False):
                            is_var = True
                    elif c in (9, 10, 11): # Tilts
                        attr = ['tilt_x_variable', 'tilt_y_variable', 'tilt_z_variable'][c - 9]
                        if getattr(surf, attr, False):
                            is_var = True

                    if is_var:
                        return QBrush(QColor("#0369a1" if self.dark_mode else "#bae6fd")) # Zemax cyan/blue variable badge
                    elif is_pickup:
                        return QBrush(QColor("#78350f" if self.dark_mode else "#fef08a")) # Soft amber pickup badge
                    elif is_solve:
                        return QBrush(QColor("#581c87" if self.dark_mode else "#e9d5ff")) # Soft lavender ray solve

                group = get_surface_element_group(self._surfaces, r)
                pal = group.get("palette")
                if pal and "bg_hex" in pal:
                    bg_col = pal["bg_hex"]
                    if self.dark_mode:
                        try:
                            from v2.gui.theme import DARK_ELEMENT_TINTS
                            bg_col = DARK_ELEMENT_TINTS.get(bg_col, "#1e293b")
                        except Exception:
                            bg_col = "#1e293b"
                    return QBrush(QColor(bg_col))
            except Exception:
                pass
            return None

        if role_val == 9: # ForegroundRole
            try:
                from optics_app.gui.qt_compat import QBrush, QColor
                from optics_app.engine.element_colors import get_surface_element_group
                
                if c == 0:
                    group = get_surface_element_group(self._surfaces, r)
                    pal = group.get("palette")
                    if pal and "badge_hex" in pal:
                        badge_col = pal["badge_hex"]
                        if self.dark_mode:
                            try:
                                from v2.gui.theme import DARK_BADGE_COLORS
                                badge_col = DARK_BADGE_COLORS.get(badge_col, "#38bdf8")
                            except Exception:
                                badge_col = "#38bdf8"
                        return QBrush(QColor(badge_col))
                elif self.dark_mode:
                    return QBrush(QColor("#f1f5f9"))
            except Exception:
                pass
        if role_val == 3: # ToolTipRole
            try:
                from optics_app.engine.element_colors import get_surface_element_group
                group = get_surface_element_group(self._surfaces, r)
                g_type = group.get("group_type", "")
                g_name = group.get("group_name", "")
                e_id = group.get("element_id", -1)
                pal = group.get("palette", {})
                pal_name = pal.get("name", "")
                if g_type == "LENS":
                    lock_str = "ðŸ”’ Locked Prescription (Fixed)" if self.is_lens_locked(r) else "ðŸ”“ Unlocked (Editable)"
                    return f"Surface {r}: {self._surfaces[r].name}\nElement: {g_name} ({lock_str})\nMaterial: {self._surfaces[r].glass}"
                elif g_type in ("OBJECT", "STOP", "DETECTOR"):
                    from optics_app.engine.element_colors import format_surface_material_display
                    mat = format_surface_material_display(self._surfaces[r], r, len(self._surfaces))
                    return f"Surface {r}: {self._surfaces[r].name}\nRole: {g_type} (Non-refractive)\nMaterial / Medium: {mat}"
                elif g_type in ("MIRROR", "GRATING"):
                    from optics_app.engine.element_colors import format_surface_material_display
                    mat = format_surface_material_display(self._surfaces[r], r, len(self._surfaces))
                    return f"Surface {r}: {self._surfaces[r].name}\nType: {g_type}\nMaterial: {mat}"
            except Exception:
                pass
            return None

        if actual_role in (0, 2, getattr(Qt, 'DisplayRole', 0), getattr(Qt, 'EditRole', 2)):
            if c < len(self._data[r]):
                if c == 0 and self._component_name_mode:
                    span = self.component_span_for_row(r)
                    if span is not None and r == span["start"]:
                        return span["label"]
                val = self._data[r][c]

                # Display Zemax 'V', 'P', 'S' indicator next to variable parameters on DisplayRole
                if actual_role in (0, getattr(Qt, 'DisplayRole', 0)):
                    if r < len(self._surfaces):
                        surf = self._surfaces[r]
                        if c == 2:
                            s_type = getattr(surf, 'radius_solve_type', '')
                            if s_type == "Pickup":
                                return f"{float(val):.3f} P"
                            elif 'MarginalRay' in s_type:
                                return f"{float(val):.3f} S"
                            elif getattr(surf, 'radius_variable', False):
                                return f"{float(val):.3f} V"
                        elif c == 3:
                            s_type = getattr(surf, 'thickness_solve_type', '')
                            if s_type == "Pickup":
                                return f"{float(val):.3f} P"
                            elif 'MarginalRay' in s_type:
                                return f"{float(val):.3f} S"
                            elif getattr(surf, 'thickness_variable', False):
                                return f"{float(val):.3f} V"
                        elif c == 6 and getattr(surf, 'conic_variable', False):
                            return f"{float(val):.3f} V"
                        elif c in (7, 8):
                            attr = 'decenter_x_variable' if c == 7 else 'decenter_y_variable'
                            if getattr(surf, attr, False):
                                return f"{float(val):.3f} V"
                        elif c in (9, 10, 11):
                            attr = ['tilt_x_variable', 'tilt_y_variable', 'tilt_z_variable'][c - 9]
                            if getattr(surf, attr, False):
                                return f"{float(val):.3f} V"
                        elif c == 4:
                            from optics_app.engine.element_colors import get_surface_element_group, format_surface_material_display
                            group = get_surface_element_group(self._surfaces, r)
                            g_type = group.get("group_type", "")
                            if g_type != "LENS" and not getattr(self._surfaces[r], 'is_glass', False):
                                return format_surface_material_display(self._surfaces[r], r, len(self._surfaces))
                return val
            return ""
        return None

    def setData(self, index, value, role=2) -> bool:
        """
        Flexible data setter supporting both Qt's (QModelIndex, value, role) 
        and custom (row, col, value) calls. Supports Zemax 'V' / 'F' solve flags.
        """
        if isinstance(index, int) and isinstance(value, int):
            r = index
            c = value
            val = role
            actual_role = 2
        elif hasattr(index, 'row'):
            r = index.row()
            c = index.column()
            val = value
            actual_role = role if role is not None else 2
        else:
            return False

        if not (0 <= r < len(self._data)) or not (0 <= c < len(self.COLUMNS)):
            return False

        # Support Zemax solve modifier: 'V' for variable, 'F' for fixed
        if isinstance(val, str):
            clean_str = val.strip().upper()
            if clean_str not in ["INF", "INFINITY", "+INF", "-INF"]:
                if clean_str.endswith('V'):
                    if c == 3 and r < len(self._surfaces):
                        self._surfaces[r].thickness_variable = True
                    elif c == 2 and r < len(self._surfaces):
                        self._surfaces[r].radius_variable = True
                    val = clean_str[:-1].strip()
                elif clean_str.endswith('F'):
                    if c == 3 and r < len(self._surfaces):
                        self._surfaces[r].thickness_variable = False
                    elif c == 2 and r < len(self._surfaces):
                        self._surfaces[r].radius_variable = False
                    val = clean_str[:-1].strip()

        # Numerical validation for numeric columns
        if c in [2, 3, 5, 6, 7, 8, 9, 10, 11, 12]:
            if isinstance(val, bool):
                return False
            if isinstance(val, str):
                sval = val.strip().lower()
                if sval in ["-inf", "-infinity"]:
                    return False
                elif sval in ["inf", "infinity", "+inf", "+infinity"]:
                    val = 0.0
                elif sval in ["nan", "-nan", "+nan"]:
                    return False
                else:
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        return False
            else:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    return False

            if math.isnan(val) or math.isinf(val) or np.isnan(val) or np.isinf(val):
                return False

            if c == 5 and val < 0.0:
                return False

        # Ensure row padding
        while len(self._data[r]) < len(self.COLUMNS):
            self._data[r].append("")

        col_name = self.COLUMNS[c] if 0 <= c < len(self.COLUMNS) else f"Col {c}"
        self._push_undo(f"Edit Surface {r} {col_name}")

        # Synchronize Aperture Type (Col 14), Aperture Params (Col 15), and Semi-Diameter (Col 5)
        surf_obj = self._surfaces[r] if 0 <= r < len(self._surfaces) else None
        if c == 14:
            clean_type = str(val).strip().capitalize()
            val = clean_type
            self._data[r][c] = val
            if clean_type == "Rectangular":
                if surf_obj:
                    sd_val = float(surf_obj.semi_diameter) if surf_obj.semi_diameter > 0 else 10.0
                    prev_type = str(getattr(surf_obj, 'aperture_type', 'Circular')).capitalize()
                    if prev_type == "Circular" or surf_obj.aperture_x_half <= 0 or surf_obj.aperture_y_half <= 0:
                        surf_obj.aperture_x_half = sd_val
                        surf_obj.aperture_y_half = sd_val
                    surf_obj.aperture_type = "Rectangular"
                    self._data[r][15] = f"X={surf_obj.aperture_x_half}, Y={surf_obj.aperture_y_half}"
                else:
                    self._data[r][15] = "X=10.0, Y=10.0"
            elif clean_type == "Circular":
                if surf_obj:
                    surf_obj.aperture_type = "Circular"
                    r_val = float(surf_obj.aperture_r_outer if surf_obj.aperture_r_outer > 0 else surf_obj.semi_diameter)
                    if r_val <= 0:
                        r_val = max(float(surf_obj.aperture_x_half), float(surf_obj.aperture_y_half), 10.0)
                    surf_obj.semi_diameter = r_val
                    surf_obj.aperture_r_outer = r_val
                    self._data[r][5] = r_val
                    self._data[r][15] = f"R={r_val}"
                else:
                    self._data[r][15] = "R=10.0"
            elif clean_type == "Elliptical":
                if surf_obj:
                    surf_obj.aperture_type = "Elliptical"
                    sd_val = float(surf_obj.semi_diameter) if surf_obj.semi_diameter > 0 else 10.0
                    if surf_obj.aperture_a_semi <= 0: surf_obj.aperture_a_semi = sd_val
                    if surf_obj.aperture_b_semi <= 0: surf_obj.aperture_b_semi = sd_val * 0.75
                    self._data[r][15] = f"A={surf_obj.aperture_a_semi}, B={surf_obj.aperture_b_semi}"
                else:
                    self._data[r][15] = "A=10.0, B=7.5"
            elif clean_type == "Annular":
                if surf_obj:
                    surf_obj.aperture_type = "Annular"
                    sd_val = float(surf_obj.semi_diameter) if surf_obj.semi_diameter > 0 else 10.0
                    if surf_obj.aperture_r_outer <= 0: surf_obj.aperture_r_outer = sd_val
                    self._data[r][15] = f"Rin={surf_obj.aperture_r_inner}, Rout={surf_obj.aperture_r_outer}"
                else:
                    self._data[r][15] = "Rin=0.0, Rout=10.0"
            elif clean_type == "Spider":
                if surf_obj:
                    surf_obj.aperture_type = "Spider"
                    self._data[r][15] = f"Rcnt={surf_obj.aperture_center_radius}, Legs={surf_obj.aperture_spider_count}, W={surf_obj.aperture_spider_width}, Ang={surf_obj.aperture_spider_angle}"
                else:
                    self._data[r][15] = "Rcnt=3.0, Legs=4, W=1.0, Ang=0.0"

        elif c == 15:
            self._data[r][c] = val
            if surf_obj:
                val_up = str(val).upper()
                if "X=" in val_up and "Y=" in val_up:
                    surf_obj.aperture_type = "Rectangular"
                    self._data[r][14] = "Rectangular"
                elif "R=" in val_up and "RIN=" not in val_up and "RCNT=" not in val_up:
                    surf_obj.aperture_type = "Circular"
                    self._data[r][14] = "Circular"
                surf_obj.set_aperture_params_from_str(str(val))
                self._data[r][5] = float(surf_obj.semi_diameter)

        elif c == 5:
            if surf_obj:
                str_val = str(val).strip().lower()
                if "x" in str_val:
                    try:
                        x_str, y_str = str_val.split("x")
                        hx = float(x_str) / 2.0 if x_str else 0.0
                        hy = float(y_str) / 2.0 if y_str else 0.0
                        # Update substrate shape directly
                        surf_obj.substrate_shape = "Rectangular"
                        surf_obj.semi_diameter_x = hx
                        surf_obj.semi_diameter_y = hy
                        surf_obj.semi_diameter = math.sqrt(hx**2 + hy**2)
                        self._data[r][c] = str_val
                    except ValueError:
                        self._data[r][c] = val
                        try:
                            surf_obj.semi_diameter = float(val)
                        except ValueError:
                            pass
                else:
                    try:
                        fval = float(val)
                        surf_obj.semi_diameter = fval
                        surf_obj.substrate_shape = "Auto"
                        self._data[r][c] = fval
                        if str(getattr(surf_obj, 'aperture_type', 'Circular')).capitalize() == 'Circular':
                            surf_obj.aperture_r_outer = fval
                            self._data[r][15] = f"R={fval}"
                    except ValueError:
                        self._data[r][c] = val
        elif c == 1:
            norm_type = SurfaceType.normalize(val)
            self._data[r][c] = norm_type
            val_type_up = str(val).upper()
            if surf_obj:
                prev_stype = surf_obj.surface_type
                surf_obj.surface_type = norm_type
                if prev_stype == SurfaceType.STOP and norm_type != SurfaceType.STOP:
                    surf_obj.is_stop = False
                if prev_stype == SurfaceType.COORDINATE_BREAK and norm_type != SurfaceType.COORDINATE_BREAK:
                    surf_obj.is_coord_break = False

                if norm_type == SurfaceType.MIRROR or "MIRROR" in val_type_up:
                    surf_obj.glass = "MIRROR"
                    self._data[r][4] = "MIRROR"
                elif norm_type == SurfaceType.STOP or "STOP" in val_type_up:
                    surf_obj.is_stop = True
                    if not surf_obj.glass:
                        surf_obj.glass = "AIR"
                        self._data[r][4] = "AIR"
                elif norm_type == SurfaceType.COORDINATE_BREAK or "COORDBRK" in val_type_up:
                    surf_obj.is_coord_break = True
                    surf_obj.glass = "AIR"
                    self._data[r][4] = "AIR"
                elif norm_type in (SurfaceType.DETECTOR, SurfaceType.OBJECT) or any(k in val_type_up for k in ["DETECTOR", "IMAGE", "OBJECT"]):
                    if not surf_obj.glass or surf_obj.glass.upper() in ("MIRROR", ""):
                        surf_obj.glass = "AIR"
                        self._data[r][4] = "AIR"
        elif c == 4:
            clean_glass = str(val).strip()
            up_glass = clean_glass.upper()
            if up_glass in ("MIRROR", "MIRR", "REFL"):
                clean_glass = "MIRROR"
                if surf_obj:
                    surf_obj.glass = "MIRROR"
                    if surf_obj.surface_type == SurfaceType.STANDARD:
                        surf_obj.surface_type = SurfaceType.MIRROR
                        self._data[r][1] = SurfaceType.MIRROR
            elif up_glass in ("", "AIR", "VACUUM", "NONE"):
                clean_glass = "AIR"
                if surf_obj:
                    surf_obj.glass = "AIR"
                    if surf_obj.surface_type == SurfaceType.MIRROR:
                        surf_obj.surface_type = SurfaceType.STANDARD
                        self._data[r][1] = SurfaceType.STANDARD
            else:
                try:
                    from optics_app.engine.materials import GlassCatalog
                    for g_key in GlassCatalog.GLASSES.keys():
                        if g_key.upper().replace("-", "").replace("_", "") == up_glass.replace("-", "").replace("_", ""):
                            clean_glass = g_key
                            break
                except Exception:
                    pass
                if surf_obj:
                    surf_obj.glass = clean_glass
                    if surf_obj.surface_type == SurfaceType.MIRROR:
                        surf_obj.surface_type = SurfaceType.STANDARD
                        self._data[r][1] = SurfaceType.STANDARD
            self._data[r][4] = clean_glass
        else:
            self._data[r][c] = val

        self._sync_stack_from_data()
        self.update_signals_emitted += 1

        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass

        if QT_AVAILABLE and hasattr(self, 'dataChanged'):
            try:
                if c in (1, 5, 14, 15):
                    idx_start = self.index(r, 0) if hasattr(self, 'index') else None
                    idx_end = self.index(r, len(self.COLUMNS) - 1) if hasattr(self, 'index') else None
                    if idx_start is not None and idx_end is not None:
                        self.dataChanged.emit(idx_start, idx_end)
                else:
                    idx = self.index(r, c) if hasattr(self, 'index') else None
                    if idx is not None:
                        self.dataChanged.emit(idx, idx)
            except Exception:
                pass

        return True

    def insertRows(self, position: int, count: int = 1, parent=None) -> bool:
        self._push_undo(f"Insert {count} surface(s) at index {position}")
        if position < 0 or position > len(self._data):
            position = len(self._data)

        if QT_AVAILABLE and hasattr(self, 'beginInsertRows') and parent is not None:
            try:
                self.beginInsertRows(parent, position, position + count - 1)
            except Exception:
                pass

        for i in range(count):
            row_idx = position + i
            # Inherit sequential parameters from previous surface
            if row_idx > 0 and row_idx - 1 < len(self._surfaces):
                prev_surf = self._surfaces[row_idx - 1]
                sd = float(prev_surf.semi_diameter) if prev_surf.semi_diameter > 0 else 12.7
                prev_thick = float(prev_surf.thickness)
                
                if prev_surf.is_glass:
                    # Glass element continuation/rear surface
                    new_radius = 0.0
                    new_thick = 10.0
                    new_glass = "AIR"
                    grp_name = getattr(prev_surf, 'group_name', '')
                else:
                    # Air gap insertion: smoothly split previous thickness if substantial
                    if prev_thick > 4.0:
                        half_thick = round(prev_thick / 2.0, 3)
                        prev_surf.thickness = half_thick
                        if row_idx - 1 < len(self._data) and len(self._data[row_idx - 1]) > 3:
                            self._data[row_idx - 1][3] = half_thick
                        new_thick = half_thick
                    else:
                        new_thick = 10.0
                    new_radius = 50.0
                    new_glass = "N-BK7"
                    grp_name = ""
            elif len(self._surfaces) > 0:
                sd = float(self._surfaces[0].semi_diameter) if self._surfaces[0].semi_diameter > 0 else 12.7
                new_radius = 0.0
                new_thick = 10.0
                new_glass = "AIR"
                grp_name = ""
            else:
                sd = 12.7
                new_radius = 0.0
                new_thick = 50.0
                new_glass = "AIR"
                grp_name = ""

            name = f"S{row_idx}" if row_idx > 0 else "OBJ"
            new_row = [name, "Standard", new_radius, new_thick, new_glass, sd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "", "Circular", f"R={sd}"]
            self._data.insert(row_idx, new_row)
            new_surf = Surface(
                name=name,
                surface_type=SurfaceType.STANDARD,
                radius=new_radius,
                thickness=new_thick,
                glass=new_glass,
                semi_diameter=sd,
                group_name=grp_name
            )
            self._surfaces.insert(row_idx, new_surf)

        self._reindex_surface_names()
        self._sync_stack_from_data()
        self.update_signals_emitted += 1

        if QT_AVAILABLE and hasattr(self, 'endInsertRows') and parent is not None:
            try:
                self.endInsertRows()
            except Exception:
                pass

        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass

        return True

    def removeRows(self, position: int, count: int = 1, parent=None) -> bool:
        self._push_undo(f"Delete {count} surface(s) at index {position}")
        if position < 0 or position >= len(self._data):
            return False

        if QT_AVAILABLE and hasattr(self, 'beginRemoveRows') and parent is not None:
            try:
                self.beginRemoveRows(parent, position, min(position + count - 1, len(self._data) - 1))
            except Exception:
                pass

        for _ in range(count):
            if 0 <= position < len(self._data):
                self._data.pop(position)
                if position < len(self._surfaces):
                    self._surfaces.pop(position)

        self._reindex_surface_names()
        self._sync_stack_from_data()
        self.update_signals_emitted += 1

        if QT_AVAILABLE and hasattr(self, 'endRemoveRows') and parent is not None:
            try:
                self.endRemoveRows()
            except Exception:
                pass

        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass

        return True

    def moveRow(self, sourceParent: Any, sourceRow: int, destinationParent: Any, destinationChild: int) -> bool:
        self._push_undo(f"Move surface row {sourceRow}")
        if sourceRow < 0 or sourceRow >= len(self._data):
            return False
        if destinationChild < 0 or destinationChild > len(self._data):
            return False
        if destinationChild == sourceRow or destinationChild == sourceRow + 1:
            return True

        p_src = sourceParent if sourceParent is not None else (QModelIndex() if QT_AVAILABLE else None)
        p_dst = destinationParent if destinationParent is not None else (QModelIndex() if QT_AVAILABLE else None)

        if QT_AVAILABLE and hasattr(self, 'beginMoveRows') and p_src is not None:
            try:
                self.beginMoveRows(p_src, sourceRow, sourceRow, p_dst, destinationChild)
            except Exception:
                pass

        row_data = self._data.pop(sourceRow)
        dest_idx = destinationChild if destinationChild < sourceRow else destinationChild - 1
        self._data.insert(dest_idx, row_data)

        if sourceRow < len(self._surfaces):
            surf_obj = self._surfaces.pop(sourceRow)
            dest_surf_idx = min(dest_idx, len(self._surfaces))
            self._surfaces.insert(dest_surf_idx, surf_obj)

        self._reindex_surface_names()
        self._sync_stack_from_data()
        self.update_signals_emitted += 1

        if QT_AVAILABLE and hasattr(self, 'endMoveRows') and p_src is not None:
            try:
                self.endMoveRows()
            except Exception:
                pass

        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass

        return True

    def move_row(self, source_row: int, target_row: int) -> bool:
        if target_row > source_row:
            return self.moveRow(None, source_row, None, target_row + 1)
        else:
            return self.moveRow(None, source_row, None, target_row)

    def move_surface_up(self, row: int) -> bool:
        if row <= 0 or row >= len(self._data):
            return False
        return self.moveRow(None, row, None, row - 1)

    def move_surface_down(self, row: int) -> bool:
        if row < 0 or row >= len(self._data) - 1:
            return False
        return self.moveRow(None, row, None, row + 2)

    def set_lens_locked(self, target: Any, locked: bool = True) -> bool:
        """Lock or unlock a lens element's parameters for editing."""
        if not hasattr(self, '_unlocked_lens_groups'):
            self._unlocked_lens_groups = set()
        
        from optics_app.engine.element_colors import compute_surface_groups
        groups = compute_surface_groups(self._surfaces)
        
        target_gname = None
        target_eid = None
        target_indices = set()

        if isinstance(target, int):
            if 0 <= target < len(groups) and groups[target] and groups[target].get("group_type") == "LENS":
                target_eid = groups[target].get("element_id")
                target_gname = groups[target].get("group_name")
                span = self.component_span_for_row(target)
                if span is not None:
                    target_indices.update(range(span["start"], span["end"] + 1))
        elif isinstance(target, str):
            target_gname = target.strip()

        for i, grp in enumerate(groups):
            if grp and grp.get("group_type") == "LENS":
                matches = (
                    i in target_indices if isinstance(target, int)
                    else bool(target_gname and grp.get("group_name") == target_gname)
                )
                if matches:
                    if i < len(self._surfaces):
                        self._surfaces[i].is_locked = locked

        if target_gname and not isinstance(target, int):
            if locked:
                self._unlocked_lens_groups.discard(target_gname)
            else:
                self._unlocked_lens_groups.add(target_gname)

        if target_eid is not None and not isinstance(target, int):
            if locked:
                self._unlocked_lens_groups.discard(target_eid)
            else:
                self._unlocked_lens_groups.add(target_eid)

        if QT_AVAILABLE and hasattr(self, 'layoutChanged'):
            try:
                self.layoutChanged.emit()
            except Exception:
                pass

        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass

        return True

    def delete_lens_element(self, target: Any) -> bool:
        """Delete all constituent surfaces of a lens element and update the system."""
        self._push_undo(f"Delete lens element '{target}'")
        if not self._surfaces:
            return False
        stack = self.get_surface_stack()
        matched_indices = stack.delete_element(target)
        if not matched_indices:
            return False
        self.set_surface_stack(stack)
        self._reindex_surface_names()
        self.update_signals_emitted += 1

        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass

        return True

    def flip_lens_element(self, target: Any) -> bool:
        """Flip one concrete lens assembly and refresh every prescription row."""
        self._push_undo(f"Flip lens element '{target}'")
        stack = self.get_surface_stack()
        changed = stack.flip_element(target)
        if not changed:
            return False
        self.set_surface_stack(stack)
        self.update_signals_emitted += 1
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            self.system_updated.emit()
        return True

    def set_mechanical_state(self, target: Any, state: str) -> bool:
        """Switch an actuated optical component as one undoable model edit."""
        from optics_app.engine.mechanisms import MechanicalStateController

        self._push_undo(f"Set mechanical state '{target}' to {str(state).upper()}")
        stack = self.get_surface_stack()
        MechanicalStateController(stack).set_state(target, state)
        self.set_surface_stack(stack)
        self.update_signals_emitted += 1
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass
        return True

    def replace_lens_element(self, target: Any, new_surfaces: List[Surface]) -> bool:
        """Replace target lens element with a new set of surfaces in the table model."""
        self._push_undo(f"Replace lens element '{target}'")
        stack = self.get_surface_stack()
        res = stack.replace_lens_element(target, new_surfaces)
        self.set_surface_stack(stack)
        self._reindex_surface_names()
        self.update_signals_emitted += 1
        if QT_AVAILABLE and hasattr(self, 'layoutChanged'):
            try:
                self.layoutChanged.emit()
            except Exception:
                pass
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass
        return bool(res)

    def replace_lens_from_zmx(self, target: Any, file_path: str, reverse: bool = False, tilt_x: float = 0.0, tilt_y: float = 0.0, tilt_z: float = 0.0) -> bool:
        """Replace target lens element with surfaces from a Zemax file."""
        self._push_undo(f"Replace lens element '{target}' with Zemax file")
        stack = self.get_surface_stack()
        res = stack.replace_lens_from_zmx(target, file_path, reverse=reverse, tilt_x=tilt_x, tilt_y=tilt_y, tilt_z=tilt_z)
        self.set_surface_stack(stack)
        self._reindex_surface_names()
        self.update_signals_emitted += 1
        if QT_AVAILABLE and hasattr(self, 'layoutChanged'):
            try:
                self.layoutChanged.emit()
            except Exception:
                pass
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass
        return bool(res)

    def replace_lens_from_custom_file(self, target: Any, file_path: str) -> bool:
        """Replace target lens element with surfaces from a custom JSON file."""
        self._push_undo(f"Replace lens element '{target}' with custom file")
        stack = self.get_surface_stack()
        res = stack.replace_lens_from_custom_file(target, file_path)
        self.set_surface_stack(stack)
        self._reindex_surface_names()
        self.update_signals_emitted += 1
        if QT_AVAILABLE and hasattr(self, 'layoutChanged'):
            try:
                self.layoutChanged.emit()
            except Exception:
                pass
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass
        return bool(res)

    def replace_lens_from_catalog(
        self,
        target: Any,
        part_number: str,
        reverse: bool = False,
        air_gap_after: Optional[float] = None,
        use_catalog_bfl: bool = False,
        auto_adjust_focus: bool = False,
        preserve_aperture: bool = False,
        tilt_x: Optional[float] = None,
        tilt_y: Optional[float] = None,
        tilt_z: Optional[float] = None,
    ) -> bool:
        """Replace target lens element with a standard Thorlabs catalog lens."""
        self._push_undo(f"Replace lens element '{target}' with {part_number}")
        stack = self.get_surface_stack()
        res = stack.replace_lens_from_catalog(
            target,
            part_number,
            reverse=reverse,
            air_gap_after=air_gap_after,
            use_catalog_bfl=use_catalog_bfl,
            auto_adjust_focus=auto_adjust_focus,
            preserve_aperture=preserve_aperture,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
        )
        self.set_surface_stack(stack)
        self._reindex_surface_names()
        self.update_signals_emitted += 1
        if QT_AVAILABLE and hasattr(self, 'layoutChanged'):
            try:
                self.layoutChanged.emit()
            except Exception:
                pass
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass
        return bool(res)

    def _reindex_surface_names(self):
        n = len(self._data)
        if n == 0:
            return
        import re
        for i in range(n):
            current_name = str(self._data[i][0]).strip()
            stype = str(self._data[i][1]).upper() if len(self._data[i]) > 1 else ""

            is_auto_name = (
                current_name.startswith("Surface ") or
                bool(re.match(r"^S\d+(\s*\(.*\))?$", current_name)) or
                current_name in ["OBJ", "STO", "IMA", "Object", "Stop", "Image", "Detector", "Standard", ""] or
                current_name.isdigit()
            )

            if i == 0 or "OBJECT" in stype:
                if is_auto_name:
                    self._data[i][0] = "S0 (OBJ)"
            elif i == n - 1 or "IMAGE" in stype or "DETECTOR" in stype:
                if is_auto_name:
                    self._data[i][0] = f"S{i} (IMA)"
            elif "STOP" in stype:
                if is_auto_name:
                    self._data[i][0] = f"S{i} (STO)"
            else:
                if is_auto_name:
                    self._data[i][0] = f"S{i}"

    def get_surface_stack(self) -> SurfaceStack:
        if self._stack is None:
            self._stack = SurfaceStack()
        self._stack.surfaces = list(self._surfaces)
        self._stack._update_accumulated_positions()
        return self._stack

    def set_surface_stack(self, stack: SurfaceStack, emit_signal: bool = True) -> None:
        if QT_AVAILABLE and hasattr(self, 'beginResetModel'):
            try:
                self.beginResetModel()
            except Exception:
                pass

        self._stack = stack
        self._surfaces = list(stack.surfaces)
        new_data = []
        for i, surf in enumerate(stack.surfaces):
            stype_upper = (surf.surface_type.value if hasattr(surf.surface_type, 'value') else str(surf.surface_type)).upper()
            if surf.name:
                name = surf.name
            elif i == 0 or "OBJECT" in stype_upper:
                name = "S0 (OBJ)"
            elif i == len(stack.surfaces) - 1 or "IMAGE" in stype_upper or "DETECTOR" in stype_upper:
                name = f"S{i} (IMA)"
            elif "STOP" in stype_upper:
                name = f"S{i} (STO)"
            else:
                name = f"S{i}"
            stype_str = surf.surface_type.value if hasattr(surf.surface_type, 'value') else str(surf.surface_type)
            cad_path = getattr(surf, 'cad_file_path', "") or ""
            ap_type = getattr(surf, 'aperture_type', "Circular")
            ap_str = surf.get_aperture_params_str() if hasattr(surf, 'get_aperture_params_str') else f"R={surf.semi_diameter}"
            
            sub_shape = str(getattr(surf, 'substrate_shape', '')).capitalize()
            if sub_shape == "Rectangular":
                hx = float(getattr(surf, "semi_diameter_x", 0.0))
                hy = float(getattr(surf, "semi_diameter_y", 0.0))
                sd_disp = f"{hx*2:g}x{hy*2:g}"
            else:
                sd_disp = float(surf.semi_diameter)

            new_data.append([
                name,
                stype_str,
                float(surf.radius),
                float(surf.thickness),
                str(surf.glass),
                sd_disp,
                float(surf.conic),
                float(surf.decenter_x),
                float(surf.decenter_y),
                float(surf.tilt_x),
                float(surf.tilt_y),
                float(surf.tilt_z),
                float(getattr(surf, 'grating_lines_per_mm', 0.0)),
                str(cad_path),
                str(ap_type),
                str(ap_str)
            ])
        self._data = new_data
        self.update_signals_emitted += 1

        if QT_AVAILABLE and hasattr(self, 'endResetModel'):
            try:
                self.endResetModel()
            except Exception:
                pass

        if emit_signal and hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass

    def _sync_stack_from_data(self):
        stack = self._stack if self._stack is not None else SurfaceStack()
        
        while len(self._surfaces) < len(self._data):
            self._surfaces.append(Surface())
        while len(self._surfaces) > len(self._data):
            self._surfaces.pop()

        for i, row in enumerate(self._data):
            stype_str = str(row[1]).upper()
            stype = SurfaceType.STANDARD
            if stype_str == SurfaceType.AIR_SPACE.upper():
                stype = SurfaceType.AIR_SPACE
            elif "MIRROR" in stype_str:
                stype = SurfaceType.MIRROR
            elif "ASPHERIC" in stype_str or "EVENASPH" in stype_str:
                stype = SurfaceType.ASPHERIC
            elif "GRATING" in stype_str:
                stype = SurfaceType.GRATING
            elif "STOP" in stype_str:
                stype = SurfaceType.STOP
            elif "DETECTOR" in stype_str or "IMAGE" in stype_str:
                stype = SurfaceType.DETECTOR
            elif "CAD" in stype_str:
                stype = SurfaceType.CAD
            elif "OBJECT" in stype_str:
                stype = SurfaceType.OBJECT

            grat_lines = float(row[12]) if len(row) > 12 else 0.0
            cad_path = str(row[13]) if len(row) > 13 else ""
            ap_type = str(row[14]) if len(row) > 14 else "Circular"
            ap_param_str = str(row[15]) if len(row) > 15 else ""

            surf = self._surfaces[i]
            surf.name = str(row[0])
            surf.surface_type = stype
            surf.is_stop = (stype == SurfaceType.STOP or "STOP" in stype_str or "(STO)" in surf.name.upper() or "(STOP)" in surf.name.upper())
            surf.radius = float(row[2])
            surf.thickness = float(row[3])
            surf.glass = str(row[4])
            
            # Handle string semi-diameter for rectangular substrates
            sd_val = row[5]
            if isinstance(sd_val, str) and "x" in sd_val.lower():
                try:
                    x_str, y_str = sd_val.lower().split("x")
                    hx = float(x_str) / 2.0
                    hy = float(y_str) / 2.0
                    surf.substrate_shape = "Rectangular"
                    surf.semi_diameter_x = hx
                    surf.semi_diameter_y = hy

                    surf.semi_diameter = math.sqrt(hx**2 + hy**2)
                except ValueError:
                    try:
                        surf.semi_diameter = float(sd_val)
                    except ValueError:
                        pass
            else:
                try:
                    surf.semi_diameter = float(sd_val)
                except ValueError:
                    pass
            
            surf.conic = float(row[6])
            surf.decenter_x = float(row[7])
            surf.decenter_y = float(row[8])
            surf.tilt_x = float(row[9])
            surf.tilt_y = float(row[10])
            surf.tilt_z = float(row[11])
            surf.grating_lines_per_mm = grat_lines
            setattr(surf, 'cad_file_path', cad_path)
            surf.aperture_type = ap_type
            if hasattr(surf, 'set_aperture_params_from_str') and ap_param_str:
                surf.set_aperture_params_from_str(ap_param_str)

            if str(surf.aperture_type).capitalize() == "Rectangular":
                if surf.aperture_x_half <= 0:
                    surf.aperture_x_half = surf.semi_diameter if surf.semi_diameter > 0 else 10.0
                if surf.aperture_y_half <= 0:
                    surf.aperture_y_half = surf.semi_diameter if surf.semi_diameter > 0 else 10.0
            elif str(surf.aperture_type).capitalize() == "Circular":
                if surf.aperture_r_outer <= 0:
                    surf.aperture_r_outer = surf.semi_diameter

        stack.surfaces = list(self._surfaces)
        stack._update_accumulated_positions()
        self._stack = stack

    def set_zemax_defaults(self, var_air_gaps: bool = True, fix_lenses: bool = True):
        """
        Apply Zemax OpticStudio solve/variable conventions:
        - Solid lens elements: Fixed (non-variable by default)
        - Air gaps / propagation distances: Variable degrees of freedom
        """
        for surf in self._surfaces:
            if getattr(surf, 'is_glass', False):
                if fix_lenses:
                    surf.thickness_variable = False
                    surf.radius_variable = False
            else:
                if var_air_gaps:
                    surf.thickness_variable = True
                    if surf.thickness_min <= 0:
                        surf.thickness_min = 0.5
        self._sync_stack_from_data()
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass

    def set_all_fixed(self):
        """Clear all variable flags, setting all surfaces to Fixed."""
        for surf in self._surfaces:
            surf.thickness_variable = False
            surf.radius_variable = False
            surf.conic_variable = False
            surf.radius_solve_type = "Fixed"
            surf.thickness_solve_type = "Fixed"
        self._sync_stack_from_data()
        if hasattr(self, 'layoutChanged'):
            self.layoutChanged.emit()
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass

    def make_all_radii_variable(self):
        """Set all optical surface radii as variable (for curvature optimization)."""
        for surf in self._surfaces:
            if getattr(surf, 'is_glass', False) or abs(getattr(surf, 'radius', 0.0)) > 1e-4:
                surf.radius_variable = True
                surf.radius_solve_type = "Variable"
        self._sync_stack_from_data()
        if hasattr(self, 'layoutChanged'):
            self.layoutChanged.emit()
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass

    def make_all_air_gaps_variable(self):
        """Set all air gap propagation distances as variable (for spacing optimization)."""
        for surf in self._surfaces:
            glass = str(getattr(surf, 'glass', '')).upper().strip()
            if not glass or glass in ['AIR', 'VACUUM']:
                surf.thickness_variable = True
                surf.thickness_solve_type = "Variable"
        self._sync_stack_from_data()
        if hasattr(self, 'layoutChanged'):
            self.layoutChanged.emit()
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass

    def insert_surface_after(self, row: int, surface: Optional[Surface] = None) -> int:
        """Insert a new surface after row."""
        self._push_undo(f"Insert surface after row {row}")
        stack = self.get_surface_stack()
        if surface is None:
            surface = Surface(name=f"Surface_{row+1}", thickness=10.0, glass="AIR", semi_diameter=12.7)
        insert_idx = min(len(stack.surfaces), row + 1)
        stack.insert_surface(insert_idx, surface)
        self.set_surface_stack(stack)
        self._reindex_surface_names()
        if QT_AVAILABLE:
            self.layoutChanged.emit()
        return insert_idx

    def insert_singlet_after(self, row: int, **kwargs) -> List[int]:
        """Insert a singlet lens after row."""
        self._push_undo(f"Insert singlet after row {row}")
        stack = self.get_surface_stack()
        res = stack.insert_singlet(row, **kwargs)
        self.set_surface_stack(stack)
        self._reindex_surface_names()
        if QT_AVAILABLE: self.layoutChanged.emit()
        return res

    def insert_doublet_after(self, row: int, **kwargs) -> List[int]:
        """Insert an achromatic doublet lens after row."""
        self._push_undo(f"Insert doublet after row {row}")
        stack = self.get_surface_stack()
        res = stack.insert_doublet(row, **kwargs)
        self.set_surface_stack(stack)
        self._reindex_surface_names()
        if QT_AVAILABLE: self.layoutChanged.emit()
        return res

    def insert_catalog_lens_after(
        self,
        row: int,
        part_number: str,
        air_gap_after: Optional[float] = None,
        reverse: bool = False,
    ) -> List[int]:
        """Insert a Thorlabs catalog lens after row."""
        self._push_undo(f"Insert catalog lens {part_number} after row {row}")
        stack = self.get_surface_stack()
        res = stack.insert_catalog_lens(row, part_number, air_gap_after=air_gap_after, reverse=reverse)
        self.set_surface_stack(stack)
        self._reindex_surface_names()
        if QT_AVAILABLE and hasattr(self, 'layoutChanged'):
            try:
                self.layoutChanged.emit()
            except Exception:
                pass
        return res

    def replace_with_catalog_lens(
        self,
        row: int,
        part_number: str,
        reverse: bool = False,
        air_gap_after: Optional[float] = None,
        use_catalog_bfl: bool = False,
        preserve_aperture: bool = False,
        auto_adjust_focus: bool = False,
    ) -> List[int]:
        """Replace lens at row with a catalog lens."""
        self._push_undo(f"Replace lens at row {row} with {part_number}")
        stack = self.get_surface_stack()
        res = stack.replace_with_catalog_lens(
            row,
            part_number,
            reverse=reverse,
            air_gap_after=air_gap_after,
            use_catalog_bfl=use_catalog_bfl,
            preserve_aperture=preserve_aperture,
            auto_adjust_focus=auto_adjust_focus,
        )
        self.set_surface_stack(stack)
        self._reindex_surface_names()
        if QT_AVAILABLE and hasattr(self, 'layoutChanged'):
            try:
                self.layoutChanged.emit()
            except Exception:
                pass
        return res

    def set_mirror_fold_angle(self, row: int, fold_angle_deg: float = 90.0, fold_plane: str = "YZ") -> Tuple[float, float]:
        """Set mirror fold angle on surface at row."""
        self._push_undo(f"Set mirror fold angle on surface {row}")
        stack = self.get_surface_stack()
        if 0 <= row < len(stack.surfaces):
            from optics_app.engine.mirror_positioner import MirrorPositioner
            tx, ty = MirrorPositioner.set_mirror_fold_angle(stack, row, fold_angle_deg=fold_angle_deg, fold_plane=fold_plane)
            self.set_surface_stack(stack)
            return (tx, ty)
        return (0.0, 0.0)

    def aim_mirror_to_target(self, row: int, target_idx: int = -1) -> bool:
        """Aim mirror at row towards target_idx."""
        self._push_undo(f"Aim mirror at surface {row}")
        stack = self.get_surface_stack()
        if 0 <= row < len(stack.surfaces):
            from optics_app.engine.mirror_positioner import MirrorPositioner
            res = MirrorPositioner.aim_mirror_to_target(stack, row, target_surface_index=target_idx)
            self.set_surface_stack(stack)
            return bool(res.success or res.target_error_mm < 0.1)
        return False

    def rotate_lens_element(
        self,
        row: int,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        tilt_z: float = 0.0,
        pivot: str = "front_vertex"
    ) -> None:
        """Rotate multi-surface lens element around pivot."""
        self._push_undo(f"Rotate lens element at surface {row}")
        stack = self.get_surface_stack()
        if 0 <= row < len(stack.surfaces):
            if hasattr(stack, "rotate_lens_element"):
                stack.rotate_lens_element(front_surface_index=row, tilt_x=tilt_x, tilt_y=tilt_y, tilt_z=tilt_z, pivot=pivot)
            else:
                front_idx = row
                while front_idx > 0 and getattr(stack.surfaces[front_idx - 1], 'is_glass', False):
                    front_idx -= 1
                rear_idx = front_idx
                while rear_idx < len(stack.surfaces) - 1 and getattr(stack.surfaces[rear_idx], 'is_glass', False):
                    rear_idx += 1
                for s_i in range(front_idx, rear_idx + 1):
                    s = stack.surfaces[s_i]
                    s.tilt_x = float(tilt_x)
                    s.tilt_y = float(tilt_y)
                    s.tilt_z = float(tilt_z)
                stack._update_accumulated_positions()
            self.set_surface_stack(stack)

    def decenter_lens_element(
        self,
        row: int,
        decenter_x: float = 0.0,
        decenter_y: float = 0.0,
        decenter_z: float = 0.0,
        tilt_x: Optional[float] = None,
        tilt_y: Optional[float] = None,
        tilt_z: Optional[float] = None,
        pivot: str = "front_vertex"
    ) -> None:
        """Translate / decenter multi-surface lens element as a rigid body in 3D space."""
        self._push_undo(f"Decenter lens element at surface {row}")
        stack = self.get_surface_stack()
        if 0 <= row < len(stack.surfaces):
            if hasattr(stack, "decenter_lens_element"):
                stack.decenter_lens_element(
                    front_surface_index=row,
                    decenter_x=decenter_x,
                    decenter_y=decenter_y,
                    decenter_z=decenter_z,
                    tilt_x=tilt_x,
                    tilt_y=tilt_y,
                    tilt_z=tilt_z,
                    pivot=pivot
                )
            else:
                front_idx = row
                while front_idx > 0 and getattr(stack.surfaces[front_idx - 1], 'is_glass', False):
                    front_idx -= 1
                rear_idx = front_idx
                while rear_idx < len(stack.surfaces) - 1 and getattr(stack.surfaces[rear_idx], 'is_glass', False):
                    rear_idx += 1
                for s_i in range(front_idx, rear_idx + 1):
                    s = stack.surfaces[s_i]
                    s.decenter_x = float(decenter_x)
                    s.decenter_y = float(decenter_y)
                    s.decenter_z = float(decenter_z)
                stack._update_accumulated_positions()
            self.set_surface_stack(stack)

    def set_stop_surface(self, row: int, is_stop: bool = True) -> bool:
        """
        Set or clear the Aperture Stop (STO) designation on a specific surface row.
        If is_stop is True, any previous Aperture Stop on other surfaces is converted to Standard.
        """
        stack = self.get_surface_stack()
        if not stack or not (0 <= row < len(stack.surfaces)):
            return False

        self._push_undo(f"Set surface {row} as Stop" if is_stop else f"Clear Stop on surface {row}")
        if is_stop:
            for i, s in enumerate(stack.surfaces):
                if i != row and (s.surface_type == SurfaceType.STOP or getattr(s, 'is_stop', False) or getattr(s, 'is_aperture_stop', False)):
                    if s.surface_type == SurfaceType.STOP:
                        s.surface_type = SurfaceType.STANDARD
                    s.is_stop = False
                    s.is_aperture_stop = False
                    if "(STO)" in s.name:
                        s.name = s.name.replace(" (STO)", "").replace("(STO)", "").strip()

            target_surf = stack.surfaces[row]
            target_surf.is_stop = True
            target_surf.is_aperture_stop = True
            if "(STO)" not in target_surf.name:
                if target_surf.name.startswith("S") and len(target_surf.name) <= 3:
                    target_surf.name = f"S{row} (STO)"
                else:
                    target_surf.name = f"{target_surf.name} (STO)"
        else:
            target_surf = stack.surfaces[row]
            if target_surf.surface_type == SurfaceType.STOP:
                target_surf.surface_type = SurfaceType.STANDARD
            target_surf.is_stop = False
            target_surf.is_aperture_stop = False
            if "(STO)" in target_surf.name:
                target_surf.name = target_surf.name.replace(" (STO)", "").replace("(STO)", "").strip()

        self.set_surface_stack(stack)
        if QT_AVAILABLE and hasattr(self, 'layoutChanged'):
            try:
                self.layoutChanged.emit()
            except Exception:
                pass
        return True

    def insert_aperture_stop_after(self, row: int, semi_diameter: float = 10.0, thickness: float = 10.0) -> int:
        """
        Insert a dedicated flat Aperture Stop (STO) surface after row.
        """
        self._push_undo(f"Insert Aperture Stop after row {row}")
        stack = self.get_surface_stack()
        # Clear previous stop flags
        for s in stack.surfaces:
            if s.surface_type == SurfaceType.STOP or getattr(s, 'is_stop', False) or getattr(s, 'is_aperture_stop', False):
                if s.surface_type == SurfaceType.STOP:
                    s.surface_type = SurfaceType.STANDARD
                s.is_stop = False
                s.is_aperture_stop = False
                if "(STO)" in s.name:
                    s.name = s.name.replace(" (STO)", "").replace("(STO)", "").strip()

        insert_idx = min(len(stack.surfaces), row + 1)
        stop_surface = Surface(
            name=f"S{insert_idx} (STO)",
            surface_type=SurfaceType.STOP,
            radius=0.0,
            thickness=float(thickness),
            glass="AIR",
            semi_diameter=float(semi_diameter),
            aperture_type="Circular",
            is_active=True
        )
        stop_surface.is_stop = True
        stop_surface.is_aperture_stop = True
        stack.insert_surface(insert_idx, stop_surface)
        self.set_surface_stack(stack)
        self._reindex_surface_names()
        if QT_AVAILABLE and hasattr(self, 'layoutChanged'):
            try:
                self.layoutChanged.emit()
            except Exception:
                pass
        return insert_idx

    def open_aperture_dialog(self, row: int, parent=None) -> bool:
        """
        Open the interactive Aperture and Stop dialog for the surface at row.
        """
        if not QT_AVAILABLE:
            return False
        stack = self.get_surface_stack()
        if not stack or not (0 <= row < len(stack.surfaces)):
            return False
        from optics_app.gui.aperture_dialog import ApertureStopDialog
        surf = stack.surfaces[row]
        dlg = ApertureStopDialog(surface=surf, surface_idx=row, model=self, parent=parent)
        exec_fn = getattr(dlg, 'exec_', dlg.exec)
        res = exec_fn()
        if res:
            self.set_surface_stack(stack)
        return bool(res)


class FrozenColumnTableView(QTableView if QT_AVAILABLE else object):
    """
    QTableView with frozen/pinned first column (Surface / Component Name)
    and vertical header (Enumeration: 0 (OBJ), 1, 2, ... N (IMA)) that stay
    permanently visible on the left during lateral/horizontal scrolling.
    """
    if QT_AVAILABLE:
        clicked = pyqtSignal(QModelIndex)
        doubleClicked = pyqtSignal(QModelIndex)
        customContextMenuRequested = pyqtSignal(object)
    else:
        clicked = DummySignal()
        doubleClicked = DummySignal()
        customContextMenuRequested = DummySignal()

    def __init__(self, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)
            self.dark_mode: bool = False
            self.frozen_table_view = QTableView(self)
            self._init_frozen_view()
        else:
            self.dark_mode: bool = False
            self.frozen_table_view = None

    def set_dark_mode(self, enabled: bool) -> None:
        """Toggle dark mode styling for the pinned/frozen column table view."""
        self.dark_mode = bool(enabled)
        if self.frozen_table_view is not None:
            if self.dark_mode:
                self.frozen_table_view.setStyleSheet("""
                    QTableView {
                        border-right: 2px solid #334155;
                        background-color: #0b0f19;
                        selection-background-color: #1e3a8a;
                        selection-color: #ffffff;
                        gridline-color: #1e293b;
                        color: #f1f5f9;
                    }
                    QHeaderView::section {
                        background-color: #1e293b;
                        color: #cbd5e1;
                        border: 1px solid #334155;
                        padding: 4px;
                        font-weight: bold;
                    }
                """)
            else:
                self.frozen_table_view.setStyleSheet("""
                    QTableView {
                        border-right: 2px solid #555555;
                        background-color: #242424;
                        selection-background-color: #2b5b84;
                        gridline-color: #383838;
                    }
                """)

    def _init_frozen_view(self):
        if not QT_AVAILABLE:
            return
        self.frozen_table_view.setFocusPolicy(getattr(Qt, 'NoFocus', 0))
        if hasattr(self.frozen_table_view, 'verticalHeader') and self.frozen_table_view.verticalHeader():
            self.frozen_table_view.verticalHeader().hide()
        if hasattr(self.frozen_table_view, 'horizontalHeader') and self.frozen_table_view.horizontalHeader():
            self.frozen_table_view.horizontalHeader().setSectionResizeMode(getattr(QHeaderView, 'Fixed', 2))
        
        self.viewport().stackUnder(self.frozen_table_view)
        
        self.frozen_table_view.setStyleSheet("""
            QTableView {
                border-right: 2px solid #555555;
                background-color: #242424;
                selection-background-color: #2b5b84;
                gridline-color: #383838;
            }
        """)
        
        self.frozen_table_view.setHorizontalScrollBarPolicy(getattr(Qt, 'ScrollBarAlwaysOff', 1))
        self.frozen_table_view.setVerticalScrollBarPolicy(getattr(Qt, 'ScrollBarAlwaysOff', 1))
        self.frozen_table_view.show()
        
        self.setHorizontalScrollMode(getattr(QAbstractItemView, 'ScrollPerPixel', 1))
        self.setVerticalScrollMode(getattr(QAbstractItemView, 'ScrollPerPixel', 1))
        self.frozen_table_view.setVerticalScrollMode(getattr(QAbstractItemView, 'ScrollPerPixel', 1))
        
        # Connect vertical scrollbars
        self.verticalScrollBar().valueChanged.connect(self.frozen_table_view.verticalScrollBar().setValue)
        self.frozen_table_view.verticalScrollBar().valueChanged.connect(self.verticalScrollBar().setValue)
        
        # Connect header resizing
        self.horizontalHeader().sectionResized.connect(self._on_section_resized)
        self.verticalHeader().sectionResized.connect(self._on_row_resized)

        # Context menu forwarding
        self.frozen_table_view.setContextMenuPolicy(getattr(Qt, 'CustomContextMenu', 3))
        self.frozen_table_view.customContextMenuRequested.connect(self._on_frozen_context_menu)

        # Click forwarding
        self.frozen_table_view.clicked.connect(self._on_frozen_clicked)
        self.frozen_table_view.doubleClicked.connect(self._on_frozen_double_clicked)

    def _on_frozen_context_menu(self, pos):
        global_pos = self.frozen_table_view.mapToGlobal(pos)
        local_pos = self.mapFromGlobal(global_pos)
        self.customContextMenuRequested.emit(local_pos)

    def _on_frozen_clicked(self, index):
        self.clicked.emit(index)

    def _on_frozen_double_clicked(self, index):
        self.doubleClicked.emit(index)

    def setModel(self, model):
        if QT_AVAILABLE:
            super().setModel(model)
            if self.frozen_table_view:
                self.frozen_table_view.setModel(model)
                self.frozen_table_view.setSelectionModel(self.selectionModel())
                
                # Show only Column 0 (Surface Name), hide all other columns
                col_count = model.columnCount() if model else 16
                for col in range(col_count):
                    if col == 0:
                        self.frozen_table_view.setColumnHidden(0, False)
                    else:
                        self.frozen_table_view.setColumnHidden(col, True)
                
                self.frozen_table_view.setColumnWidth(0, self.columnWidth(0))
                self.update_frozen_table_geometry()

                if hasattr(model, 'modelReset'):
                    try:
                        model.modelReset.connect(self._sync_all_row_heights)
                    except Exception:
                        pass
                if hasattr(model, 'layoutChanged'):
                    try:
                        model.layoutChanged.connect(self._sync_all_row_heights)
                    except Exception:
                        pass
                if hasattr(model, 'rowsInserted'):
                    try:
                        model.rowsInserted.connect(self._sync_all_row_heights)
                    except Exception:
                        pass
                if hasattr(model, 'rowsRemoved'):
                    try:
                        model.rowsRemoved.connect(self._sync_all_row_heights)
                    except Exception:
                        pass
                if hasattr(model, 'system_updated'):
                    try:
                        model.system_updated.connect(self._sync_all_row_heights)
                    except Exception:
                        pass
                for signal_name in ('modelReset', 'layoutChanged', 'rowsInserted', 'rowsRemoved', 'system_updated'):
                    signal = getattr(model, signal_name, None)
                    if signal is not None and hasattr(signal, 'connect'):
                        try:
                            signal.connect(self.apply_component_name_spans)
                        except Exception:
                            pass
                self.apply_component_name_spans()

    def set_component_name_mode(self, merged: bool) -> None:
        model = self.model() if QT_AVAILABLE else None
        if model is not None and hasattr(model, "set_component_name_mode"):
            model.set_component_name_mode(merged)
        self.apply_component_name_spans()

    def apply_component_name_spans(self, *args) -> None:
        """Apply vertical spans to both the scrolling and frozen first column."""
        if not QT_AVAILABLE:
            return
        for view in (self, self.frozen_table_view):
            if view is not None and hasattr(view, "clearSpans"):
                view.clearSpans()
        model = self.model()
        if model is None or not getattr(model, "component_name_mode", lambda: False)():
            return
        for span in model.component_name_spans():
            if span["count"] <= 1:
                continue
            for view in (self, self.frozen_table_view):
                if view is not None and hasattr(view, "setSpan"):
                    view.setSpan(span["start"], 0, span["count"], 1)

    def setItemDelegateForColumn(self, column, delegate):
        if QT_AVAILABLE:
            super().setItemDelegateForColumn(column, delegate)
            if column == 0 and self.frozen_table_view:
                self.frozen_table_view.setItemDelegateForColumn(0, delegate)

    def _on_section_resized(self, logical_index, old_size, new_size):
        if logical_index == 0 and self.frozen_table_view:
            self.frozen_table_view.setColumnWidth(0, new_size)
            self.update_frozen_table_geometry()

    def _on_row_resized(self, logical_index, old_size, new_size):
        if self.frozen_table_view:
            self.frozen_table_view.setRowHeight(logical_index, new_size)

    def _sync_all_row_heights(self):
        if not QT_AVAILABLE or not self.frozen_table_view:
            return
        model = self.model()
        if model:
            for r in range(model.rowCount()):
                self.frozen_table_view.setRowHeight(r, self.rowHeight(r))
        self.update_frozen_table_geometry()

    def update_frozen_table_geometry(self):
        if not QT_AVAILABLE or not self.frozen_table_view:
            return
        v_hdr_w = self.verticalHeader().width() if self.verticalHeader().isVisible() else 0
        f_w = self.frameWidth()
        c0_w = self.columnWidth(0)
        h_hdr_h = self.horizontalHeader().height() if self.horizontalHeader().isVisible() else 0
        vp_h = self.viewport().height()
        
        self.frozen_table_view.setGeometry(
            v_hdr_w + f_w,
            f_w,
            c0_w,
            vp_h + h_hdr_h
        )

    def resizeEvent(self, event):
        if QT_AVAILABLE:
            super().resizeEvent(event)
            self.update_frozen_table_geometry()

    def moveCursor(self, cursor_action, modifiers):
        if not QT_AVAILABLE:
            return None
        current = super().moveCursor(cursor_action, modifiers)
        if (cursor_action == getattr(QAbstractItemView, 'MoveLeft', 1) and
                current is not None and hasattr(current, 'column') and current.column() > 0 and
                self.visualRect(current).topLeft().x() < self.columnWidth(0)):
            new_val = self.horizontalScrollBar().value() + self.visualRect(current).topLeft().x() - self.columnWidth(0)
            self.horizontalScrollBar().setValue(new_val)
        return current

    def scrollTo(self, index, hint=getattr(QAbstractItemView, 'EnsureVisible', 0)):
        if not QT_AVAILABLE:
            return
        if index is not None and hasattr(index, 'column') and index.column() > 0:
            super().scrollTo(index, hint)

    def keyPressEvent(self, event):
        if not QT_AVAILABLE:
            return
        key = event.key()
        modifiers = event.modifiers()
        no_ctrl_alt = not (modifiers & (getattr(Qt, 'ControlModifier', 0x04000000) | getattr(Qt, 'AltModifier', 0x08000000)))
        if key == getattr(Qt, 'Key_V', 86) and no_ctrl_alt:
            win = self.window()
            if hasattr(win, '_set_selected_cells_solve'):
                win._set_selected_cells_solve(is_variable=True)
                event.accept()
                return
        elif key == getattr(Qt, 'Key_F', 70) and no_ctrl_alt:
            win = self.window()
            if hasattr(win, '_set_selected_cells_solve'):
                win._set_selected_cells_solve(is_variable=False)
                event.accept()
                return
        super().keyPressEvent(event)


class LensDataEditorWidget(QWidget if QT_AVAILABLE else object):
    """
    Widget container wrapping LensDataTableModel and FrozenColumnTableView.
    Provides right-click context menu for Zemax solve toggles (Variable V / Fixed F),
    mirror fold presets, aiming actions, inserting new surfaces/lenses, and replacing lenses.
    """
    if QT_AVAILABLE:
        system_updated = pyqtSignal()
    else:
        system_updated = None

    def __init__(self, model: Optional[LensDataTableModel] = None, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)
            self.setLayout(QVBoxLayout(self))
        else:
            self.system_updated = DummySignal()

        self.model = model or LensDataTableModel()
        self.table_view = FrozenColumnTableView(self) if QT_AVAILABLE else None
        self.btn_merge_component_names = None
        self.btn_split_component_names = None
        if self.table_view:
            self.table_view.setModel(self.model)
            try:
                from optics_app.gui.qt_compat import QAbstractItemView
                self.table_view.setSelectionBehavior(getattr(QAbstractItemView, 'SelectRows', 1))
                self.table_view.setSelectionMode(getattr(QAbstractItemView, 'ExtendedSelection', 3))
                self.table_view.setDragEnabled(True)
                self.table_view.setAcceptDrops(True)
                if hasattr(self.table_view, 'viewport') and self.table_view.viewport():
                    self.table_view.viewport().setAcceptDrops(True)
                self.table_view.setDragDropMode(getattr(QAbstractItemView, 'InternalMove', 4))
                self.table_view.setDropIndicatorShown(True)
                self.table_view.setDragDropOverwriteMode(False)
                self.table_view.setDefaultDropAction(getattr(Qt, 'MoveAction', 2))
            except Exception:
                pass

            if hasattr(self.table_view, 'setContextMenuPolicy'):
                try:
                    from optics_app.gui.qt_compat import Qt, QMenu, QAction
                    self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
                    self.table_view.customContextMenuRequested.connect(self._show_context_menu)
                except Exception:
                    pass
            if hasattr(self.table_view, 'doubleClicked'):
                try:
                    self.table_view.doubleClicked.connect(self._on_table_double_clicked)
                except Exception:
                    pass
            if hasattr(self.table_view, "setItemDelegateForColumn"):
                try:
                    from optics_app.gui.delegates import (
                        SurfaceTypeDelegate, GlassSelectionDelegate, NumericValueDelegate,
                        CADFilePathDelegate, ApertureTypeDelegate, ApertureParamDelegate
                    )
                    self.table_view.setItemDelegateForColumn(1, SurfaceTypeDelegate(self.table_view))
                    self.table_view.setItemDelegateForColumn(4, GlassSelectionDelegate(self.table_view))
                    for col in [2, 3, 6, 7, 8, 9, 10, 11, 12]:
                        self.table_view.setItemDelegateForColumn(col, NumericValueDelegate(self.table_view))
                    self.table_view.setItemDelegateForColumn(5, ApertureParamDelegate(self.table_view))
                    self.table_view.setItemDelegateForColumn(13, CADFilePathDelegate(self.table_view))
                    self.table_view.setItemDelegateForColumn(14, ApertureTypeDelegate(self.table_view))
                    self.table_view.setItemDelegateForColumn(15, ApertureParamDelegate(self.table_view))
                except Exception:
                    pass

            self.layout().addWidget(self.table_view)

        if hasattr(self.model, "system_updated") and hasattr(self.model.system_updated, "connect"):
            try:
                self.model.system_updated.connect(self._on_model_updated)
            except Exception:
                pass

    def _on_table_double_clicked(self, index):
        if not index.isValid():
            return
        col = index.column()
        row = index.row()
        if col in (14, 15):
            self._open_aperture_dialog(row)

    def _open_aperture_dialog(self, row: int):
        """Open the interactive Aperture and Stop dialog for the selected row."""
        if self.model and hasattr(self.model, "open_aperture_dialog"):
            self.model.open_aperture_dialog(row, parent=self)

    def _set_component_name_mode(self, merged: bool) -> None:
        if self.table_view is not None:
            self.table_view.set_component_name_mode(merged)
        self._update_component_mode_buttons()

    def _update_component_mode_buttons(self) -> None:
        if not QT_AVAILABLE or getattr(self, "btn_merge_component_names", None) is None:
            return
        merged = self.model.component_name_mode()
        self.btn_merge_component_names.setEnabled(not merged)
        self.btn_split_component_names.setEnabled(merged)

    def _show_context_menu(self, pos):
        """Right-click context menu for Zemax variables, mirror presets, lens insertion, catalog replacement, and apertures/stops."""
        try:
            from optics_app.gui.qt_compat import QMenu, QAction
            index = self.table_view.indexAt(pos)
            if not index.isValid():
                return
            
            row = index.row()
            col = index.column()
            menu = QMenu(self)
            
            # 1. Variable / Fixed Solve Toggles
            if col in (2, 3): # Radius or Thickness
                param_name = "Thickness" if col == 3 else "Radius"
                act_var = QAction(f"Set {param_name} as Variable (V)", self)
                act_var.triggered.connect(lambda: self._set_cell_solve(row, col, True))
                menu.addAction(act_var)
                
                act_fix = QAction(f"Set {param_name} as Fixed (F)", self)
                act_fix.triggered.connect(lambda: self._set_cell_solve(row, col, False))
                menu.addAction(act_fix)
                menu.addSeparator()

            # 1b. Aperture & Stop Controls Submenu
            menu_aper = menu.addMenu(f"â­• Aperture & Stop (Surface {row})")
            act_aper_dlg = QAction("âš™ï¸ Define Aperture & Stop Parameters...", self)
            act_aper_dlg.setStatusTip("Configure clear aperture, stops, slits, obscurations, and spider vanes")
            act_aper_dlg.triggered.connect(lambda checked, r=row: self._open_aperture_dialog(r))
            menu_aper.addAction(act_aper_dlg)

            is_curr_stop = False
            if 0 <= row < len(self.model._surfaces):
                s = self.model._surfaces[row]
                is_curr_stop = (
                    (getattr(s, 'surface_type', None) == SurfaceType.STOP)
                    or getattr(s, 'is_stop', False)
                    or getattr(s, 'is_aperture_stop', False)
                    or "STOP" in str(getattr(s, 'surface_type', '')).upper()
                )

            if is_curr_stop:
                act_toggle_stop = QAction("âšª Clear Stop Status (Set to Standard)", self)
                act_toggle_stop.triggered.connect(lambda checked, r=row: self.model.set_stop_surface(r, False))
            else:
                act_toggle_stop = QAction("ðŸ”´ Set as System Aperture Stop (STO)", self)
                act_toggle_stop.triggered.connect(lambda checked, r=row: self.model.set_stop_surface(r, True))
            menu_aper.addAction(act_toggle_stop)

            act_ins_stop = QAction(f"âž• Insert Aperture Stop Surface After Surface {row}...", self)
            act_ins_stop.triggered.connect(lambda checked, r=row: self.model.insert_aperture_stop_after(r))
            menu_aper.addAction(act_ins_stop)
            menu.addSeparator()

            # 2. Mirror Quick-Fold & Aiming Presets Submenu
            menu_mirror = menu.addMenu(f"Mirror Actions (Surface {row})")
            act_fold_90_yz = QAction("Set 90Â° Fold Angle (YZ Plane)", self)
            act_fold_90_yz.triggered.connect(lambda checked: self.model.set_mirror_fold_angle(row, 90.0, "YZ"))
            menu_mirror.addAction(act_fold_90_yz)

            act_fold_90_xz = QAction("Set 90Â° Fold Angle (XZ Plane)", self)
            act_fold_90_xz.triggered.connect(lambda checked: self.model.set_mirror_fold_angle(row, 90.0, "XZ"))
            menu_mirror.addAction(act_fold_90_xz)

            act_fold_45 = QAction("Set 45Â° Fold Angle", self)
            act_fold_45.triggered.connect(lambda checked: self.model.set_mirror_fold_angle(row, 45.0, "YZ"))
            menu_mirror.addAction(act_fold_45)

            act_fold_retro = QAction("Retro-Reflector (180Â° Fold)", self)
            act_fold_retro.triggered.connect(lambda checked: self.model.set_mirror_fold_angle(row, 180.0, "YZ"))
            menu_mirror.addAction(act_fold_retro)

            act_aim_det = QAction("Aim Mirror at Detector / Image Plane", self)
            act_aim_det.triggered.connect(lambda checked: self.model.aim_mirror_to_target(row, -1))
            menu_mirror.addAction(act_aim_det)

            # 3. Lens Element Management (Locking, Unlocking Edition, Deletion)
            from optics_app.engine.element_colors import get_surface_element_group
            grp_info = get_surface_element_group(self.model._surfaces, row)
            g_type = grp_info.get("group_type", "")
            g_name = grp_info.get("group_name", f"Surface {row}")

            if g_type == "LENS":
                menu_lens = menu.addMenu(f"Lens Element: {g_name}")
                is_locked = self.model.is_lens_locked(row)
                if is_locked:
                    act_unlock = QAction(f"ðŸ”“ Enable Lens Edition (Unlock '{g_name}' Parameters)", self)
                    act_unlock.triggered.connect(lambda checked, r=row: self.model.set_lens_locked(r, False))
                    menu_lens.addAction(act_unlock)
                else:
                    act_lock = QAction(f"ðŸ”’ Lock Lens Parameters (Fix '{g_name}' Prescription)", self)
                    act_lock.triggered.connect(lambda checked, r=row: self.model.set_lens_locked(r, True))
                    menu_lens.addAction(act_lock)

                act_del_lens = QAction(f"ðŸ—‘ Delete Lens Element '{g_name}'", self)
                act_del_lens.triggered.connect(lambda checked, r=row: self.model.delete_lens_element(r))
                menu_lens.addAction(act_del_lens)

                menu_rep_lens = menu_lens.addMenu(f"ðŸ”„ Replace Lens Element '{g_name}'")
                act_rep_cat_tool = QAction("ðŸ” Thorlabs Catalog Tool (Choose Parameters)...", self)
                act_rep_cat_tool.triggered.connect(lambda checked, g=g_name: self._open_catalog_lens_dialog(target=g, mode="replace"))
                menu_rep_lens.addAction(act_rep_cat_tool)
                menu_rep_lens.addSeparator()

                act_rep_zmx_elem = QAction("ðŸ“ From Zemax File (.zmx)...", self)
                act_rep_zmx_elem.triggered.connect(lambda checked, g=g_name: self._replace_lens_zmx(g))
                menu_rep_lens.addAction(act_rep_zmx_elem)

                act_rep_cust_elem = QAction("ðŸ“„ From Custom File (.json)...", self)
                act_rep_cust_elem.triggered.connect(lambda checked, g=g_name: self._replace_lens_custom(g))
                menu_rep_lens.addAction(act_rep_cust_elem)

                menu_rep_cat_elem = menu_rep_lens.addMenu("ðŸ” From Thorlabs Catalog (Quick Replace)")
                from optics_app.gui.catalog_menu import populate_lens_catalog_menu
                lens_catalog = self.model.get_surface_stack()._get_lens_catalog()
                populate_lens_catalog_menu(
                    menu_rep_cat_elem,
                    lambda part, target_row=row: self.model.replace_lens_from_catalog(target_row, part),
                    lens_catalog,
                )

                act_rot_grp = QAction(f"Rotate Lens Element '{g_name}' as Group...", self)
                act_rot_grp.triggered.connect(lambda checked, r=row: self.model.rotate_lens_element(r, tilt_x=5.0))
                menu_lens.addAction(act_rot_grp)
            else:
                act_del_surf = QAction(f"ðŸ—‘ Delete Surface {row}", self)
                act_del_surf.triggered.connect(lambda checked, r=row: self.model.removeRows(r, 1))
                menu.addAction(act_del_surf)

            menu.addSeparator()

            # 4. Insert After Selected Surface
            act_ins_surf = QAction(f"Insert Surface After Surface {row}", self)
            act_ins_surf.triggered.connect(lambda: self.model.insert_surface_after(row))
            menu.addAction(act_ins_surf)

            act_ins_singlet = QAction(f"Insert Singlet Lens After Surface {row}", self)
            act_ins_singlet.triggered.connect(lambda: self.model.insert_singlet_after(row))
            menu.addAction(act_ins_singlet)

            act_ins_doublet = QAction(f"Insert Doublet Lens After Surface {row}", self)
            act_ins_doublet.triggered.connect(lambda: self.model.insert_doublet_after(row))
            menu.addAction(act_ins_doublet)

            act_ins_cat_dialog = QAction(f"ðŸ” Insert Thorlabs Catalog Lens (Choose Parameters)...", self)
            act_ins_cat_dialog.triggered.connect(lambda checked, r=row: self._open_catalog_lens_dialog(target=r, mode="insert"))
            menu.addAction(act_ins_cat_dialog)

            # Catalog Insert Submenu
            menu_cat_ins = menu.addMenu(f"Insert Thorlabs Catalog Lens After Surface {row}")
            from optics_app.gui.catalog_menu import populate_lens_catalog_menu
            lens_catalog = self.model.get_surface_stack()._get_lens_catalog()
            populate_lens_catalog_menu(
                menu_cat_ins,
                lambda part: self.model.insert_catalog_lens_after(row, part),
                lens_catalog,
            )

            act_rep_cat_dialog = QAction(f"ðŸ” Replace Lens at Surface {row} With Catalog Lens (Choose Parameters)...", self)
            act_rep_cat_dialog.triggered.connect(lambda checked, r=row: self._open_catalog_lens_dialog(target=r, mode="replace"))
            menu.addAction(act_rep_cat_dialog)

            # Catalog Replace Submenu
            menu_cat_rep = menu.addMenu(f"Replace Lens at Surface {row} With Catalog Lens")
            populate_lens_catalog_menu(
                menu_cat_rep,
                lambda part: self.model.replace_with_catalog_lens(row, part),
                lens_catalog,
            )

            act_rep_zmx = QAction(f"Replace Lens at Surface {row} With Zemax File (.zmx)...", self)
            act_rep_zmx.triggered.connect(lambda checked, r=row: self._replace_with_zmx(r))
            menu.addAction(act_rep_zmx)

            menu.addSeparator()

            # 5. Zemax Defaults & Bulk Solves
            act_zemax_def = QAction("Zemax Defaults: Set Air Gaps Variable, Lenses Fixed", self)
            act_zemax_def.triggered.connect(self.model.set_zemax_defaults)
            menu.addAction(act_zemax_def)

            act_all_fix = QAction("Set All Parameters Fixed", self)
            act_all_fix.triggered.connect(self.model.set_all_fixed)
            menu.addAction(act_all_fix)

            execute_fn = getattr(menu, 'exec_', menu.exec)
            execute_fn(self.table_view.viewport().mapToGlobal(pos))
        except Exception:
            pass

    def _replace_with_zmx(self, row: int):
        try:
            from optics_app.gui.qt_compat import QFileDialog, QMessageBox, QDialog
            from optics_app.io.zmx_parser import parse_zmx_file
            from optics_app.gui.zemax_import_dialog import ZemaxImportDialog, process_imported_surfaces
            import os
            
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Replace with Zemax Element", "", "Zemax Files (*.zmx);;All Files (*)"
            )
            if not file_path or not os.path.exists(file_path):
                return
                
            stack = parse_zmx_file(file_path)
            
            if QT_AVAILABLE and QDialog is not None:
                dialog = ZemaxImportDialog(self, max_surfaces=len(self.model._surfaces))
                dialog.index_spin.setValue(row)
                dialog.index_spin.setEnabled(False)
                dialog.setWindowTitle("Replace Zemax Element")
                
                exec_fn = getattr(dialog, 'exec_', dialog.exec)
                accepted_val = getattr(QDialog, 'Accepted', 1)
                if exec_fn() != accepted_val:
                    return
                settings = dialog.get_settings()
            else:
                settings = {"insert_index": row, "reverse": False, "tilt_x": 0.0, "tilt_y": 0.0, "tilt_z": 0.0}
                
            processed = process_imported_surfaces(stack.surfaces, settings)
            
            # The model resolves any clicked constituent surface to the full
            # physical lens and performs one atomic splice while preserving its
            # trailing gap and undo snapshot.
            self.model.replace_lens_element(row, processed)
            if self.system_updated:
                self.system_updated.emit()
                
        except Exception as e:
            from optics_app.gui.qt_compat import QMessageBox
            if hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.critical(self, "Error", str(e))

    def _set_cell_solve(self, row: int, col: int, is_variable: bool):
        self.model._push_undo(f"Set solve on surface {row}")
        if row < len(self.model._surfaces):
            surf = self.model._surfaces[row]
            if col == 3:
                surf.thickness_variable = is_variable
            elif col == 2:
                surf.radius_variable = is_variable
            self.model._sync_stack_from_data()
            if hasattr(self.model, 'system_updated') and self.model.system_updated is not None:
                self.model.system_updated.emit()

    def _replace_lens_zmx(self, target: Any):
        try:
            from optics_app.gui.qt_compat import QFileDialog, QMessageBox
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"Replace Lens '{target}' with Zemax File", "", "Zemax Files (*.zmx);;All Files (*)"
            )
            if not file_path:
                return
            self.model.replace_lens_from_zmx(target, file_path)
        except Exception as e:
            if hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.critical(self, "Replace Lens Error", str(e))

    def _replace_lens_custom(self, target: Any):
        try:
            from optics_app.gui.qt_compat import QFileDialog, QMessageBox
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"Replace Lens '{target}' with Custom File", "", "Custom Lens / JSON Files (*.json *.txt);;All Files (*)"
            )
            if not file_path:
                return
            self.model.replace_lens_from_custom_file(target, file_path)
        except Exception as e:
            if hasattr(self, 'isVisible') and self.isVisible():
                QMessageBox.critical(self, "Replace Lens Error", str(e))

    def _open_catalog_lens_dialog(self, target: Any = None, mode: str = "replace"):
        """Open the interactive Thorlabs Catalog Lens Dialog."""
        if not QT_AVAILABLE:
            return
        from optics_app.gui.catalog_lens_dialog import CatalogLensDialog
        dialog = CatalogLensDialog(model=self.model, parent=self, target=target, mode=mode)
        exec_fn = getattr(dialog, 'exec_', dialog.exec)
        exec_fn()

    def _on_model_updated(self):
        if hasattr(self, "system_updated") and hasattr(self.system_updated, "emit"):
            try:
                self.system_updated.emit()
            except Exception:
                pass


