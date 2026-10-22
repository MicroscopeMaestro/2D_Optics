"""
Zemax OpticStudio-style Parameter Solve and Variable Definition Dialog.
=====================================================================
Allows setting parameters (Radius, Thickness, Conic, Decenters, Tilts) to:
- Fixed (F): Unchanging value during optimization.
- Variable (V): Active degree of freedom for optimizer with min/max bounds.
- Pickup (P): Pick up value from another surface with scale factor and offset.
- Marginal Ray Solve (S): Paraxial focus or normal incidence solve.
"""

from typing import Optional, Any, Dict, List
import math
import numpy as np

from optics_app.gui.qt_compat import (
    QT_AVAILABLE, QtWidgets, QtCore,
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QRadioButton, QGroupBox,
    QMessageBox, Qt
)


class ZemaxSolveDialog(QDialog if QT_AVAILABLE else object):
    """
    Zemax OpticStudio-style Solve Dialog for LDE parameters.
    """

    def __init__(self, surface_index: int, param_col: int, model: Any, parent: Optional[Any] = None):
        if QT_AVAILABLE:
            super().__init__(parent)
            self.setWindowTitle(f"Solve: Surface {surface_index} Parameter")
            self.resize(440, 420)

        self.surface_index = surface_index
        self.param_col = param_col
        self.model = model
        self.surf = self.model._surfaces[surface_index] if 0 <= surface_index < len(self.model._surfaces) else None

        # Determine parameter name from column
        col_to_param = {
            2: "Radius",
            3: "Thickness",
            6: "Conic",
            7: "Decenter X",
            8: "Decenter Y",
            9: "Tilt X",
            10: "Tilt Y",
            11: "Tilt Z"
        }
        self.param_name = col_to_param.get(param_col, "Radius")

        if QT_AVAILABLE and self.surf is not None:
            self._setup_ui()
            self._load_current_values()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Surface & Parameter Header Info
        header_group = QGroupBox("Target Parameter", self)
        h_layout = QGridLayout(header_group)

        h_layout.addWidget(QLabel("<b>Surface:</b>"), 0, 0)
        surf_name = getattr(self.surf, 'name', f'Surface {self.surface_index}')
        glass_name = getattr(self.surf, 'glass', 'AIR')
        h_layout.addWidget(QLabel(f"Surface {self.surface_index} ({surf_name}) — Glass: {glass_name}"), 0, 1)

        h_layout.addWidget(QLabel("<b>Parameter:</b>"), 1, 0)
        h_layout.addWidget(QLabel(f"<b>{self.param_name}</b>"), 1, 1)

        h_layout.addWidget(QLabel("Current Value:"), 2, 0)
        val = getattr(self.surf, self.param_name.lower().replace(" ", "_"), 0.0)
        self.lbl_current_val = QLabel(f"{float(val):.4f}")
        h_layout.addWidget(self.lbl_current_val, 2, 1)

        layout.addWidget(header_group)

        # 2. Solve Type Selection
        solve_group = QGroupBox("Solve Type", self)
        s_layout = QVBoxLayout(solve_group)

        self.radio_fixed = QRadioButton("🔒 Fixed (F) — Constant value", solve_group)
        self.radio_variable = QRadioButton("⚡ Variable (V) — Degree of freedom for optimization", solve_group)
        self.radio_pickup = QRadioButton("🔗 Pickup (P) — Pick up from another surface", solve_group)
        self.radio_ray_solve = QRadioButton("🎯 Marginal Ray Solve (S) — Paraxial solve", solve_group)

        s_layout.addWidget(self.radio_fixed)
        s_layout.addWidget(self.radio_variable)
        s_layout.addWidget(self.radio_pickup)
        s_layout.addWidget(self.radio_ray_solve)

        self.radio_fixed.toggled.connect(self._on_solve_type_changed)
        self.radio_variable.toggled.connect(self._on_solve_type_changed)
        self.radio_pickup.toggled.connect(self._on_solve_type_changed)
        self.radio_ray_solve.toggled.connect(self._on_solve_type_changed)

        layout.addWidget(solve_group)

        # 3. Parameters / Config Box for Active Solve
        self.config_group = QGroupBox("Solve Configuration", self)
        self.c_layout = QGridLayout(self.config_group)

        # Variable Bounds Controls
        self.lbl_min = QLabel("Min Limit:")
        self.spin_min = QDoubleSpinBox(self.config_group)
        self.spin_min.setRange(-1e6, 1e6)
        self.spin_min.setDecimals(4)

        self.lbl_max = QLabel("Max Limit:")
        self.spin_max = QDoubleSpinBox(self.config_group)
        self.spin_max.setRange(-1e6, 1e6)
        self.spin_max.setDecimals(4)

        # Pickup Controls
        self.lbl_from_surf = QLabel("From Surface:")
        self.combo_from_surf = QComboBox(self.config_group)
        for i, s in enumerate(self.model._surfaces):
            s_name = getattr(s, 'name', f'Surface {i}')
            self.combo_from_surf.addItem(f"Surface {i} ({s_name})", i)

        self.lbl_scale = QLabel("Scale Factor:")
        self.spin_scale = QDoubleSpinBox(self.config_group)
        self.spin_scale.setRange(-1000.0, 1000.0)
        self.spin_scale.setDecimals(4)
        self.spin_scale.setValue(1.0)
        self.spin_scale.valueChanged.connect(self._update_pickup_preview)

        self.lbl_offset = QLabel("Offset (mm):")
        self.spin_offset = QDoubleSpinBox(self.config_group)
        self.spin_offset.setRange(-1e6, 1e6)
        self.spin_offset.setDecimals(4)
        self.spin_offset.setValue(0.0)
        self.spin_offset.valueChanged.connect(self._update_pickup_preview)

        self.lbl_preview = QLabel("Formula: Value = (FromSurface * 1.0) + 0.0")
        self.lbl_preview.setStyleSheet("color: #0284c7; font-weight: bold; font-size: 11px;")

        self.combo_from_surf.currentIndexChanged.connect(self._update_pickup_preview)

        # Ray Solve Controls
        self.lbl_solve_desc = QLabel(
            "Automatically calculates paraxial marginal ray focus at this surface."
            if self.param_name == "Thickness" else
            "Sets radius so marginal rays hit at normal incidence (zero spherical aberration)."
        )
        self.lbl_solve_desc.setWordWrap(True)
        self.lbl_solve_desc.setStyleSheet("color: #475569; font-style: italic;")

        layout.addWidget(self.config_group)

        # 4. Action Buttons (Apply, OK, Cancel)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        self.btn_apply = QPushButton("Apply", self)
        self.btn_apply.clicked.connect(self.apply_solve)
        btn_layout.addWidget(self.btn_apply)

        self.btn_ok = QPushButton("OK", self)
        self.btn_ok.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 5px 15px; border-radius: 4px;")
        self.btn_ok.clicked.connect(lambda: (self.apply_solve(), self.accept()))
        btn_layout.addWidget(self.btn_ok)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

    def _load_current_values(self):
        """Load current solve state from surface."""
        # Determine variable status
        attr_var = f"{self.param_name.lower().replace(' ', '_')}_variable"
        is_var = bool(getattr(self.surf, attr_var, False))
        solve_type = getattr(self.surf, f"{self.param_name.lower()}_solve_type", "Variable" if is_var else "Fixed")

        if solve_type == "Pickup":
            self.radio_pickup.setChecked(True)
        elif solve_type in ("MarginalRayFocus", "MarginalRayNormal", "Solve"):
            self.radio_ray_solve.setChecked(True)
        elif is_var:
            self.radio_variable.setChecked(True)
        else:
            self.radio_fixed.setChecked(True)

        # Bounds
        attr_min = f"{self.param_name.lower().replace(' ', '_')}_min"
        attr_max = f"{self.param_name.lower().replace(' ', '_')}_max"
        self.spin_min.setValue(getattr(self.surf, attr_min, -1000.0 if "radius" in attr_min else 0.0))
        self.spin_max.setValue(getattr(self.surf, attr_max, 1000.0))

        # Pickup
        from_surf = getattr(self.surf, f"{self.param_name.lower()}_pickup_surface", max(0, self.surface_index - 1))
        idx = self.combo_from_surf.findData(from_surf)
        if idx >= 0:
            self.combo_from_surf.setCurrentIndex(idx)
        self.spin_scale.setValue(getattr(self.surf, f"{self.param_name.lower()}_pickup_scale", 1.0))
        self.spin_offset.setValue(getattr(self.surf, f"{self.param_name.lower()}_pickup_offset", 0.0))

        self._on_solve_type_changed()

    def _on_solve_type_changed(self):
        """Update visible controls based on selected solve type."""
        # Clear layout items
        while self.c_layout.count():
            item = self.c_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        if self.radio_fixed.isChecked():
            self.config_group.setTitle("Fixed Status")
            lbl = QLabel("This parameter is locked and will remain constant during optimization.")
            lbl.setStyleSheet("color: #64748b;")
            self.c_layout.addWidget(lbl, 0, 0, 1, 2)

        elif self.radio_variable.isChecked():
            self.config_group.setTitle("Optimization Variable Limits")
            self.c_layout.addWidget(self.lbl_min, 0, 0)
            self.c_layout.addWidget(self.spin_min, 0, 1)
            self.c_layout.addWidget(self.lbl_max, 1, 0)
            self.c_layout.addWidget(self.spin_max, 1, 1)

        elif self.radio_pickup.isChecked():
            self.config_group.setTitle("Pickup Configuration")
            self.c_layout.addWidget(self.lbl_from_surf, 0, 0)
            self.c_layout.addWidget(self.combo_from_surf, 0, 1)
            self.c_layout.addWidget(self.lbl_scale, 1, 0)
            self.c_layout.addWidget(self.spin_scale, 1, 1)
            self.c_layout.addWidget(self.lbl_offset, 2, 0)
            self.c_layout.addWidget(self.spin_offset, 2, 1)
            self.c_layout.addWidget(self.lbl_preview, 3, 0, 1, 2)
            self._update_pickup_preview()

        elif self.radio_ray_solve.isChecked():
            self.config_group.setTitle("Paraxial Ray Solve")
            self.c_layout.addWidget(self.lbl_solve_desc, 0, 0, 1, 2)

    def _update_pickup_preview(self):
        from_idx = self.combo_from_surf.currentData()
        scale = self.spin_scale.value()
        offset = self.spin_offset.value()
        if from_idx is not None and 0 <= from_idx < len(self.model._surfaces):
            from_surf = self.model._surfaces[from_idx]
            from_val = getattr(from_surf, self.param_name.lower().replace(" ", "_"), 0.0)
            calc_val = (from_val * scale) + offset
            self.lbl_preview.setText(f"Value = (S{from_idx} [{from_val:.3f}] × {scale:.2f}) + {offset:.2f} = <b>{calc_val:.4f} mm</b>")

    def apply_solve(self):
        """Apply the chosen solve to the active surface and refresh model."""
        if self.surf is None:
            return

        param_attr = self.param_name.lower().replace(" ", "_")
        var_attr = f"{param_attr}_variable"
        prefix = self.param_name.lower()

        if hasattr(self.model, '_push_undo'):
            self.model._push_undo(f"Set {self.param_name} solve on Surface {self.surface_index}")

        if self.radio_fixed.isChecked():
            setattr(self.surf, var_attr, False)
            setattr(self.surf, f"{prefix}_solve_type", "Fixed")

        elif self.radio_variable.isChecked():
            setattr(self.surf, var_attr, True)
            setattr(self.surf, f"{prefix}_solve_type", "Variable")
            setattr(self.surf, f"{var_attr.replace('_variable', '_min')}", self.spin_min.value())
            setattr(self.surf, f"{var_attr.replace('_variable', '_max')}", self.spin_max.value())

        elif self.radio_pickup.isChecked():
            setattr(self.surf, var_attr, False)
            setattr(self.surf, f"{prefix}_solve_type", "Pickup")
            from_idx = self.combo_from_surf.currentData()
            scale = self.spin_scale.value()
            offset = self.spin_offset.value()
            setattr(self.surf, f"{prefix}_pickup_surface", from_idx)
            setattr(self.surf, f"{prefix}_pickup_scale", scale)
            setattr(self.surf, f"{prefix}_pickup_offset", offset)

            if from_idx is not None and 0 <= from_idx < len(self.model._surfaces):
                from_surf = self.model._surfaces[from_idx]
                from_val = getattr(from_surf, param_attr, 0.0)
                calc_val = (from_val * scale) + offset
                setattr(self.surf, param_attr, calc_val)
                # update table data
                if hasattr(self.model, '_data') and self.surface_index < len(self.model._data):
                    self.model._data[self.surface_index][self.param_col] = calc_val

        elif self.radio_ray_solve.isChecked():
            setattr(self.surf, var_attr, False)
            solve_name = "MarginalRayFocus" if self.param_name == "Thickness" else "MarginalRayNormal"
            setattr(self.surf, f"{prefix}_solve_type", solve_name)

            # Compute paraxial solve if possible
            if self.param_name == "Thickness":
                try:
                    stack = self.model.get_surface_stack()
                    from optics_app.engine.tracer import OpticalSystem
                    from optics_app.engine.ray import Ray
                    sys_opt = OpticalSystem(stack)
                    r = Ray(pos=np.array([0.0, 0.01, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)
                    tr = sys_opt.trace_ray(r)
                    if len(tr.history) > self.surface_index:
                        h = tr.history[self.surface_index]
                        dy = h.get("dir_y", 0.0)
                        dz = h.get("dir_z", 1.0)
                        y_pos = h.get("y", 0.0)
                        if abs(dy) > 1e-12:
                            solved_t = abs(y_pos / (dy / dz))
                            setattr(self.surf, "thickness", float(solved_t))
                            if self.surface_index < len(self.model._data):
                                self.model._data[self.surface_index][3] = float(solved_t)
                except Exception:
                    pass

        # Notify model and UI
        if hasattr(self.model, '_sync_stack_from_data'):
            self.model._sync_stack_from_data()
        if hasattr(self.model, 'layoutChanged'):
            self.model.layoutChanged.emit()
        if hasattr(self.model, 'system_updated') and self.model.system_updated is not None:
            self.model.system_updated.emit()
