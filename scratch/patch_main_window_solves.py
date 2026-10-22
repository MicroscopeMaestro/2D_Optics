"""
Patch script for adding Zemax solve controls and context menus to main_window.py
"""

def update_main_window(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Update button bar above LDE table
    old_btn_bar = """        self.btn_toggle_comp_view = QPushButton("🔲 Component View", lde_container)
        self.btn_toggle_comp_view.setCheckable(True)
        self.btn_toggle_comp_view.setToolTip("Toggle merged component view in the Lens Data Editor")
        self.btn_toggle_comp_view.toggled.connect(self._set_lde_component_name_mode)
        btn_bar.addWidget(self.btn_toggle_comp_view)

        btn_bar.addStretch(1)"""

    new_btn_bar = """        self.btn_toggle_comp_view = QPushButton("🔲 Component View", lde_container)
        self.btn_toggle_comp_view.setCheckable(True)
        self.btn_toggle_comp_view.setToolTip("Toggle merged component view in the Lens Data Editor")
        self.btn_toggle_comp_view.toggled.connect(self._set_lde_component_name_mode)
        btn_bar.addWidget(self.btn_toggle_comp_view)

        # Zemax Solve & Variable Quick Buttons
        btn_var = QPushButton("⚡ Variable (V)", lde_container)
        btn_var.setToolTip("Mark selected parameter as optimization variable (V) [Zemax style]")
        btn_var.clicked.connect(lambda: self._set_selected_cells_solve(is_variable=True))
        btn_bar.addWidget(btn_var)

        btn_fix = QPushButton("🔒 Fixed (F)", lde_container)
        btn_fix.setToolTip("Mark selected parameter as fixed (F)")
        btn_fix.clicked.connect(lambda: self._set_selected_cells_solve(is_variable=False))
        btn_bar.addWidget(btn_fix)

        btn_solve = QPushButton("⚙️ Solve...", lde_container)
        btn_solve.setToolTip("Open Zemax Solve & Pickup Dialog for selected parameter")
        btn_solve.clicked.connect(lambda: self.open_solve_dialog())
        btn_bar.addWidget(btn_solve)

        btn_bar.addStretch(1)"""

    if old_btn_bar in text:
        text = text.replace(old_btn_bar, new_btn_bar)
        print(f"Updated LDE button bar in {file_path}")
    else:
        print(f"old_btn_bar not found in {file_path}")

    # 2. Update show_lde_context_menu to detect col_idx and add Zemax solve submenu
    old_ctx_start = """        # Determine row index at position if possible
        row_idx = 0
        if hasattr(self.table_view, "indexAt"):
            idx = self.table_view.indexAt(pos)
            if idx and hasattr(idx, "row") and idx.row() >= 0:
                row_idx = idx.row()
        elif hasattr(self.table_view, "currentIndex"):
            c_idx = self.table_view.currentIndex()
            if c_idx and hasattr(c_idx, "row") and c_idx.row() >= 0:
                row_idx = c_idx.row()"""

    new_ctx_start = """        # Determine cell index at position if possible
        row_idx = 0
        col_idx = 0
        if hasattr(self.table_view, "indexAt"):
            idx = self.table_view.indexAt(pos)
            if idx and hasattr(idx, "row") and idx.row() >= 0:
                row_idx = idx.row()
                col_idx = idx.column()
        elif hasattr(self.table_view, "currentIndex"):
            c_idx = self.table_view.currentIndex()
            if c_idx and hasattr(c_idx, "row") and c_idx.row() >= 0:
                row_idx = c_idx.row()
                col_idx = c_idx.column()"""

    if old_ctx_start in text:
        text = text.replace(old_ctx_start, new_ctx_start)
        print(f"Updated context menu cell detection in {file_path}")
    else:
        print(f"old_ctx_start not found in {file_path}")

    # 3. Insert Zemax solve actions into show_lde_context_menu
    old_actions = """        act_insert = QAction("Insert Surface Row", self)
        act_insert.triggered.connect(lambda: self.on_insert_surface())

        act_cluster = QAction("🔗 Cluster Selected Surfaces as Lens...", self)
        act_cluster.setToolTip("Group selected surfaces into a single multi-surface lens element")
        act_cluster.triggered.connect(self._on_menu_group_selected_surfaces)

        act_uncluster = QAction("🔓 Uncluster / Clear Group", self)
        act_uncluster.triggered.connect(lambda: self._on_menu_clear_group(row_idx))

        act_achromat = QAction("🎨 Design Achromatic Doublet Lens...", self)
        act_achromat.triggered.connect(lambda: self.on_open_achromat_designer_dialog())
        
        component_span = self.model.component_span_for_row(row_idx) if self.model.component_name_mode() else None
        delete_label = f"Delete Complete Lens '{component_span['name']}'" if component_span else "Delete Surface Row"
        act_delete = QAction(delete_label, self)
        act_delete.triggered.connect(lambda: self.on_delete_surface())
        
        menu.addAction(act_cluster)
        menu.addAction(act_uncluster)
        menu.addSeparator()
        menu.addAction(act_insert)
        menu.addAction(act_achromat)
        menu.addAction(act_delete)
        menu.addSeparator()"""

    new_actions = """        # Zemax Parameter Solve Actions for the clicked cell
        param_cols = {
            2: "Radius",
            3: "Thickness",
            6: "Conic",
            7: "Decenter X",
            8: "Decenter Y",
            9: "Tilt X",
            10: "Tilt Y",
            11: "Tilt Z"
        }
        if col_idx in param_cols and 0 <= row_idx < len(self.model._surfaces):
            p_name = param_cols[col_idx]
            surf = self.model._surfaces[row_idx]
            attr_var = f"{p_name.lower().replace(' ', '_')}_variable"
            is_var = bool(getattr(surf, attr_var, False))
            
            menu_solve = menu.addMenu(f"⚡ Zemax Solve ({p_name} — S{row_idx})")
            
            act_set_var = QAction(f"⚡ Set {p_name} as Variable (V)", self)
            act_set_var.setCheckable(True)
            act_set_var.setChecked(is_var)
            act_set_var.triggered.connect(lambda checked, r=row_idx, c=col_idx: self._set_cell_variable(r, c, True))
            menu_solve.addAction(act_set_var)
            
            act_set_fix = QAction(f"🔒 Set {p_name} as Fixed (F)", self)
            act_set_fix.setCheckable(True)
            act_set_fix.setChecked(not is_var)
            act_set_fix.triggered.connect(lambda checked, r=row_idx, c=col_idx: self._set_cell_variable(r, c, False))
            menu_solve.addAction(act_set_fix)
            
            act_open_solve = QAction(f"⚙️ Open Solve / Pickup Dialog...", self)
            act_open_solve.triggered.connect(lambda checked, r=row_idx, c=col_idx: self.open_solve_dialog(r, c))
            menu_solve.addAction(act_open_solve)
            menu.addSeparator()

        # Zemax Bulk Presets Menu
        menu_presets = menu.addMenu("📐 Zemax Variable Presets")
        act_z_def = QAction("Zemax Defaults: Air Gaps Variable, Lenses Fixed", self)
        act_z_def.triggered.connect(self._apply_zemax_defaults)
        menu_presets.addAction(act_z_def)

        act_z_rad = QAction("Make All Radii Variable (Curvature Optimization)", self)
        act_z_rad.triggered.connect(self._apply_all_radii_variable)
        menu_presets.addAction(act_z_rad)

        act_z_air = QAction("Make All Air Gaps Variable (Distance Optimization)", self)
        act_z_air.triggered.connect(self._apply_all_air_gaps_variable)
        menu_presets.addAction(act_z_air)

        act_z_fix = QAction("Clear All Variables (All Fixed)", self)
        act_z_fix.triggered.connect(self._apply_all_fixed)
        menu_presets.addAction(act_z_fix)
        menu.addSeparator()

        act_insert = QAction("Insert Surface Row", self)
        act_insert.triggered.connect(lambda: self.on_insert_surface())

        act_cluster = QAction("🔗 Cluster Selected Surfaces as Lens...", self)
        act_cluster.setToolTip("Group selected surfaces into a single multi-surface lens element")
        act_cluster.triggered.connect(self._on_menu_group_selected_surfaces)

        act_uncluster = QAction("🔓 Uncluster / Clear Group", self)
        act_uncluster.triggered.connect(lambda: self._on_menu_clear_group(row_idx))

        act_achromat = QAction("🎨 Design Achromatic Doublet Lens...", self)
        act_achromat.triggered.connect(lambda: self.on_open_achromat_designer_dialog())
        
        component_span = self.model.component_span_for_row(row_idx) if self.model.component_name_mode() else None
        delete_label = f"Delete Complete Lens '{component_span['name']}'" if component_span else "Delete Surface Row"
        act_delete = QAction(delete_label, self)
        act_delete.triggered.connect(lambda: self.on_delete_surface())
        
        menu.addAction(act_cluster)
        menu.addAction(act_uncluster)
        menu.addSeparator()
        menu.addAction(act_insert)
        menu.addAction(act_achromat)
        menu.addAction(act_delete)
        menu.addSeparator()"""

    if old_actions in text:
        text = text.replace(old_actions, new_actions)
        print(f"Updated context menu actions in {file_path}")
    else:
        print(f"old_actions not found in {file_path}")

    # 4. Add helper solve methods to MainWindow
    old_group_clear = """            if hasattr(self, 'component_editor_dock') and hasattr(self.component_editor_dock, '_rebuild_surface_list'):
                self.component_editor_dock._rebuild_surface_list()
            self.on_trace_rays()
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage(f"Unclustered surfaces from '{g_name or f'Surface {row}'}'", 3500)"""

    new_group_clear = """            if hasattr(self, 'component_editor_dock') and hasattr(self.component_editor_dock, '_rebuild_surface_list'):
                self.component_editor_dock._rebuild_surface_list()
            self.on_trace_rays()
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage(f"Unclustered surfaces from '{g_name or f'Surface {row}'}'", 3500)

    def _set_cell_variable(self, row: int, col: int, is_variable: bool):
        if not (0 <= row < len(self.model._surfaces)):
            return
        surf = self.model._surfaces[row]
        if hasattr(self.model, '_push_undo'):
            self.model._push_undo(f"Set solve on surface {row}")
        if col == 2:
            surf.radius_variable = is_variable
            surf.radius_solve_type = "Variable" if is_variable else "Fixed"
        elif col == 3:
            surf.thickness_variable = is_variable
            surf.thickness_solve_type = "Variable" if is_variable else "Fixed"
        elif col == 6:
            surf.conic_variable = is_variable
        elif col in (7, 8):
            setattr(surf, 'decenter_x_variable' if col == 7 else 'decenter_y_variable', is_variable)
        elif col in (9, 10, 11):
            attr = ['tilt_x_variable', 'tilt_y_variable', 'tilt_z_variable'][col - 9]
            setattr(surf, attr, is_variable)
        
        if hasattr(self.model, 'layoutChanged'):
            self.model.layoutChanged.emit()
        if hasattr(self.model, 'system_updated') and self.model.system_updated is not None:
            self.model.system_updated.emit()
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        if hasattr(self, "statusBar") and self.statusBar():
            status_str = "Variable (V)" if is_variable else "Fixed (F)"
            self.statusBar().showMessage(f"Surface {row} parameter set to {status_str}", 2500)

    def _set_selected_cells_solve(self, is_variable: bool):
        \"\"\"Set all currently selected cells in LDE as Variable (V) or Fixed (F).\"\"\"
        if not self.table_view:
            return
        sel = self.table_view.selectionModel()
        indexes = sel.selectedIndexes() if sel and sel.hasSelection() else []
        if not indexes and hasattr(self.table_view, 'currentIndex'):
            c_idx = self.table_view.currentIndex()
            if c_idx and c_idx.isValid():
                indexes = [c_idx]
        if not indexes:
            return
        for idx in indexes:
            r = idx.row()
            c = idx.column()
            if c in (2, 3, 6, 7, 8, 9, 10, 11):
                self._set_cell_variable(r, c, is_variable)
            elif c in (0, 1): # If whole row selected, set radius & thickness
                self._set_cell_variable(r, 2, is_variable)
                self._set_cell_variable(r, 3, is_variable)

    def open_solve_dialog(self, row: int = None, col: int = None):
        \"\"\"Open the Zemax OpticStudio-style Solve & Variable Dialog.\"\"\"
        if not QT_AVAILABLE:
            return
        if row is None or col is None:
            if hasattr(self.table_view, 'currentIndex'):
                idx = self.table_view.currentIndex()
                if idx and idx.isValid():
                    row = idx.row()
                    col = idx.column()
        if row is None or col is None or not (0 <= row < len(self.model._surfaces)):
            row = 0
            col = 2
        
        try:
            from v2.gui.solve_dialog import ZemaxSolveDialog
        except ImportError:
            from optics_app.gui.solve_dialog import ZemaxSolveDialog
            
        dlg = ZemaxSolveDialog(surface_index=row, param_col=col, model=self.model, parent=self)
        exec_fn = getattr(dlg, 'exec_', dlg.exec)
        exec_fn()
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        self.on_trace_rays()

    def _apply_zemax_defaults(self):
        if hasattr(self.model, '_push_undo'):
            self.model._push_undo("Apply Zemax Defaults (Air Gaps Variable)")
        if hasattr(self.model, 'set_zemax_defaults'):
            self.model.set_zemax_defaults(var_air_gaps=True, fix_lenses=True)
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage("Applied Zemax Defaults: Air Gaps Variable, Lenses Fixed", 3000)

    def _apply_all_radii_variable(self):
        if hasattr(self.model, '_push_undo'):
            self.model._push_undo("Make All Radii Variable")
        if hasattr(self.model, 'make_all_radii_variable'):
            self.model.make_all_radii_variable()
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage("Set all surface radii as Variable for curvature optimization", 3000)

    def _apply_all_air_gaps_variable(self):
        if hasattr(self.model, '_push_undo'):
            self.model._push_undo("Make All Air Gaps Variable")
        if hasattr(self.model, 'make_all_air_gaps_variable'):
            self.model.make_all_air_gaps_variable()
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage("Set all air gaps as Variable for distance optimization", 3000)

    def _apply_all_fixed(self):
        if hasattr(self.model, '_push_undo'):
            self.model._push_undo("Set All Parameters Fixed")
        if hasattr(self.model, 'set_all_fixed'):
            self.model.set_all_fixed()
        if hasattr(self, 'optimizer_dock') and hasattr(self.optimizer_dock, '_rebuild_surface_checkboxes'):
            self.optimizer_dock._rebuild_surface_checkboxes()
        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage("Cleared all variables: All parameters set to Fixed", 3000)"""

    if old_group_clear in text:
        text = text.replace(old_group_clear, new_group_clear)
        print(f"Updated helper solve methods in {file_path}")
    else:
        print(f"old_group_clear not found in {file_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == '__main__':
    update_main_window('v2/gui/main_window.py')
    update_main_window('optics_app/gui/main_window.py')
