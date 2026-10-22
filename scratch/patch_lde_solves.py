"""
Script to apply Zemax-style solve highlights and methods to lde_table.py and main_window.py
"""

def update_lde_table(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Update BackgroundRole
    old_bg = """        if role_val == 8: # BackgroundRole
            try:
                from optics_app.gui.qt_compat import QBrush, QColor
                from optics_app.engine.element_colors import get_surface_element_group
                
                group = get_surface_element_group(self._surfaces, r)
                pal = group.get("palette")
                if pal and "bg_hex" in pal:
                    return QBrush(QColor(pal["bg_hex"]))
            except Exception:
                pass
            return None"""

    new_bg = """        if role_val == 8: # BackgroundRole
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
                        return QBrush(QColor("#bae6fd")) # Zemax cyan/blue variable badge
                    elif is_pickup:
                        return QBrush(QColor("#fef08a")) # Soft amber pickup badge
                    elif is_solve:
                        return QBrush(QColor("#e9d5ff")) # Soft lavender ray solve

                group = get_surface_element_group(self._surfaces, r)
                pal = group.get("palette")
                if pal and "bg_hex" in pal:
                    return QBrush(QColor(pal["bg_hex"]))
            except Exception:
                pass
            return None"""

    if old_bg in text:
        text = text.replace(old_bg, new_bg)
        print(f"Updated BackgroundRole in {file_path}")
    else:
        print(f"old_bg not found in {file_path}")

    # 2. Update DisplayRole
    old_disp = """                # Display Zemax 'V' indicator next to variable parameters on DisplayRole
                if actual_role in (0, getattr(Qt, 'DisplayRole', 0)):
                    if c == 3 and r < len(self._surfaces) and getattr(self._surfaces[r], 'thickness_variable', False):
                        try:
                            return f"{float(val):.3f} V"
                        except (ValueError, TypeError):
                            return f"{val} V"
                    elif c == 2 and r < len(self._surfaces) and getattr(self._surfaces[r], 'radius_variable', False):
                        try:
                            return f"{float(val):.3f} V"
                        except (ValueError, TypeError):
                            return f"{val} V"
                    elif c == 4 and r < len(self._surfaces):
                        from optics_app.engine.element_colors import get_surface_element_group, format_surface_material_display
                        group = get_surface_element_group(self._surfaces, r)
                        g_type = group.get("group_type", "")
                        if g_type != "LENS" and not getattr(self._surfaces[r], 'is_glass', False):
                            return format_surface_material_display(self._surfaces[r], r, len(self._surfaces))
                return val"""

    new_disp = """                # Display Zemax 'V', 'P', 'S' indicator next to variable parameters on DisplayRole
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
                return val"""

    if old_disp in text:
        text = text.replace(old_disp, new_disp)
        print(f"Updated DisplayRole in {file_path}")
    else:
        print(f"old_disp not found in {file_path}")

    # 3. Add helper methods to LensDataTableModel: make_all_radii_variable, make_all_air_gaps_variable
    old_methods = """    def set_all_fixed(self):
        \"\"\"Clear all variable flags, setting all surfaces to Fixed.\"\"\"
        for surf in self._surfaces:
            surf.thickness_variable = False
            surf.radius_variable = False
        self._sync_stack_from_data()
        if hasattr(self, 'system_updated') and self.system_updated is not None:
            try:
                self.system_updated.emit()
            except Exception:
                pass"""

    new_methods = """    def set_all_fixed(self):
        \"\"\"Clear all variable flags, setting all surfaces to Fixed.\"\"\"
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
        \"\"\"Set all optical surface radii as variable (for curvature optimization).\"\"\"
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
        \"\"\"Set all air gap propagation distances as variable (for spacing optimization).\"\"\"
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
                pass"""

    if old_methods in text:
        text = text.replace(old_methods, new_methods)
        print(f"Updated methods in {file_path}")
    else:
        print(f"old_methods not found in {file_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == '__main__':
    update_lde_table('optics_app/gui/lde_table.py')
    update_lde_table('v2/gui/lde_table.py')
