import re

path = 'optics_app/gui/main_window.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''        self.visualizer_3d = Visualizer3DWidget(model=self.model, parent=self, auto_refresh=False)
        if hasattr(self.visualizer_3d, "transparency_changed"):
            try:
                self.visualizer_3d.transparency_changed.connect(self._on_3d_transparency_changed)
            except Exception:
                pass'''

new_logic = '''        self.visualizer_3d = Visualizer3DWidget(model=self.model, parent=self, auto_refresh=False)
        if hasattr(self.visualizer_3d, "transparency_changed"):
            try:
                self.visualizer_3d.transparency_changed.connect(self._on_3d_transparency_changed)
            except Exception:
                pass
        if hasattr(self.visualizer_3d, "surface_selected"):
            try:
                self.visualizer_3d.surface_selected.connect(self._on_3d_surface_selected)
            except Exception:
                pass'''

content = content.replace(old_logic, new_logic)

# Add _on_3d_surface_selected method
handler_code = '''
    def _on_3d_surface_selected(self, idx: int):
        if hasattr(self, 'table_view'):
            self.table_view.selectRow(idx)
        if hasattr(self, 'comp_editor'):
            self.comp_editor.set_selected_surface(idx)
            
    def _on_toggle_3d_no_transparency(self, checked: bool):'''
content = content.replace('    def _on_toggle_3d_no_transparency(self, checked: bool):', handler_code.strip('\n'))

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
