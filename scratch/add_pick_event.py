import re

path = 'optics_app/gui/visualizer_3d.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add signal
content = content.replace('transparency_changed = pyqtSignal(bool) if QT_AVAILABLE else DummySignal()', 
                          'transparency_changed = pyqtSignal(bool) if QT_AVAILABLE else DummySignal()\n    surface_selected = pyqtSignal(int) if QT_AVAILABLE else DummySignal()')

# Add mpl_connect
content = content.replace("self.canvas.mpl_connect('button_press_event', self.on_button_press)",
                          "self.canvas.mpl_connect('button_press_event', self.on_button_press)\n        self.canvas.mpl_connect('pick_event', self.on_pick)")

# Add on_pick method
on_pick_method = '''
    def on_pick(self, event):
        if not hasattr(event, "artist"): return
        idx = getattr(event.artist, '_surface_index', None)
        if idx is not None:
            self.surface_selected.emit(idx)
            
    def on_button_press'''
content = content.replace('    def on_button_press', on_pick_method.strip('\n'))

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
