import os
from optics_app.gui.qt_compat import QtWidgets, QtCore
Qt = QtCore.Qt
from optics_app.engine.surface import Surface, SurfaceType

class ZemaxImportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, max_surfaces=0):
        super().__init__(parent)
        self.setWindowTitle("Insert Zemax Element")
        self.max_surfaces = max_surfaces
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # 1. Insert After Index
        idx_layout = QtWidgets.QHBoxLayout()
        idx_label = QtWidgets.QLabel("Insert After Surface:")
        self.index_spin = QtWidgets.QSpinBox()
        self.index_spin.setRange(0, max(0, self.max_surfaces))
        self.index_spin.setValue(self.max_surfaces)
        idx_layout.addWidget(idx_label)
        idx_layout.addWidget(self.index_spin)
        layout.addLayout(idx_layout)

        # 2. Reverse Element
        self.reverse_check = QtWidgets.QCheckBox("Flip Element Front-to-Back (Reverse)")
        self.reverse_check.setToolTip("Reverses the order and flips the radii of the imported surfaces")
        layout.addWidget(self.reverse_check)

        # 3. Rotation (Tilt X, Y, Z)
        rot_label = QtWidgets.QLabel("Apply Rotation (Tilt on first surface):")
        layout.addWidget(rot_label)
        
        rot_layout = QtWidgets.QHBoxLayout()
        self.tilt_x = QtWidgets.QDoubleSpinBox()
        self.tilt_x.setRange(-360, 360)
        self.tilt_x.setSuffix(" deg")
        self.tilt_y = QtWidgets.QDoubleSpinBox()
        self.tilt_y.setRange(-360, 360)
        self.tilt_y.setSuffix(" deg")
        self.tilt_z = QtWidgets.QDoubleSpinBox()
        self.tilt_z.setRange(-360, 360)
        self.tilt_z.setSuffix(" deg")
        
        rot_layout.addWidget(QtWidgets.QLabel("X:"))
        rot_layout.addWidget(self.tilt_x)
        rot_layout.addWidget(QtWidgets.QLabel("Y:"))
        rot_layout.addWidget(self.tilt_y)
        rot_layout.addWidget(QtWidgets.QLabel("Z:"))
        rot_layout.addWidget(self.tilt_z)
        layout.addLayout(rot_layout)

        # Buttons
        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_settings(self):
        return {
            "insert_index": self.index_spin.value(),
            "reverse": self.reverse_check.isChecked(),
            "tilt_x": self.tilt_x.value(),
            "tilt_y": self.tilt_y.value(),
            "tilt_z": self.tilt_z.value()
        }

def process_imported_surfaces(imported_surfaces, settings):
    import copy
    surfs = [copy.deepcopy(s) for s in imported_surfaces]

    # 1. Filter out Object surface from the beginning if present
    if len(surfs) > 0 and (
        surfs[0].surface_type in (SurfaceType.OBJECT, "Object") or
        surfs[0].name.upper() in ("OBJ", "OBJECT", "SURFACE_0", "S0") or
        (str(surfs[0].glass or "").upper() in ("AIR", "", "NONE") and len(surfs) > 1 and getattr(surfs[1], 'is_glass', False))
    ):
        surfs = surfs[1:]

    # 2. Filter out trailing Detector / Image surface from the end if present
    if len(surfs) > 0 and (
        surfs[-1].surface_type in (SurfaceType.DETECTOR, "Detector", "Image") or
        surfs[-1].name.upper() in ("IMA", "IMAGE", "DETECTOR") or
        (str(surfs[-1].glass or "").upper() in ("AIR", "", "NONE") and surfs[-1].thickness == 0 and len(surfs) > 1)
    ):
        surfs = surfs[:-1]

    # Ensure none of the imported lens surfaces retain OBJECT or DETECTOR surface type
    for s in surfs:
        if s.surface_type in (SurfaceType.OBJECT, SurfaceType.DETECTOR):
            s.surface_type = SurfaceType.STANDARD

    # Ensure the rear surface of the lens element has glass="AIR" and a sensible propagation thickness
    if len(surfs) > 0:
        if surfs[-1].thickness <= 0:
            surfs[-1].thickness = 10.0

    if settings.get("reverse", False) and len(surfs) > 1:
        reversed_surfs = []
        n = len(surfs)
        for i in range(n - 1, -1, -1):
            s = copy.deepcopy(surfs[i])
            if s.radius != 0:
                s.radius = -s.radius
            
            if i > 0:
                s.thickness = surfs[i-1].thickness
                s.glass = surfs[i-1].glass
            else:
                s.thickness = surfs[-1].thickness
                s.glass = "AIR"
            reversed_surfs.append(s)
        surfs = reversed_surfs

    tilt_x = settings.get("tilt_x", 0.0)
    tilt_y = settings.get("tilt_y", 0.0)
    tilt_z = settings.get("tilt_z", 0.0)
    if tilt_x != 0 or tilt_y != 0 or tilt_z != 0:
        if surfs:
            surfs[0].tilt_x += tilt_x
            surfs[0].tilt_y += tilt_y
            surfs[0].tilt_z += tilt_z

    return surfs
