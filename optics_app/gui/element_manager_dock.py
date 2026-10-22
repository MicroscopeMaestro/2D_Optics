"""
Optical Element Manager UI Dock for V2 Spectrometer Design Studio.
Provides UI controls and programmatic interfaces to:
- Import Thorlabs Zemax (.zmx) files
- Insert catalog doublet / singlet lenses
- Flip lens elements (R -> -R, glass sequence reversal, air gap preservation)
- Rotate / tilt lens elements in 3D around 3 selectable pivot points (front_vertex, center_of_mass, rear_vertex)
- Decenter lens elements in 3D
- Delete lens elements with automatic air gap merging
- Replace lens elements from catalog or file
"""
from typing import Optional, List, Dict, Any
import os

from v2.gui.qt_compat import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox,
    pyqtSignal, Signal, QT_AVAILABLE
)
from v2.engine.lde import SurfaceStack
from v2.engine.surface import Surface, SurfaceType
from v2.io.zmx_parser import parse_zmx_file


class ElementManagerDockWidget(QDockWidget):
    """
    Optical Element Manager UI Dock.
    """
    elementModified = Signal() if QT_AVAILABLE else pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, stack: Optional[SurfaceStack] = None):
        super().__init__("Optical Element Manager", parent)
        self.stack = stack
        self._init_ui()

    def _init_ui(self):
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Target element selection
        h_sel = QHBoxLayout()
        h_sel.addWidget(QLabel("Target Surface Index:"))
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 100)
        self.target_spin.setValue(4)  # Focuser default
        h_sel.addWidget(self.target_spin)
        layout.addLayout(h_sel)

        # 1. ZMX Import & Insert
        grp_import = QVBoxLayout()
        self.btn_import_zmx = QPushButton("Import & Replace from .zmx...")
        self.btn_import_zmx.clicked.connect(self._on_import_zmx_clicked)
        grp_import.addWidget(self.btn_import_zmx)

        self.btn_insert_zmx = QPushButton("Insert .zmx After Target Index...")
        self.btn_insert_zmx.clicked.connect(self._on_insert_zmx_clicked)
        grp_import.addWidget(self.btn_insert_zmx)
        layout.addLayout(grp_import)

        # 2. Catalog Lens Insertion & Swap
        h_cat = QHBoxLayout()
        h_cat.addWidget(QLabel("Catalog Lens:"))
        self.combo_catalog = QComboBox()
        self.combo_catalog.addItems(sorted(SurfaceStack._get_lens_catalog().keys()))
        self.combo_catalog.setCurrentText("AC254-075-C")
        h_cat.addWidget(self.combo_catalog)
        layout.addLayout(h_cat)

        h_cat_btns = QHBoxLayout()
        self.btn_swap_catalog = QPushButton("Swap with Catalog")
        self.btn_swap_catalog.clicked.connect(self._on_swap_catalog_clicked)
        self.btn_insert_catalog = QPushButton("Insert Catalog Lens")
        self.btn_insert_catalog.clicked.connect(self._on_insert_catalog_clicked)
        h_cat_btns.addWidget(self.btn_swap_catalog)
        h_cat_btns.addWidget(self.btn_insert_catalog)
        layout.addLayout(h_cat_btns)

        # 3. Dynamic Flip (R -> -R)
        self.btn_flip = QPushButton("Flip Element (Front-to-Back)")
        self.btn_flip.clicked.connect(self._on_flip_clicked)
        layout.addWidget(self.btn_flip)

        # 4. 3D Rigid Body Rotation around Pivots
        grid_rot = QGridLayout()
        grid_rot.addWidget(QLabel("Pivot Point:"), 0, 0)
        self.combo_pivot = QComboBox()
        self.combo_pivot.addItems(["front_vertex", "center_of_mass", "rear_vertex"])
        grid_rot.addWidget(self.combo_pivot, 0, 1, 1, 2)

        grid_rot.addWidget(QLabel("Tilt X (°):"), 1, 0)
        self.spin_tilt_x = QDoubleSpinBox()
        self.spin_tilt_x.setRange(-45.0, 45.0)
        self.spin_tilt_x.setValue(0.0)
        grid_rot.addWidget(self.spin_tilt_x, 1, 1)

        grid_rot.addWidget(QLabel("Tilt Y (°):"), 2, 0)
        self.spin_tilt_y = QDoubleSpinBox()
        self.spin_tilt_y.setRange(-45.0, 45.0)
        self.spin_tilt_y.setValue(0.0)
        grid_rot.addWidget(self.spin_tilt_y, 2, 1)

        grid_rot.addWidget(QLabel("Tilt Z (°):"), 3, 0)
        self.spin_tilt_z = QDoubleSpinBox()
        self.spin_tilt_z.setRange(-45.0, 45.0)
        self.spin_tilt_z.setValue(0.0)
        grid_rot.addWidget(self.spin_tilt_z, 3, 1)

        self.btn_apply_rotation = QPushButton("Apply 3D Rotation")
        self.btn_apply_rotation.clicked.connect(self._on_apply_rotation_clicked)
        grid_rot.addWidget(self.btn_apply_rotation, 4, 0, 1, 3)
        layout.addLayout(grid_rot)

        # 5. Delete Element with Air Gap Merge
        self.btn_delete = QPushButton("Delete Element (Merge Air Gap)")
        self.btn_delete.setStyleSheet("background-color: #8B0000; color: white;")
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        layout.addWidget(self.btn_delete)

        layout.addStretch()
        self.setWidget(container)

    def set_surface_stack(self, stack: SurfaceStack):
        self.stack = stack

    def _on_import_zmx_clicked(self):
        if self.stack is None:
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Zemax .zmx File", "", "Zemax Files (*.zmx);;All Files (*)")
        if file_path and os.path.exists(file_path):
            idx = self.target_spin.value()
            self.stack.replace_lens_from_zmx(target=idx, file_path=file_path)
            self.elementModified.emit()

    def _on_insert_zmx_clicked(self):
        if self.stack is None:
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Zemax .zmx File", "", "Zemax Files (*.zmx);;All Files (*)")
        if file_path and os.path.exists(file_path):
            zmx_stack = parse_zmx_file(file_path)
            if zmx_stack and len(zmx_stack.surfaces) >= 2:
                idx = self.target_spin.value()
                # Insert surfaces after idx
                for s in zmx_stack.surfaces[1:-1]:
                    self.stack.surfaces.insert(idx + 1, s)
                    idx += 1
                self.stack._update_accumulated_positions()
                self.elementModified.emit()

    def _on_swap_catalog_clicked(self):
        if self.stack is None:
            return
        idx = self.target_spin.value()
        part = self.combo_catalog.currentText()
        self.stack.replace_lens_from_catalog(target=idx, part_number=part)
        self.elementModified.emit()

    def _on_insert_catalog_clicked(self):
        if self.stack is None:
            return
        idx = self.target_spin.value()
        part = self.combo_catalog.currentText()
        self.stack.insert_doublet(after_surface_index=idx, name=f"Catalog_{part}")
        self.elementModified.emit()

    def _on_flip_clicked(self):
        if self.stack is None:
            return
        idx = self.target_spin.value()
        # Delegate to the optical model so radii, glass order, center
        # thicknesses, and the trailing air gap are all transformed together.
        indices = self.stack.flip_element(idx)
        if indices:
            self.elementModified.emit()

    def _on_apply_rotation_clicked(self):
        if self.stack is None:
            return
        idx = self.target_spin.value()
        tx = self.spin_tilt_x.value()
        ty = self.spin_tilt_y.value()
        tz = self.spin_tilt_z.value()
        pivot = self.combo_pivot.currentText()
        if hasattr(self.stack, "rotate_lens_element"):
            self.stack.rotate_lens_element(front_surface_index=idx, tilt_x=tx, tilt_y=ty, tilt_z=tz, pivot=pivot)
        self.elementModified.emit()

    def _on_delete_clicked(self):
        if self.stack is None:
            return
        idx = self.target_spin.value()
        if hasattr(self.stack, "delete_lens_element"):
            self.stack.delete_lens_element(target=idx)
        self.elementModified.emit()
