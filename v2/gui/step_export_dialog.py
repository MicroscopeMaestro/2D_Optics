import os
import sys
from typing import List, Dict, Any, Optional, Collection

from optics_app.gui.qt_compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QLineEdit, QFileDialog, QComboBox, QGroupBox, QMessageBox,
    QFrame, QAbstractItemView, Qt, QT_AVAILABLE, QtWidgets
)

from optics_app.cad.exporter import get_exportable_elements, STEPExportEngine


def _checkmark_icon_path() -> str:
    """Render a white check-mark PNG once (temp dir) for use in Qt stylesheets."""
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "2d_optics_checkmark_16.png").replace("\\", "/")
    if os.path.exists(path):
        return path
    try:
        QtGui = __import__(QtWidgets.__name__.rsplit(".", 1)[0] + ".QtGui", fromlist=["QtGui"])
        QtCore = __import__(QtWidgets.__name__.rsplit(".", 1)[0] + ".QtCore", fromlist=["QtCore"])
        pix = QtGui.QPixmap(16, 16)
        pix.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor("#ffffff"))
        pen.setWidthF(2.4)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPolyline([QtCore.QPointF(3.5, 8.5), QtCore.QPointF(6.5, 11.5), QtCore.QPointF(12.5, 4.5)])
        painter.end()
        pix.save(path, "PNG")
    except Exception:
        return ""
    return path


class STEPExportDialog(QDialog if QT_AVAILABLE else object):
    """
    Interactive Pop-Up Dialog to choose which optical elements to export together into a 3D STEP assembly.
    Supports individual component selection, batch category filtering, CAD engine choice, and output path configuration.
    """

    def __init__(
        self,
        system_or_stack: Any,
        parent=None,
        default_file_path: Optional[str] = None,
        default_engine: str = "auto",
        default_air_gap_diam: float = 25.4,
        default_mount_diam: float = 30.5,
        default_mount_z: float = 0.0,
        default_export_mounts: bool = False,
        default_tol_z: float = 1.0,
    ):
        if QT_AVAILABLE:
            if hasattr(QtWidgets, "QApplication") and QtWidgets.QApplication.instance() is None:
                self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
                    sys.argv if hasattr(sys, "argv") and sys.argv else ["2D_Optics"]
                )
            super().__init__(parent)
            self.setWindowTitle("Export 3D STEP Assembly")
            self.setMinimumSize(780, 520)
            self.resize(840, 560)
            if hasattr(self, "setWindowModality"):
                self.setWindowModality(Qt.WindowModal)

        self.system_or_stack = system_or_stack
        self.elements: List[Dict[str, Any]] = get_exportable_elements(system_or_stack)
        self.default_file_path = default_file_path or "optical_assembly.step"
        self.default_engine = default_engine
        self.default_air_gap_diam = float(default_air_gap_diam if default_air_gap_diam and default_air_gap_diam > 0 else 25.4)
        self.default_mount_diam = float(default_mount_diam if default_mount_diam and default_mount_diam > 0 else 30.5)
        self.default_mount_z = float(default_mount_z if default_mount_z and default_mount_z >= 0 else 0.0)
        self.default_export_mounts = bool(default_export_mounts)
        self.default_tol_z = float(default_tol_z if default_tol_z and default_tol_z >= 0 else 1.0)

        # Internal state for headless/fallback environments
        self._selection_map: Dict[int, bool] = {
            el["index"]: bool(el.get("is_default_selected", True))
            for el in self.elements
        }
        self._chosen_path: str = self.default_file_path
        self._chosen_engine: str = self.default_engine

        # GUI widget references
        self.table_elements: Optional[Any] = None
        self.lbl_counter: Optional[Any] = None
        self.txt_file_path: Optional[Any] = None
        self.combo_engine: Optional[Any] = None
        self.combo_air_gap_diam: Optional[Any] = None
        self.chk_export_mounts: Optional[Any] = None
        self.combo_mount_diam: Optional[Any] = None
        self.combo_mount_z: Optional[Any] = None
        self.combo_tol_z: Optional[Any] = None
        self.chk_enclosure: Optional[Any] = None
        self.btn_export: Optional[Any] = None
        self.btn_cancel: Optional[Any] = None

        if QT_AVAILABLE:
            self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        # 1. Header Banner
        banner_frame = QFrame()
        banner_frame.setStyleSheet(
            "QFrame { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px; }"
        )
        banner_layout = QVBoxLayout(banner_frame)
        banner_layout.setContentsMargins(8, 6, 8, 6)
        banner_layout.setSpacing(2)

        lbl_title = QLabel("📦 Export 3D STEP CAD Assembly (.step)")
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #0f172a;")
        banner_layout.addWidget(lbl_title)

        lbl_desc = QLabel(
            "Select which optical components to export together into a single unified 3D STEP assembly. "
            "All elements preserve their exact global 3D assembly coordinates and orientations."
        )
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("font-size: 11px; color: #475569;")
        banner_layout.addWidget(lbl_desc)

        root_layout.addWidget(banner_frame)

        # 2. Batch Selection Toolbar
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(6)

        btn_style_batch = (
            "QPushButton { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; "
            "padding: 4px 8px; font-size: 11px; font-weight: 600; color: #334155; } "
            "QPushButton:hover { background-color: #e2e8f0; border-color: #94a3b8; color: #0f172a; } "
            "QPushButton:pressed { background-color: #cbd5e1; }"
        )

        btn_all = QPushButton("✅ Select All")
        btn_all.setStyleSheet(btn_style_batch)
        btn_all.clicked.connect(self.select_all)
        btn_bar.addWidget(btn_all)

        btn_none = QPushButton("❌ Deselect All")
        btn_none.setStyleSheet(btn_style_batch)
        btn_none.clicked.connect(self.deselect_all)
        btn_bar.addWidget(btn_none)

        btn_lenses = QPushButton("🔍 Lenses Only")
        btn_lenses.setStyleSheet(btn_style_batch)
        btn_lenses.clicked.connect(lambda: self.select_by_type(["Singlet", "Doublet", "Lens"]))
        btn_bar.addWidget(btn_lenses)

        btn_mirrors = QPushButton("🪞 Mirrors & Gratings")
        btn_mirrors.setStyleSheet(btn_style_batch)
        btn_mirrors.clicked.connect(lambda: self.select_by_type(["Mirror", "Grating", "Mechanism", "DMD"]))
        btn_bar.addWidget(btn_mirrors)

        btn_stops = QPushButton("⭕ Stops & Apertures")
        btn_stops.setStyleSheet(btn_style_batch)
        btn_stops.clicked.connect(lambda: self.select_by_type(["Aperture Stop", "Stop"]))
        btn_bar.addWidget(btn_stops)

        btn_invert = QPushButton("🔄 Invert Selection")
        btn_invert.setStyleSheet(btn_style_batch)
        btn_invert.clicked.connect(self.invert_selection)
        btn_bar.addWidget(btn_invert)

        btn_bar.addStretch(1)

        self.lbl_counter = QLabel("Selected: 0 of 0 components")
        self.lbl_counter.setStyleSheet("font-size: 11px; font-weight: bold; color: #1e40af; padding-right: 4px;")
        btn_bar.addWidget(self.lbl_counter)

        root_layout.addLayout(btn_bar)

        # 3. Elements Table Widget
        self.table_elements = QTableWidget()
        headers = [
            "Include", "#", "Component Name", "Type", "Surfaces",
            "Material", "Semi-Diam (mm)", "Ø Clear Aperture (mm)", "Global Z (mm)"
        ]
        self.table_elements.setColumnCount(len(headers))
        self.table_elements.setHorizontalHeaderLabels(headers)
        self.table_elements.setAlternatingRowColors(True)
        self.table_elements.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_elements.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_elements.verticalHeader().setVisible(False)
        check_img = _checkmark_icon_path()
        checked_img_rule = f"image: url({check_img}); " if check_img else ""
        self.table_elements.setStyleSheet(
            "QTableWidget { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; gridline-color: #e2e8f0; font-size: 11px; } "
            "QHeaderView::section { background-color: #f8fafc; color: #334155; font-weight: bold; border: 1px solid #e2e8f0; padding: 4px; } "
            "QTableWidget::item:selected { background-color: #e0f2fe; color: #0369a1; } "
            "QTableWidget::indicator { width: 16px; height: 16px; border: 1px solid #64748b; border-radius: 3px; background-color: #ffffff; } "
            "QTableWidget::indicator:hover { border-color: #16a34a; } "
            f"QTableWidget::indicator:checked {{ background-color: #16a34a; border-color: #15803d; {checked_img_rule}}} "
        )

        self.table_elements.setRowCount(len(self.elements))
        for row, el in enumerate(self.elements):
            idx = el["index"]
            is_sel = self._selection_map.get(idx, True)

            # Col 0: Checkbox
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            chk_item.setCheckState(Qt.Checked if is_sel else Qt.Unchecked)
            self.table_elements.setItem(row, 0, chk_item)

            # Col 1: Index (#)
            item_num = QTableWidgetItem(str(row + 1))
            item_num.setTextAlignment(Qt.AlignCenter)
            self.table_elements.setItem(row, 1, item_num)

            # Col 2: Name
            item_name = QTableWidgetItem(str(el.get("name", f"Element {row+1}")))
            item_name.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.table_elements.setItem(row, 2, item_name)

            # Col 3: Type
            item_type = QTableWidgetItem(str(el.get("type", "Unknown")))
            item_type.setTextAlignment(Qt.AlignCenter)
            self.table_elements.setItem(row, 3, item_type)

            # Col 4: Surfaces
            surfs = el.get("surface_indices", [])
            if len(surfs) == 1:
                s_str = f"S{surfs[0]}"
            elif len(surfs) > 1:
                s_str = f"S{surfs[0]} - S{surfs[-1]}"
            else:
                s_str = "-"
            item_surfs = QTableWidgetItem(s_str)
            item_surfs.setTextAlignment(Qt.AlignCenter)
            self.table_elements.setItem(row, 4, item_surfs)

            # Col 5: Material
            mat_str = str(el.get("material", "-"))
            item_mat = QTableWidgetItem(mat_str)
            item_mat.setTextAlignment(Qt.AlignCenter)
            self.table_elements.setItem(row, 5, item_mat)

            # Col 6: Semi-Diameter
            sd = float(el.get("semi_diameter", 0.0))
            item_sd = QTableWidgetItem(f"{sd:.2f} mm")
            item_sd.setTextAlignment(Qt.AlignCenter)
            self.table_elements.setItem(row, 6, item_sd)

            # Col 7: Clear Aperture Diameter
            ca_diam = float(el.get("clear_aperture_diam", 2.0 * sd))
            item_diam = QTableWidgetItem(f"{ca_diam:.2f} mm")
            item_diam.setTextAlignment(Qt.AlignCenter)
            self.table_elements.setItem(row, 7, item_diam)

            # Col 8: Global Z Position
            z_pos = float(el.get("z_position", 0.0))
            item_z = QTableWidgetItem(f"{z_pos:.2f} mm")
            item_z.setTextAlignment(Qt.AlignCenter)
            self.table_elements.setItem(row, 8, item_z)

        # Connect cell changed signal to update live counter and selection map
        self.table_elements.itemChanged.connect(self._on_table_item_changed)

        # Resize header columns appropriately
        header = self.table_elements.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)

        root_layout.addWidget(self.table_elements, 1)

        # 4. CAD Export Options Box
        opt_group = QGroupBox("CAD Export Settings")
        opt_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #1e293b; border: 1px solid #cbd5e1; border-radius: 4px; margin-top: 6px; padding-top: 10px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        opt_layout = QVBoxLayout(opt_group)
        opt_layout.setSpacing(6)

        # Output File Row
        row_file = QHBoxLayout()
        lbl_file = QLabel("Output STEP File:")
        lbl_file.setStyleSheet("font-size: 11px; font-weight: 600; color: #334155;")
        row_file.addWidget(lbl_file)

        self.txt_file_path = QLineEdit(self.default_file_path)
        self.txt_file_path.setStyleSheet(
            "QLineEdit { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px 6px; font-size: 11px; }"
        )
        row_file.addWidget(self.txt_file_path, 1)

        btn_browse = QPushButton("Browse...")
        btn_browse.setStyleSheet(btn_style_batch)
        btn_browse.clicked.connect(self._on_browse_file)
        row_file.addWidget(btn_browse)

        opt_layout.addLayout(row_file)

        # Engine Row
        row_engine = QHBoxLayout()
        lbl_engine = QLabel("CAD Engine:")
        lbl_engine.setStyleSheet("font-size: 11px; font-weight: 600; color: #334155;")
        row_engine.addWidget(lbl_engine)

        self.combo_engine = QComboBox()
        self.combo_engine.setStyleSheet(
            "QComboBox { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 3px 6px; font-size: 11px; }"
        )
        self.combo_engine.addItem("Auto (build123d / OpenCASCADE if available, else Pure Python)", "auto")
        self.combo_engine.addItem("build123d / OpenCASCADE (High-Precision 3D BREP Solids)", "build123d")
        self.combo_engine.addItem("Pure Python STEP AP203 (Direct ASCII ISO-10303-21)", "pure_python")

        if self.default_engine == "build123d":
            self.combo_engine.setCurrentIndex(1)
        elif self.default_engine == "pure_python":
            self.combo_engine.setCurrentIndex(2)
        else:
            self.combo_engine.setCurrentIndex(0)

        row_engine.addWidget(self.combo_engine, 1)
        opt_layout.addLayout(row_engine)

        # Air Gap Bore Diameter Row
        row_air = QHBoxLayout()
        lbl_air = QLabel("Air Gap Bore Ø:")
        lbl_air.setStyleSheet("font-size: 11px; font-weight: 600; color: #334155;")
        lbl_air.setToolTip("Set constant bore diameter for air tunnels/cylinders, or Auto to loft between optical clear apertures.")
        row_air.addWidget(lbl_air)

        self.combo_air_gap_diam = QComboBox()
        self.combo_air_gap_diam.setEditable(True)
        self.combo_air_gap_diam.setStyleSheet(
            "QComboBox { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 3px 6px; font-size: 11px; }"
        )
        self.combo_air_gap_diam.addItems([
            "25.4 mm (1\")",
            "12.7 mm (1/2\")",
            "20.0 mm",
            "25.0 mm",
            "30.0 mm (Cage)",
            "38.1 mm (1.5\")",
            "50.0 mm",
            "50.8 mm (2\")",
        ])
        init_diam = float(self.default_air_gap_diam or 25.4)
        import re
        found = False
        for idx in range(self.combo_air_gap_diam.count()):
            item_text = self.combo_air_gap_diam.itemText(idx)
            m = re.search(r"[-+]?\d*\.?\d+", item_text)
            if m and abs(float(m.group(0)) - init_diam) < 1e-3:
                self.combo_air_gap_diam.setCurrentIndex(idx)
                found = True
                break
        if not found:
            self.combo_air_gap_diam.setEditText(f"{init_diam:.1f} mm")
        row_air.addWidget(self.combo_air_gap_diam, 1)
        opt_layout.addLayout(row_air)

        # Optical Element Mechanical Mounts Row
        row_mounts = QHBoxLayout()
        self.chk_export_mounts = QCheckBox("Generate Mounts / Cells")
        self.chk_export_mounts.setChecked(self.default_export_mounts)
        self.chk_export_mounts.setStyleSheet(
            "QCheckBox { font-size: 11px; font-weight: bold; color: #1e293b; } "
            "QCheckBox::indicator { width: 14px; height: 14px; }"
        )
        self.chk_export_mounts.setToolTip("Export outer mechanical mounting sleeves/cells around lenses and optical components.")
        row_mounts.addWidget(self.chk_export_mounts)

        lbl_m_diam = QLabel("Mount Ø:")
        lbl_m_diam.setStyleSheet("font-size: 11px; font-weight: 600; color: #334155;")
        row_mounts.addWidget(lbl_m_diam)

        self.combo_mount_diam = QComboBox()
        self.combo_mount_diam.setEditable(True)
        self.combo_mount_diam.setStyleSheet(
            "QComboBox { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 3px 6px; font-size: 11px; }"
        )
        self.combo_mount_diam.addItems([
            "30.5 mm (SM1)",
            "Auto (+5 mm)",
            "25.4 mm (1\")",
            "28.0 mm",
            "30.0 mm",
            "32.0 mm",
            "35.0 mm",
            "50.0 mm",
            "56.0 mm (SM2)",
        ])
        init_m_diam = float(self.default_mount_diam or 30.5)
        found_m = False
        for idx in range(self.combo_mount_diam.count()):
            item_text = self.combo_mount_diam.itemText(idx)
            if init_m_diam == 0.0 and "auto" in item_text.lower():
                self.combo_mount_diam.setCurrentIndex(idx)
                found_m = True
                break
            m = re.search(r"[-+]?\d*\.?\d+", item_text)
            if m and abs(float(m.group(0)) - init_m_diam) < 1e-3:
                self.combo_mount_diam.setCurrentIndex(idx)
                found_m = True
                break
        if not found_m:
            self.combo_mount_diam.setEditText(f"{init_m_diam:.1f} mm")
        row_mounts.addWidget(self.combo_mount_diam, 1)

        lbl_m_z = QLabel("Z Size:")
        lbl_m_z.setStyleSheet("font-size: 11px; font-weight: 600; color: #334155;")
        row_mounts.addWidget(lbl_m_z)

        self.combo_mount_z = QComboBox()
        self.combo_mount_z.setEditable(True)
        self.combo_mount_z.setStyleSheet(
            "QComboBox { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 3px 6px; font-size: 11px; }"
        )
        self.combo_mount_z.addItems([
            "Auto (Lens Z)",
            "3.0 mm",
            "5.0 mm",
            "8.0 mm",
            "10.0 mm",
            "12.0 mm",
            "15.0 mm",
            "20.0 mm",
        ])
        init_m_z = float(self.default_mount_z or 0.0)
        found_mz = False
        for idx in range(self.combo_mount_z.count()):
            item_text = self.combo_mount_z.itemText(idx)
            if init_m_z == 0.0 and "auto" in item_text.lower():
                self.combo_mount_z.setCurrentIndex(idx)
                found_mz = True
                break
            m = re.search(r"[-+]?\d*\.?\d+", item_text)
            if m and abs(float(m.group(0)) - init_m_z) < 1e-3:
                self.combo_mount_z.setCurrentIndex(idx)
                found_mz = True
                break
        if not found_mz:
            self.combo_mount_z.setEditText(f"{init_m_z:.1f} mm")
        row_mounts.addWidget(self.combo_mount_z, 1)

        opt_layout.addLayout(row_mounts)

        # Pocket Tolerance Clearance Row
        row_tol = QHBoxLayout()
        lbl_tol = QLabel("Pocket Tol Z:")
        lbl_tol.setStyleSheet("font-size: 11px; font-weight: 600; color: #0369a1;")
        lbl_tol.setToolTip("Axial tolerance clearance (+/- mm) for optical element mounting pockets in 3D CAD and printed enclosure.")
        row_tol.addWidget(lbl_tol)

        self.combo_tol_z = QComboBox()
        self.combo_tol_z.setEditable(True)
        self.combo_tol_z.setStyleSheet(
            "QComboBox { background-color: #f0f9ff; border: 1px solid #bae6fd; border-radius: 4px; padding: 3px 6px; font-size: 11px; color: #0369a1; }"
        )
        self.combo_tol_z.addItems([
            "±1.0 mm",
            "±0.0 mm (Exact)",
            "±0.2 mm",
            "±0.5 mm",
            "±1.5 mm",
            "±2.0 mm",
            "±3.0 mm",
            "±5.0 mm",
        ])
        init_tol = float(getattr(self, "default_tol_z", 1.0) or 1.0)
        found_tol = False
        for idx in range(self.combo_tol_z.count()):
            item_text = self.combo_tol_z.itemText(idx)
            m = re.search(r"[-+]?\d*\.?\d+", item_text)
            if m and abs(float(m.group(0)) - init_tol) < 1e-3:
                self.combo_tol_z.setCurrentIndex(idx)
                found_tol = True
                break
        if not found_tol:
            self.combo_tol_z.setEditText(f"±{init_tol:.1f} mm")
        row_tol.addWidget(self.combo_tol_z, 1)
        opt_layout.addLayout(row_tol)

        # 3D Printable Optomechanical Housing / Enclosure Row
        self.chk_enclosure = QCheckBox("Generate 3D-Printable Optomechanical Bench / Enclosure (Boolean Subtraction)")
        self.chk_enclosure.setStyleSheet(
            "QCheckBox { font-size: 11px; font-weight: bold; color: #1e40af; } "
            "QCheckBox::indicator { width: 14px; height: 14px; }"
        )
        self.chk_enclosure.setToolTip("Creates an outer mounting block and carves out optical pockets and air tunnels with zero space gaps.")
        opt_layout.addWidget(self.chk_enclosure)

        root_layout.addWidget(opt_group)

        # 5. Dialog Action Buttons
        row_actions = QHBoxLayout()
        row_actions.setSpacing(8)
        row_actions.addStretch(1)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(
            "QPushButton { background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; "
            "padding: 6px 14px; font-size: 11px; font-weight: 600; color: #475569; } "
            "QPushButton:hover { background-color: #e2e8f0; color: #1e293b; } "
            "QPushButton:pressed { background-color: #cbd5e1; }"
        )
        self.btn_cancel.clicked.connect(self.reject)
        row_actions.addWidget(self.btn_cancel)

        self.btn_export = QPushButton("📦 Export 3D STEP")
        self.btn_export.setStyleSheet(
            "QPushButton { background-color: #16a34a; border: 1px solid #15803d; border-radius: 4px; "
            "padding: 6px 18px; font-size: 12px; font-weight: bold; color: #ffffff; } "
            "QPushButton:hover { background-color: #15803d; } "
            "QPushButton:pressed { background-color: #166534; } "
            "QPushButton:disabled { background-color: #94a3b8; border-color: #64748b; color: #f1f5f9; }"
        )
        self.btn_export.clicked.connect(self._on_export_clicked)
        row_actions.addWidget(self.btn_export)

        root_layout.addLayout(row_actions)

        self.update_counter()

    def _on_table_item_changed(self, item):
        if item.column() == 0:
            row = item.row()
            if 0 <= row < len(self.elements):
                el_idx = self.elements[row]["index"]
                self._selection_map[el_idx] = (item.checkState() == Qt.Checked)
                self.update_counter()

    def _on_browse_file(self):
        if not QT_AVAILABLE:
            return
        curr = self.txt_file_path.text() if self.txt_file_path else self.default_file_path
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 3D STEP Assembly",
            curr,
            "STEP Assembly Files (*.step *.stp);;All Files (*)"
        )
        if path:
            if not (path.lower().endswith(".step") or path.lower().endswith(".stp")):
                path += ".step"
            if self.txt_file_path:
                self.txt_file_path.setText(path)
            self._chosen_path = path

    def _on_export_clicked(self):
        selected = self.get_selected_elements()
        if not selected:
            if QT_AVAILABLE:
                QMessageBox.warning(
                    self,
                    "No Elements Selected",
                    "Please select at least one optical element to include in the 3D STEP assembly."
                )
            return

        path = self.get_file_path()
        if not path:
            self._on_browse_file()
            path = self.get_file_path()
            if not path:
                return

        self._chosen_path = path
        self._chosen_engine = self.get_engine()
        self.accept()

    def update_counter(self):
        total = len(self.elements)
        num_sel = len(self.get_selected_element_indices())
        if self.lbl_counter:
            self.lbl_counter.setText(f"Selected: {num_sel} of {total} components")
            if num_sel == 0:
                self.lbl_counter.setStyleSheet("font-size: 11px; font-weight: bold; color: #dc2626; padding-right: 4px;")
            else:
                self.lbl_counter.setStyleSheet("font-size: 11px; font-weight: bold; color: #1e40af; padding-right: 4px;")
        if self.btn_export:
            self.btn_export.setEnabled(num_sel > 0)

    # -------------------------------------------------------------------------
    # Batch Selection Controls
    # -------------------------------------------------------------------------
    def select_all(self):
        for el in self.elements:
            self._selection_map[el["index"]] = True
        self._sync_table_checks()

    def deselect_all(self):
        for el in self.elements:
            self._selection_map[el["index"]] = False
        self._sync_table_checks()

    def select_by_type(self, types: Collection[str]):
        type_set = {t.lower() for t in types}
        for el in self.elements:
            el_type = el.get("type", "").lower()
            match = any(t in el_type for t in type_set)
            self._selection_map[el["index"]] = match
        self._sync_table_checks()

    def invert_selection(self):
        for el in self.elements:
            idx = el["index"]
            self._selection_map[idx] = not self._selection_map.get(idx, True)
        self._sync_table_checks()

    def set_element_selected(self, index: int, selected: bool):
        self._selection_map[index] = bool(selected)
        self._sync_table_checks()

    def _sync_table_checks(self):
        if self.table_elements and QT_AVAILABLE:
            self.table_elements.blockSignals(True)
            for row, el in enumerate(self.elements):
                is_sel = self._selection_map.get(el["index"], True)
                item = self.table_elements.item(row, 0)
                if item:
                    item.setCheckState(Qt.Checked if is_sel else Qt.Unchecked)
            self.table_elements.blockSignals(False)
        self.update_counter()

    # -------------------------------------------------------------------------
    # Accessors for Export Results
    # -------------------------------------------------------------------------
    def get_selected_elements(self) -> List[Dict[str, Any]]:
        return [el for el in self.elements if self._selection_map.get(el["index"], True)]

    def get_selected_element_indices(self) -> List[int]:
        return [el["index"] for el in self.elements if self._selection_map.get(el["index"], True)]

    def get_selected_surface_indices(self) -> List[int]:
        surfs = []
        for el in self.elements:
            if self._selection_map.get(el["index"], True):
                surfs.extend(el.get("surface_indices", []))
        return sorted(list(set(surfs)))

    def get_file_path(self) -> str:
        if self.txt_file_path:
            text = self.txt_file_path.text().strip()
            if text:
                return text
        return self._chosen_path

    def get_engine(self) -> str:
        if self.combo_engine:
            return self.combo_engine.currentData() or "auto"
        return self._chosen_engine

    def get_export_enclosure(self) -> bool:
        if self.chk_enclosure and QT_AVAILABLE:
            return bool(self.chk_enclosure.isChecked())
        return False

    def get_air_gap_constant_diameter(self) -> float:
        if self.combo_air_gap_diam and QT_AVAILABLE:
            text = self.combo_air_gap_diam.currentText().strip()
            import re
            m = re.search(r"[-+]?\d*\.?\d+", text)
            if m:
                try:
                    val = float(m.group(0))
                    if val > 0.0:
                        return val
                except ValueError:
                    pass
        return float(getattr(self, "default_air_gap_diam", 25.4) or 25.4)

    def get_export_mounts(self) -> bool:
        if self.chk_export_mounts and QT_AVAILABLE:
            return bool(self.chk_export_mounts.isChecked())
        return getattr(self, "default_export_mounts", True)

    def get_mount_diameter(self) -> float:
        if self.combo_mount_diam and QT_AVAILABLE:
            text = self.combo_mount_diam.currentText().strip()
            if "auto" in text.lower():
                return 0.0
            import re
            m = re.search(r"[-+]?\d*\.?\d+", text)
            if m:
                try:
                    val = float(m.group(0))
                    if val > 0.0:
                        return val
                except ValueError:
                    pass
        return float(getattr(self, "default_mount_diam", 30.5) or 30.5)

    def get_mount_z_thickness(self) -> float:
        if self.combo_mount_z and QT_AVAILABLE:
            text = self.combo_mount_z.currentText().strip()
            if "auto" in text.lower():
                return 0.0
            import re
            m = re.search(r"[-+]?\d*\.?\d+", text)
            if m:
                try:
                    val = float(m.group(0))
                    if val >= 0.0:
                        return val
                except ValueError:
                    pass
        return float(getattr(self, "default_mount_z", 0.0) or 0.0)

    def get_tolerance_thickness(self) -> float:
        if self.combo_tol_z and QT_AVAILABLE:
            text = self.combo_tol_z.currentText().strip()
            import re
            m = re.search(r"[-+]?\d*\.?\d+", text)
            if m:
                try:
                    val = float(m.group(0))
                    if val >= 0.0:
                        return val
                except ValueError:
                    pass
        return float(getattr(self, "default_tol_z", 1.0) or 1.0)


