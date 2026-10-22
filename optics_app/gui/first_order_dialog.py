"""
Zemax OpticStudio-Style First Order Optical Data Dialog.
=======================================================
Displays paraxial first-order properties (EFFL, BFL, FFL, F/#, NA, EPD, EXPD, TOTR,
ABCD matrix) in an authentic Zemax OpticStudio report format with one-click clipboard copying.
"""

from typing import Optional, Any, Dict
import math

from optics_app.gui.qt_compat import (
    QT_AVAILABLE, QtWidgets, QtCore, QtGui,
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, Qt
)
from optics_app.engine.first_order import calculate_first_order_data, format_zemax_first_order_report


class ZemaxFirstOrderDataDialog(QDialog if QT_AVAILABLE else object):
    """
    Zemax OpticStudio First Order Optical Data Report Dialog.
    """

    def __init__(self, surface_stack: Any, config: Optional[Any] = None, parent: Optional[Any] = None):
        if QT_AVAILABLE:
            super().__init__(parent)
            self.setWindowTitle("📑 Zemax OpticStudio — First Order Optical Data")
            self.resize(650, 580)

        self.surface_stack = surface_stack
        self.config = config
        self.cached_report_text = ""

        if QT_AVAILABLE:
            self._setup_ui()
            self._calculate_and_display()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # 1. Quick KPI Cards Header
        kpi_group = QGroupBox("Key Paraxial First Order Properties", self)
        kpi_layout = QGridLayout(kpi_group)
        kpi_layout.setContentsMargins(8, 8, 8, 8)
        kpi_layout.setSpacing(6)

        self.lbl_effl = QLabel("EFFL: -- mm")
        self.lbl_effl.setStyleSheet("font-weight: bold; color: #1a73e8; font-size: 13px;")
        self.lbl_bfl = QLabel("BFL: -- mm")
        self.lbl_bfl.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.lbl_fnum = QLabel("F/#: --")
        self.lbl_fnum.setStyleSheet("font-weight: bold; color: #129eaf; font-size: 13px;")
        self.lbl_pwr = QLabel("Power: -- dpt")
        self.lbl_pwr.setStyleSheet("font-weight: bold; color: #d97706; font-size: 13px;")

        self.lbl_na = QLabel("Paraxial NA: --")
        self.lbl_na.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.lbl_epd = QLabel("EPD: -- mm")
        self.lbl_epd.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.lbl_expd = QLabel("EXPD (Back Pupil): -- mm")
        self.lbl_expd.setStyleSheet("font-weight: bold; color: #8b5cf6; font-size: 13px;")
        self.lbl_wfnum = QLabel("Working F/#: --")
        self.lbl_wfnum.setStyleSheet("font-weight: bold; font-size: 13px;")

        self.lbl_totr = QLabel("TOTR: -- mm")
        self.lbl_totr.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.lbl_p1 = QLabel("P1: -- mm")
        self.lbl_p1.setStyleSheet("font-weight: bold; color: #2563eb; font-size: 13px;")
        self.lbl_p2 = QLabel("P2: -- mm")
        self.lbl_p2.setStyleSheet("font-weight: bold; color: #dc2626; font-size: 13px;")
        self.lbl_pupil_dist = QLabel("ExP to Image: -- mm")
        self.lbl_pupil_dist.setStyleSheet("font-weight: bold; font-size: 13px;")

        kpi_layout.addWidget(self.lbl_effl, 0, 0)
        kpi_layout.addWidget(self.lbl_bfl, 0, 1)
        kpi_layout.addWidget(self.lbl_fnum, 0, 2)
        kpi_layout.addWidget(self.lbl_pwr, 0, 3)

        kpi_layout.addWidget(self.lbl_na, 1, 0)
        kpi_layout.addWidget(self.lbl_epd, 1, 1)
        kpi_layout.addWidget(self.lbl_expd, 1, 2)
        kpi_layout.addWidget(self.lbl_wfnum, 1, 3)

        kpi_layout.addWidget(self.lbl_totr, 2, 0)
        kpi_layout.addWidget(self.lbl_p1, 2, 1)
        kpi_layout.addWidget(self.lbl_p2, 2, 2)
        kpi_layout.addWidget(self.lbl_pupil_dist, 2, 3)

        layout.addWidget(kpi_group)

        # 2. Monospace Prescription Data Report
        self.text_report = QtWidgets.QTextEdit(self)
        self.text_report.setReadOnly(True)
        mono_font = QtGui.QFont("Courier New", 10)
        mono_font.setStyleHint(QtGui.QFont.Monospace)
        self.text_report.setFont(mono_font)
        self.text_report.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "border: 1px solid #3c3c3c; border-radius: 4px; padding: 6px; }"
        )
        layout.addWidget(self.text_report, 1)

        # 3. Action Buttons
        btn_layout = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Refresh", self)
        btn_refresh.setToolTip("Recalculate first-order optical data")
        btn_refresh.clicked.connect(self._calculate_and_display)
        btn_layout.addWidget(btn_refresh)

        btn_copy = QPushButton("📋 Copy Report to Clipboard", self)
        btn_copy.setToolTip("Copy the complete Zemax first-order optical data report to system clipboard")
        btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_layout.addWidget(btn_copy)

        btn_layout.addStretch(1)

        btn_close = QPushButton("Close", self)
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def _calculate_and_display(self):
        surfaces = []
        if hasattr(self.surface_stack, "surfaces"):
            surfaces = self.surface_stack.surfaces
        elif isinstance(self.surface_stack, list):
            surfaces = self.surface_stack

        wl_um = 0.5876
        if self.config and hasattr(self.config, "primary_wavelength_um"):
            wl_um = float(self.config.primary_wavelength_um)

        data = calculate_first_order_data(surfaces, wavelength_um=wl_um, config=self.config)

        effl = data.get("effl", float("inf"))
        bfl = data.get("bfl", float("inf"))
        f_num = data.get("f_number", float("inf"))
        na = data.get("na", 0.0)
        epd = data.get("epd", 0.0)
        expd = data.get("expd", 0.0)
        p1 = data.get("p1", 0.0)
        p2 = data.get("p2", 0.0)
        pwr_dpt = data.get("optical_power_diopters", 0.0)
        wf_num = data.get("working_f_number", float("inf"))
        exp_to_img = data.get("exp_to_image", 0.0)
        totr = data.get("totr", 0.0)
        is_afocal = data.get("is_afocal", False)

        if math.isinf(effl) or is_afocal:
            self.lbl_effl.setText("EFFL: Afocal")
            self.lbl_bfl.setText("BFL: Afocal")
            self.lbl_fnum.setText("F/#: Afocal")
            self.lbl_pwr.setText("Power: 0.00 dpt")
        else:
            self.lbl_effl.setText(f"EFFL: {effl:.3f} mm")
            self.lbl_bfl.setText(f"BFL: {bfl:.3f} mm")
            self.lbl_fnum.setText(f"F/#: {f_num:.2f}")
            self.lbl_pwr.setText(f"Power: {pwr_dpt:+.2f} dpt")

        self.lbl_na.setText(f"Paraxial NA: {na:.4f}")
        self.lbl_epd.setText(f"EPD: {epd:.2f} mm")
        self.lbl_expd.setText(f"EXPD (Back Pupil): {expd:.2f} mm")
        self.lbl_wfnum.setText(f"Working F/#: {wf_num:.2f}" if not math.isinf(wf_num) else "Working F/#: Afocal")
        self.lbl_totr.setText(f"TOTR: {totr:.2f} mm")
        self.lbl_p1.setText(f"P1: {p1:+.2f} mm")
        self.lbl_p2.setText(f"P2: {p2:+.2f} mm")
        self.lbl_pupil_dist.setText(f"ExP to Image: {exp_to_img:.2f} mm")

        title = "2D Optics System"
        report = format_zemax_first_order_report(data, title=title)
        self.cached_report_text = report
        self.text_report.setPlainText(report)

    def _copy_to_clipboard(self):
        if not QT_AVAILABLE:
            return
        cb = QtWidgets.QApplication.clipboard()
        if cb:
            cb.setText(self.cached_report_text)
            parent = self.parent()
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage("Copied Zemax First Order Data to clipboard!", 3000)
