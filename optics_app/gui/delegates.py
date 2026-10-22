"""
Custom QStyledItemDelegate Implementations for Lens Data Editor.
Delegates: SurfaceTypeDelegate, GlassSelectionDelegate, NumericValueDelegate, CADFilePathDelegate, ApertureTypeDelegate, ApertureParamDelegate.
Aliases: ComboBoxDelegate, NumericSpinBoxDelegate, CADFilePickerDelegate.
"""
from typing import Any
from optics_app.gui.qt_compat import (
    QStyledItemDelegate, QComboBox, QDoubleSpinBox, QLineEdit,
    QPushButton, QHBoxLayout, QWidget, QFileDialog, QCompleter,
    Qt, QT_AVAILABLE
)
from optics_app.utils.materials import GlassCatalog
from optics_app.engine.surface import SurfaceType


class SurfaceTypeDelegate(QStyledItemDelegate if QT_AVAILABLE else object):
    """Delegate for Surface Type column using a dropdown QComboBox."""
    TYPES = [
        "Standard",
        "Coordinate Break",
        "Even Asphere",
        "Diffraction Grating",
        "Mirror",
        "Stop",
        "Detector",
        "CAD",
        "Toroidal",
        "Object",
        "Air Space"
    ]

    def __init__(self, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.TYPES)
        return combo

    def setEditorData(self, editor, index):
        if editor is None:
            return
        val = "Standard"
        if hasattr(index, "data"):
            val = str(index.data(Qt.EditRole) or index.data(Qt.DisplayRole) or "Standard")
        norm = SurfaceType.normalize(val)
        matched_idx = -1
        for i, t in enumerate(self.TYPES):
            if SurfaceType.normalize(t).lower() == norm.lower() or t.lower() == val.lower():
                matched_idx = i
                break
        if matched_idx >= 0:
            if hasattr(editor, "setCurrentIndex"):
                editor.setCurrentIndex(matched_idx)
            elif hasattr(editor, "setCurrentText"):
                editor.setCurrentText(self.TYPES[matched_idx])
        elif hasattr(editor, "setCurrentText"):
            editor.setCurrentText(val)

    def setModelData(self, editor, model, index):
        if editor is None or model is None:
            return
        text = editor.currentText() if hasattr(editor, "currentText") else "Standard"
        norm = SurfaceType.normalize(text)
        model.setData(index, norm, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        if editor is not None and hasattr(editor, "setGeometry") and hasattr(option, "rect"):
            editor.setGeometry(option.rect)


class GlassSelectionDelegate(QStyledItemDelegate if QT_AVAILABLE else object):
    """Delegate for Glass Selection column using an editable searchable QComboBox."""

    COMMON_GLASSES = [
        "AIR", "MIRROR", "N-BK7", "FUSED_SILICA", "N-SF11", "N-SK16",
        "N-BAF10", "N-SF6", "N-SF6HT", "N-LAK22", "N-LASF9", "N-FK51A",
        "N-SF1", "N-SF2", "N-SF4", "N-SF5", "N-SF8", "N-SF10", "N-SSK8",
        "N-KZFS11", "F2", "SF2", "SF4", "SF5", "SF10", "SF11", "B270",
        "CAF2", "ZNSE", "GERMANIUM", "SILICON",
        "S-LAH64", "S-NPH1", "S-NPH2", "S-NPH3", "S-NPH11",
        "AL", "AU", "AG", "CU"
    ]

    def __init__(self, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)
        seen = set()
        self.glasses = []
        for g in self.COMMON_GLASSES:
            if g.upper() not in seen:
                self.glasses.append(g)
                seen.add(g.upper())
        for g in sorted(GlassCatalog.GLASSES.keys()):
            if g.upper() not in seen:
                self.glasses.append(g)
                seen.add(g.upper())

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.setEditable(True)
        combo.addItems(self.glasses)
        completer = QCompleter(self.glasses, combo)
        try:
            completer.setCaseSensitivity(Qt.CaseInsensitive if hasattr(Qt, 'CaseInsensitive') else 0)
        except Exception:
            pass
        combo.setCompleter(completer)
        return combo

    def setEditorData(self, editor, index):
        if editor is None:
            return
        val = "AIR"
        if hasattr(index, "data"):
            val = str(index.data(Qt.EditRole) or index.data(Qt.DisplayRole) or "AIR")
        if hasattr(editor, "setCurrentText"):
            editor.setCurrentText(val)

    def setModelData(self, editor, model, index):
        if editor is None or model is None:
            return
        val = editor.currentText().strip().upper() if hasattr(editor, "currentText") else "AIR"
        model.setData(index, val, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        if editor is not None and hasattr(editor, "setGeometry") and hasattr(option, "rect"):
            editor.setGeometry(option.rect)


class NumericValueDelegate(QStyledItemDelegate if QT_AVAILABLE else object):
    """Delegate for numerical columns (Radius, Thickness, Semi-Diameter, Decenters, Tilts)."""

    def __init__(self, parent=None, min_val=-1e6, max_val=1e6, decimals=4, step=1.0):
        if QT_AVAILABLE:
            super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
        self.step = step

    def createEditor(self, parent, option, index):
        spin = QDoubleSpinBox(parent)
        spin.setRange(self.min_val, self.max_val)
        spin.setDecimals(self.decimals)
        spin.setSingleStep(self.step)
        return spin

    def setEditorData(self, editor, index):
        if editor is None:
            return
        raw_val = 0.0
        if hasattr(index, "data"):
            raw_val = index.data(Qt.EditRole)
            if raw_val is None:
                raw_val = index.data(Qt.DisplayRole)
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            val = 0.0
        if hasattr(editor, "setValue"):
            editor.setValue(val)

    def setModelData(self, editor, model, index):
        if editor is None or model is None:
            return
        val = editor.value() if hasattr(editor, "value") else 0.0
        model.setData(index, val, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        if editor is not None and hasattr(editor, "setGeometry") and hasattr(option, "rect"):
            editor.setGeometry(option.rect)


class CADFilePathDelegate(QStyledItemDelegate if QT_AVAILABLE else object):
    """Delegate for CAD File Path column featuring line edit and file open dialog browse button."""

    def __init__(self, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)

    def createEditor(self, parent, option, index):
        widget = QWidget(parent)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        line_edit = QLineEdit("", widget)
        btn_browse = QPushButton("...", widget)

        layout.addWidget(line_edit)
        layout.addWidget(btn_browse)
        widget.setLayout(layout)

        widget.line_edit = line_edit
        widget.btn_browse = btn_browse

        def on_browse():
            file_path, _ = QFileDialog.getOpenFileName(
                parent, "Select CAD File", "",
                "CAD Files (*.step *.stp *.sldprt *.sldasm);;All Files (*)"
            )
            if file_path:
                line_edit.setText(file_path)

        if hasattr(btn_browse, 'clicked'):
            if hasattr(btn_browse.clicked, "connect"):
                btn_browse.clicked.connect(on_browse)

        return widget

    def setEditorData(self, editor, index):
        if editor is None or not hasattr(editor, 'line_edit'):
            return
        val = ""
        if hasattr(index, "data"):
            val = str(index.data(Qt.EditRole) or index.data(Qt.DisplayRole) or "")
        editor.line_edit.setText(val)

    def setModelData(self, editor, model, index):
        if editor is None or model is None or not hasattr(editor, 'line_edit'):
            return
        model.setData(index, editor.line_edit.text(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        if editor is not None and hasattr(editor, "setGeometry") and hasattr(option, "rect"):
            editor.setGeometry(option.rect)


class ApertureTypeDelegate(QStyledItemDelegate if QT_AVAILABLE else object):
    """Delegate for Aperture Type column using a dropdown QComboBox."""
    TYPES = ["Circular", "Rectangular", "Elliptical", "Annular", "Spider"]

    def __init__(self, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.TYPES)
        return combo

    def setEditorData(self, editor, index):
        if editor is None:
            return
        val = "Circular"
        if hasattr(index, "data"):
            val = str(index.data(Qt.EditRole) or index.data(Qt.DisplayRole) or "Circular")
        for i, t in enumerate(self.TYPES):
            if t.lower() == val.lower():
                if hasattr(editor, "setCurrentText"):
                    editor.setCurrentText(t)
                elif hasattr(editor, "setCurrentIndex"):
                    editor.setCurrentIndex(i)
                break

    def setModelData(self, editor, model, index):
        if editor is None or model is None:
            return
        text = editor.currentText() if hasattr(editor, "currentText") else "Circular"
        model.setData(index, text, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        if editor is not None and hasattr(editor, "setGeometry") and hasattr(option, "rect"):
            editor.setGeometry(option.rect)


class ApertureParamDelegate(QStyledItemDelegate if QT_AVAILABLE else object):
    """Delegate for Aperture Parameters column using QLineEdit."""

    def __init__(self, parent=None):
        if QT_AVAILABLE:
            super().__init__(parent)

    def createEditor(self, parent, option, index):
        line = QLineEdit(parent)
        return line

    def setEditorData(self, editor, index):
        if editor is None:
            return
        val = ""
        if hasattr(index, "data"):
            val = str(index.data(Qt.EditRole) or index.data(Qt.DisplayRole) or "")
        if hasattr(editor, "setText"):
            editor.setText(val)

    def setModelData(self, editor, model, index):
        if editor is None or model is None:
            return
        text = editor.text() if hasattr(editor, "text") else ""
        model.setData(index, text, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        if editor is not None and hasattr(editor, "setGeometry") and hasattr(option, "rect"):
            editor.setGeometry(option.rect)


# Aliases for specification compliance
ComboBoxDelegate = SurfaceTypeDelegate
NumericSpinBoxDelegate = NumericValueDelegate
CADFilePickerDelegate = CADFilePathDelegate
