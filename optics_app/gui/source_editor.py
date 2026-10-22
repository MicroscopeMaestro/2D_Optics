from optics_app.gui.qt_compat import (
    QWidget, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton, 
    QHBoxLayout, QVBoxLayout, QLabel, QFormLayout, QSpinBox, Signal
)

from optics_app.engine.source import SourceConfig, SourceType

class SourceEditorWidget(QWidget):
    config_changed = Signal()

    def __init__(self, config: SourceConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title_label = QLabel("<b>Source Profile & Input Aperture</b>")
        layout.addWidget(title_label)

        self.form_layout = QFormLayout()
        
        # Source Type
        self.type_combo = QComboBox()
        for t in SourceType:
            self.type_combo.addItem(t.value, t)
        self.type_combo.setCurrentText(self.config.source_type.value)
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.form_layout.addRow("Source Type:", self.type_combo)

        # Input Aperture Shape
        self.aperture_shape_combo = QComboBox()
        self.aperture_shape_combo.addItems(["Circular (Fiber Core)", "Rectangular (Entrance Slit)"])
        if "Rectangular" in getattr(self.config, 'input_aperture_shape', 'Circular') or getattr(self.config, 'source_type', None) == SourceType.SLIT:
            self.aperture_shape_combo.setCurrentIndex(1)
        else:
            self.aperture_shape_combo.setCurrentIndex(0)
        self.aperture_shape_combo.currentIndexChanged.connect(self.on_param_changed)
        self.form_layout.addRow("Input Aperture Shape:", self.aperture_shape_combo)

        # Input Aperture Width / Diameter (um)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(1.0, 100000.0)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setSingleStep(5.0)
        self.radius_spin.setValue(self.config.input_aperture_width * 1000.0)
        self.radius_spin.setSuffix(" μm")
        self.radius_spin.valueChanged.connect(self.on_param_changed)
        self.form_layout.addRow("Input Aperture Width / Core Ø:", self.radius_spin)

        # Input Slit Height (mm)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.01, 20.0)
        self.height_spin.setDecimals(2)
        self.height_spin.setSingleStep(0.1)
        self.height_spin.setValue(getattr(self.config, 'input_aperture_height', 1.0))
        self.height_spin.setSuffix(" mm")
        self.height_spin.valueChanged.connect(self.on_param_changed)
        self.form_layout.addRow("Input Slit Height:", self.height_spin)

        # Numerical Aperture (NA)
        self.na_spin = QDoubleSpinBox()
        self.na_spin.setRange(0.01, 0.99)
        self.na_spin.setDecimals(3)
        self.na_spin.setSingleStep(0.01)
        self.na_spin.setValue(getattr(self.config, 'numerical_aperture', 0.12))
        self.na_spin.valueChanged.connect(self.on_na_changed)
        self.form_layout.addRow("Numerical Aperture (NA):", self.na_spin)

        # Divergence Half-Angle (deg)
        self.divergence_spin = QDoubleSpinBox()
        self.divergence_spin.setRange(0.001, 90.0)
        self.divergence_spin.setDecimals(2)
        self.divergence_spin.setValue(self.config.divergence_angle)
        self.divergence_spin.setSuffix(" deg")
        self.divergence_spin.valueChanged.connect(self.on_param_changed)
        self.form_layout.addRow("Divergence Half-Angle:", self.divergence_spin)

        # Num Rays
        self.rays_spin = QSpinBox()
        self.rays_spin.setRange(1, 1000)
        self.rays_spin.setValue(self.config.num_rays)
        self.rays_spin.valueChanged.connect(self.on_param_changed)
        self.form_layout.addRow("Num Rays (1D grid):", self.rays_spin)

        # Laser Mode (CW or Ultrafast)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["CW", "Ultrafast"])
        self.mode_combo.setCurrentText(getattr(self.config, 'laser_mode', "CW"))
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.form_layout.addRow("Laser Mode:", self.mode_combo)

        # Pulse Duration (fs)
        self.pulse_spin = QDoubleSpinBox()
        self.pulse_spin.setRange(1.0, 1000000.0)
        self.pulse_spin.setDecimals(1)
        self.pulse_spin.setValue(getattr(self.config, 'pulse_duration_fs', 100.0))
        self.pulse_spin.setSuffix(" fs")
        self.pulse_spin.valueChanged.connect(self.on_param_changed)
        self.form_layout.addRow("Pulse Duration:", self.pulse_spin)

        # Bandwidth (nm)
        self.bandwidth_spin = QDoubleSpinBox()
        self.bandwidth_spin.setRange(0.01, 1000.0)
        self.bandwidth_spin.setDecimals(2)
        self.bandwidth_spin.setValue(getattr(self.config, 'bandwidth_nm', 10.0))
        self.bandwidth_spin.setSuffix(" nm")
        self.bandwidth_spin.valueChanged.connect(self.on_param_changed)
        self.form_layout.addRow("Bandwidth (FWHM):", self.bandwidth_spin)

        layout.addLayout(self.form_layout)
        layout.addStretch()

        self.update_visibility()

    def on_mode_changed(self, index):
        self.config.laser_mode = self.mode_combo.currentText()
        self.update_visibility()
        self.config_changed.emit()

    def on_type_changed(self, index):
        self.config.source_type = self.type_combo.currentData()
        self.update_visibility()
        self.config_changed.emit()

    def on_na_changed(self):
        na = self.na_spin.value()
        self.config.set_na(na)
        self.divergence_spin.blockSignals(True)
        self.divergence_spin.setValue(self.config.divergence_angle)
        self.divergence_spin.blockSignals(False)
        self.config_changed.emit()

    def on_param_changed(self):
        shape_idx = self.aperture_shape_combo.currentIndex()
        self.config.input_aperture_shape = "Circular" if shape_idx == 0 else "Rectangular"
        self.config.input_aperture_width = self.radius_spin.value() / 1000.0
        self.config.beam_radius = self.config.input_aperture_width / 2.0
        self.config.input_aperture_height = self.height_spin.value()
        self.config.divergence_angle = self.divergence_spin.value()
        self.config.num_rays = self.rays_spin.value()
        
        self.config.laser_mode = self.mode_combo.currentText()
        self.config.pulse_duration_fs = self.pulse_spin.value()
        self.config.bandwidth_nm = self.bandwidth_spin.value()
        
        self.update_visibility()
        self.config_changed.emit()

    def set_source_config(self, config: SourceConfig):
        self.config = config
        self.update_ui()

    def update_ui(self):
        # Update the UI components from the underlying config
        self.blockSignals(True)
        self.type_combo.setCurrentText(self.config.source_type.value)
        if 'Rectangular' in getattr(self.config, 'input_aperture_shape', 'Circular') or getattr(self.config, 'source_type', None) == SourceType.SLIT:
            self.aperture_shape_combo.setCurrentIndex(1)
        else:
            self.aperture_shape_combo.setCurrentIndex(0)
        self.radius_spin.setValue(self.config.input_aperture_width * 1000.0)
        if hasattr(self, 'height_spin'):
            self.height_spin.setValue(self.config.input_aperture_height)
        self.na_spin.setValue(self.config.numerical_aperture)
        self.divergence_spin.setValue(self.config.divergence_angle)
        self.rays_spin.setValue(self.config.num_rays)
        if hasattr(self, 'wavelengths_input'):
            self.wavelengths_input.setText(self.config.wavelengths)
            
        self.mode_combo.setCurrentText(getattr(self.config, 'laser_mode', "CW"))
        self.pulse_spin.setValue(getattr(self.config, 'pulse_duration_fs', 100.0))
        self.bandwidth_spin.setValue(getattr(self.config, 'bandwidth_nm', 10.0))
            
        self.update_visibility()
        self.blockSignals(False)

    def update_visibility(self):
        stype = self.config.source_type
        is_rect = self.aperture_shape_combo.currentIndex() == 1
        
        show_radius = stype in (SourceType.COLLIMATED, SourceType.GAUSSIAN, SourceType.FIBER, SourceType.SLIT)
        show_div = stype in (SourceType.POINT, SourceType.GAUSSIAN, SourceType.FIBER, SourceType.SLIT)

        self.radius_spin.setVisible(show_radius)
        self.height_spin.setVisible(show_radius and is_rect)
        self.na_spin.setVisible(show_div)
        self.divergence_spin.setVisible(show_div)
        self.aperture_shape_combo.setVisible(show_radius)
        
        is_ultrafast = self.mode_combo.currentText() == "Ultrafast"
        self.pulse_spin.setVisible(is_ultrafast)
        self.bandwidth_spin.setVisible(is_ultrafast)
