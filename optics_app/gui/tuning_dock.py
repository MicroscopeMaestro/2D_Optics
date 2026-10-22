"""
Specialized Spectrometer Parameter Tuning Dock for V2 Spectrometer Design Studio.
Provides high-precision sliders and double spinboxes for:
- Grating angle (tilt_x)
- Slit width (um)
- Collimator spacing (air gap)
- Focuser lens spacing (air gap)
- Condenser lens spacing (air gap)
- Detector spacing (air gap)

Emits:
- fastParameterChanged(str, float) on slider move (<50ms fast trace dispatch)
- fullParameterCommitted(str, float) on slider release or 150ms debounce timeout
"""
from typing import Optional, Dict, Any
from v2.gui.qt_compat import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QSlider, QDoubleSpinBox, QPushButton, QTimer, Qt, pyqtSignal, Signal, QT_AVAILABLE
)
from v2.engine.lde import SurfaceStack
from v2.engine.surface import Surface, SurfaceType


class TuningDockWidget(QDockWidget):
    """
    Specialized parameter tuning dock with dual-speed fast/full event dispatch.
    """
    # Signals for dual-speed dispatching
    fastParameterChanged = Signal(str, float) if QT_AVAILABLE else pyqtSignal(str, float)
    fullParameterCommitted = Signal(str, float) if QT_AVAILABLE else pyqtSignal(str, float)

    def __init__(self, parent: Optional[QWidget] = None, stack: Optional[SurfaceStack] = None):
        super().__init__("Spectrometer Parameter Tuning", parent)
        self.stack = stack
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.timeout.connect(self._on_debounce_timeout)

        self._pending_param: Optional[str] = None
        self._pending_val: float = 0.0

        self._init_ui()

    def _init_ui(self):
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        grid = QGridLayout()
        row = 0

        # 1. Grating Angle Slider (-45.0 to +45.0 deg)
        grid.addWidget(QLabel("Grating Tilt (°):"), row, 0)
        self.grating_slider = QSlider(Qt.Horizontal)
        self.grating_slider.setRange(-450, 450)
        self.grating_slider.setValue(-120)  # -12.0 deg
        self.grating_spin = QDoubleSpinBox()
        self.grating_spin.setRange(-45.0, 45.0)
        self.grating_spin.setSingleStep(0.1)
        self.grating_spin.setValue(-12.0)
        grid.addWidget(self.grating_slider, row, 1)
        grid.addWidget(self.grating_spin, row, 2)
        row += 1

        # 2. Slit Width Slider (5 to 200 um)
        grid.addWidget(QLabel("Slit Width (µm):"), row, 0)
        self.slit_slider = QSlider(Qt.Horizontal)
        self.slit_slider.setRange(5, 200)
        self.slit_slider.setValue(30)
        self.slit_spin = QDoubleSpinBox()
        self.slit_spin.setRange(5.0, 200.0)
        self.slit_spin.setValue(30.0)
        grid.addWidget(self.slit_slider, row, 1)
        grid.addWidget(self.slit_spin, row, 2)
        row += 1

        # 3. Collimator Spacing (Surface 2 thickness, 0 to 200 mm)
        grid.addWidget(QLabel("Collimator Gap (mm):"), row, 0)
        self.collimator_slider = QSlider(Qt.Horizontal)
        self.collimator_slider.setRange(0, 2000)
        self.collimator_slider.setValue(700)  # 70.0 mm
        self.collimator_spin = QDoubleSpinBox()
        self.collimator_spin.setRange(0.0, 200.0)
        self.collimator_spin.setValue(70.0)
        grid.addWidget(self.collimator_slider, row, 1)
        grid.addWidget(self.collimator_spin, row, 2)
        row += 1

        # 4. Focuser Spacing (Surface 6 thickness, 0 to 200 mm)
        grid.addWidget(QLabel("Focuser Gap (mm):"), row, 0)
        self.focuser_slider = QSlider(Qt.Horizontal)
        self.focuser_slider.setRange(0, 2000)
        self.focuser_slider.setValue(450)  # 45.0 mm
        self.focuser_spin = QDoubleSpinBox()
        self.focuser_spin.setRange(0.0, 200.0)
        self.focuser_spin.setValue(45.0)
        grid.addWidget(self.focuser_slider, row, 1)
        grid.addWidget(self.focuser_spin, row, 2)
        row += 1

        # 5. Condenser Spacing (Surface 11 thickness, 0 to 200 mm)
        grid.addWidget(QLabel("Condenser Gap (mm):"), row, 0)
        self.condenser_slider = QSlider(Qt.Horizontal)
        self.condenser_slider.setRange(0, 2000)
        self.condenser_slider.setValue(534)  # 53.44 mm
        self.condenser_spin = QDoubleSpinBox()
        self.condenser_spin.setRange(0.0, 200.0)
        self.condenser_spin.setValue(53.44)
        grid.addWidget(self.condenser_slider, row, 1)
        grid.addWidget(self.condenser_spin, row, 2)
        row += 1

        # 6. Detector Spacing (Surface 15 thickness, 0 to 50 mm)
        grid.addWidget(QLabel("Detector Gap (mm):"), row, 0)
        self.detector_slider = QSlider(Qt.Horizontal)
        self.detector_slider.setRange(0, 500)
        self.detector_slider.setValue(100)  # 10.0 mm
        self.detector_spin = QDoubleSpinBox()
        self.detector_spin.setRange(0.0, 50.0)
        self.detector_spin.setValue(10.0)
        grid.addWidget(self.detector_slider, row, 1)
        grid.addWidget(self.detector_spin, row, 2)
        row += 1

        layout.addLayout(grid)

        # Reset button
        self.btn_reset = QPushButton("Reset to Nominal Prescription")
        self.btn_reset.clicked.connect(self.reset_to_nominal)
        layout.addWidget(self.btn_reset)

        layout.addStretch()
        self.setWidget(container)

        # Wire slider and spinbox synchronizations
        self._wire_signals()

    def _wire_signals(self):
        # Grating
        self.grating_slider.valueChanged.connect(lambda val: self._on_slider_moved("grating_tilt", val / 10.0, self.grating_spin))
        self.grating_spin.valueChanged.connect(lambda val: self._on_spin_changed("grating_tilt", val, self.grating_slider, 10.0))
        self.grating_slider.sliderReleased.connect(lambda: self._on_slider_released("grating_tilt", self.grating_spin.value()))

        # Slit
        self.slit_slider.valueChanged.connect(lambda val: self._on_slider_moved("slit_width", float(val), self.slit_spin))
        self.slit_spin.valueChanged.connect(lambda val: self._on_spin_changed("slit_width", val, self.slit_slider, 1.0))
        self.slit_slider.sliderReleased.connect(lambda: self._on_slider_released("slit_width", self.slit_spin.value()))

        # Collimator
        self.collimator_slider.valueChanged.connect(lambda val: self._on_slider_moved("collimator_gap", val / 10.0, self.collimator_spin))
        self.collimator_spin.valueChanged.connect(lambda val: self._on_spin_changed("collimator_gap", val, self.collimator_slider, 10.0))
        self.collimator_slider.sliderReleased.connect(lambda: self._on_slider_released("collimator_gap", self.collimator_spin.value()))

        # Focuser
        self.focuser_slider.valueChanged.connect(lambda val: self._on_slider_moved("focuser_gap", val / 10.0, self.focuser_spin))
        self.focuser_spin.valueChanged.connect(lambda val: self._on_spin_changed("focuser_gap", val, self.focuser_slider, 10.0))
        self.focuser_slider.sliderReleased.connect(lambda: self._on_slider_released("focuser_gap", self.focuser_spin.value()))

        # Condenser
        self.condenser_slider.valueChanged.connect(lambda val: self._on_slider_moved("condenser_gap", val / 10.0, self.condenser_spin))
        self.condenser_spin.valueChanged.connect(lambda val: self._on_spin_changed("condenser_gap", val, self.condenser_slider, 10.0))
        self.condenser_slider.sliderReleased.connect(lambda: self._on_slider_released("condenser_gap", self.condenser_spin.value()))

        # Detector
        self.detector_slider.valueChanged.connect(lambda val: self._on_slider_moved("detector_gap", val / 10.0, self.detector_spin))
        self.detector_spin.valueChanged.connect(lambda val: self._on_spin_changed("detector_gap", val, self.detector_slider, 10.0))
        self.detector_slider.sliderReleased.connect(lambda: self._on_slider_released("detector_gap", self.detector_spin.value()))

    def _on_slider_moved(self, param: str, val: float, spin: QDoubleSpinBox):
        spin.blockSignals(True)
        spin.setValue(val)
        spin.blockSignals(False)

        self._apply_param_to_stack(param, val)
        self.fastParameterChanged.emit(param, val)

        self._pending_param = param
        self._pending_val = val
        self._debounce_timer.start()

    def _on_spin_changed(self, param: str, val: float, slider: QSlider, scale: float):
        slider.blockSignals(True)
        slider.setValue(int(val * scale))
        slider.blockSignals(False)

        self._apply_param_to_stack(param, val)
        self.fastParameterChanged.emit(param, val)

        self._pending_param = param
        self._pending_val = val
        self._debounce_timer.start()

    def _on_slider_released(self, param: str, val: float):
        self._debounce_timer.stop()
        self._apply_param_to_stack(param, val)
        self.fullParameterCommitted.emit(param, val)

    def _on_debounce_timeout(self):
        if self._pending_param is not None:
            self.fullParameterCommitted.emit(self._pending_param, self._pending_val)
            self._pending_param = None

    def _apply_param_to_stack(self, param: str, val: float):
        if self.stack is None:
            return

        surfs = self.stack.surfaces if hasattr(self.stack, "surfaces") else []
        if not surfs:
            return

        if param == "grating_tilt" and len(surfs) > 3:
            surfs[3].tilt_x = val
        elif param == "collimator_gap" and len(surfs) > 2:
            surfs[2].thickness = val
        elif param == "focuser_gap" and len(surfs) > 6:
            surfs[6].thickness = val
        elif param == "condenser_gap" and len(surfs) > 11:
            surfs[11].thickness = val
        elif param == "detector_gap" and len(surfs) > 15:
            surfs[15].thickness = val

        if hasattr(self.stack, "_update_accumulated_positions"):
            self.stack._update_accumulated_positions()

    def reset_to_nominal(self):
        """Reset all parameters to nominal reference values."""
        self.grating_spin.setValue(-12.0)
        self.slit_spin.setValue(30.0)
        self.collimator_spin.setValue(70.0)
        self.focuser_spin.setValue(45.0)
        self.condenser_spin.setValue(53.44)
        self.detector_spin.setValue(10.0)
        self.fullParameterCommitted.emit("reset", 0.0)
