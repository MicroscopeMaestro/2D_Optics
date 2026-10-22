"""
Real-Time PyQtGraph & Canvas Visualizers for V2 Spectrometer Design Studio.
Provides:
1. SystemLayoutPlot / SystemLayoutWidget: 2D longitudinal ray fan view (Z-Y), optical surfaces, multi-wavelength ray paths.
2. SpectrumSpreadPlot / SpectrumSpreadWidget: Linear dispersion curve dx(lambda), NIR band distribution.
3. SpotDiagramPlot / SpotDiagramWidget: 0.8mm detector aperture circle, 2D ray hit coordinates, centroid, RMS radius, Airy disk circle.
4. DualSpeedVisualizerWidget: Synchronized interactive visualizer dock.
"""
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np

from v2.gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QCheckBox,
    pyqtSignal, Signal, QT_AVAILABLE, QTabWidget, QSplitter
)
from v2.engine.models import FastTraceResult, FullTraceResult, SpectrometerSystem
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, DualSpeedTracer

# Try importing pyqtgraph
HAS_PYQTGRAPH = False
try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class SystemLayoutPlot(QWidget):
    """
    2D Longitudinal Ray Fan & Optical Layout Visualizer (Z-Y plane).
    Renders optical surfaces, clear apertures, mirror reflections, grating diffraction,
    and multi-wavelength ray fans in real time (<50ms fast updates).
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        if HAS_PYQTGRAPH and QT_AVAILABLE:
            self.plot_widget = pg.PlotWidget(title="Spectrometer Optical Layout (Z-Y Plane)")
            self.plot_widget.setLabel("left", "Y Height (mm)")
            self.plot_widget.setLabel("bottom", "Z Global Axis (mm)")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setAspectLocked(True)
            self.layout.addWidget(self.plot_widget)
        else:
            self.plot_widget = None
            self.placeholder_label = QLabel("System Layout Plot (2D Longitudinal Z-Y)")
            self.layout.addWidget(self.placeholder_label)

        self._surface_curves = []
        self._ray_curves = []
        self._wavelength_colors = {
            1.100: (220, 20, 60),    # Crimson
            1.225: (34, 139, 34),    # Forest Green (Primary)
            1.350: (30, 144, 255)    # Dodger Blue
        }

    def update_layout(self, surface_stack: SurfaceStack, trace_result: Optional[Union[FastTraceResult, FullTraceResult]] = None):
        """Redraw optical surfaces and ray paths."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return

        self.plot_widget.clear()

        # 1. Draw Optical Surfaces
        surfaces = surface_stack.surfaces if hasattr(surface_stack, "surfaces") else []
        for i, surf in enumerate(surfaces):
            origin = getattr(surf, "global_origin", np.zeros(3))
            R_mat = getattr(surf, "global_R", np.eye(3))
            sd = float(getattr(surf, "semi_diameter", 12.5))
            if sd <= 0:
                sd = 12.5

            # Local surface profile points
            num_pts = 31
            y_loc = np.linspace(-sd, sd, num_pts)
            x_loc = np.zeros_like(y_loc)
            rad = float(getattr(surf, "radius", 0.0))
            conic = float(getattr(surf, "conic", 0.0))
            asph_coeffs = getattr(surf, "asphere_coeffs", [])

            if rad != 0.0:
                c = 1.0 / rad
                r2 = y_loc**2
                denom = 1.0 + np.sqrt(np.maximum(0.0, 1.0 - (1.0 + conic) * (c**2) * r2))
                z_loc = (c * r2) / np.maximum(1e-12, denom)
                if asph_coeffs:
                    for p_idx, coeff in enumerate(asph_coeffs):
                        z_loc += float(coeff) * (r2 ** (p_idx + 1))
            else:
                z_loc = np.zeros_like(y_loc)

            # Transform to global coordinates
            pts_loc = np.vstack([x_loc, y_loc, z_loc])  # 3 x N
            pts_glob = origin[:, None] + R_mat @ pts_loc

            z_glob = pts_glob[2, :]
            y_glob = pts_glob[1, :]

            # Styling by surface type
            stype = getattr(surf, "surface_type", SurfaceType.STANDARD)
            glass = str(getattr(surf, "glass", "")).upper()

            if stype == SurfaceType.GRATING:
                pen = pg.mkPen(color=(255, 140, 0), width=2.5, style=pg.QtCore.Qt.SolidLine)
            elif stype == SurfaceType.DETECTOR:
                pen = pg.mkPen(color=(255, 0, 0), width=3.0, style=pg.QtCore.Qt.SolidLine)
            elif "MIRROR" in glass or stype == SurfaceType.MIRROR:
                pen = pg.mkPen(color=(180, 180, 180), width=2.5, style=pg.QtCore.Qt.SolidLine)
            elif stype == SurfaceType.ASPHERIC:
                pen = pg.mkPen(color=(148, 0, 211), width=2.0, style=pg.QtCore.Qt.SolidLine)
            else:
                pen = pg.mkPen(color=(70, 130, 180), width=1.5, style=pg.QtCore.Qt.SolidLine)

            self.plot_widget.plot(z_glob, y_glob, pen=pen)

        # 2. Draw Ray Paths
        if trace_result is not None and hasattr(trace_result, "ray_paths"):
            for path in trace_result.ray_paths:
                if len(path) >= 2:
                    z_pts = path[:, 2]
                    y_pts = path[:, 1]
                    pen = pg.mkPen(color=(0, 200, 100, 150), width=1.0)
                    self.plot_widget.plot(z_pts, y_pts, pen=pen)


class SpectrumSpreadPlot(QWidget):
    """
    Linear Dispersion & Spectral Band Spread Visualizer.
    Plots spatial detector position x(lambda) versus wavelength lambda across the NIR band.
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        if HAS_PYQTGRAPH and QT_AVAILABLE:
            self.plot_widget = pg.PlotWidget(title="Linear Dispersion Spectrum Spread")
            self.plot_widget.setLabel("left", "Detector X Position (mm)")
            self.plot_widget.setLabel("bottom", "Wavelength (µm)")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.layout.addWidget(self.plot_widget)
        else:
            self.plot_widget = None
            self.placeholder_label = QLabel("Spectrum Spread Plot (x vs lambda)")
            self.layout.addWidget(self.placeholder_label)

    def update_spectrum(self, trace_result: Union[FastTraceResult, FullTraceResult]):
        """Plot dispersion curve from trace result."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return

        self.plot_widget.clear()

        wavelengths = []
        x_centroids = []

        if hasattr(trace_result, "spot_diagrams_by_wl") and trace_result.spot_diagrams_by_wl:
            for wl, d in sorted(trace_result.spot_diagrams_by_wl.items()):
                wavelengths.append(wl)
                x_centroids.append(d.get("centroid_x", 0.0))
        elif hasattr(trace_result, "spot_coordinates") and trace_result.spot_coordinates:
            for wl, pts in sorted(trace_result.spot_coordinates.items()):
                if len(pts) > 0:
                    wavelengths.append(wl)
                    x_centroids.append(float(np.mean(pts[:, 0])))

        if len(wavelengths) >= 2:
            pen = pg.mkPen(color=(255, 165, 0), width=2.0)
            self.plot_widget.plot(wavelengths, x_centroids, pen=pen, symbol='o', symbolSize=8, symbolBrush=(255, 69, 0))


class SpotDiagramPlot(QWidget):
    """
    0.8mm Single-Pixel Detector Spot Diagram & Aberration Canvas.
    Plots 2D ray hits, 0.8mm detector clear aperture circle, centroid, RMS radius, and Airy disk.
    """
    def __init__(self, parent: Optional[QWidget] = None, detector_diameter_mm: float = 0.8):
        super().__init__(parent)
        self.detector_diameter = detector_diameter_mm
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        if HAS_PYQTGRAPH and QT_AVAILABLE:
            self.plot_widget = pg.PlotWidget(title=f"Detector Spot Diagram ({detector_diameter_mm:.1f} mm Aperture)")
            self.plot_widget.setLabel("left", "Y (mm)")
            self.plot_widget.setLabel("bottom", "X (mm)")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setAspectLocked(True)
            self.layout.addWidget(self.plot_widget)
        else:
            self.plot_widget = None
            self.placeholder_label = QLabel(f"Spot Diagram ({detector_diameter_mm:.1f} mm Detector)")
            self.layout.addWidget(self.placeholder_label)

    def update_spot(self, trace_result: Union[FastTraceResult, FullTraceResult]):
        """Plot ray hits and reference aperture circle."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return

        self.plot_widget.clear()

        # 1. Draw 0.8mm Detector Aperture Circle
        r_det = self.detector_diameter / 2.0
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = r_det * np.cos(theta)
        y_circle = r_det * np.sin(theta)
        circle_pen = pg.mkPen(color=(200, 50, 50), width=2.0, style=pg.QtCore.Qt.DashLine)
        self.plot_widget.plot(x_circle, y_circle, pen=circle_pen)

        # 2. Draw Airy Disk if available
        if hasattr(trace_result, "airy_radius") and trace_result.airy_radius > 0:
            r_airy = trace_result.airy_radius
            x_airy = r_airy * np.cos(theta)
            y_airy = r_airy * np.sin(theta)
            airy_pen = pg.mkPen(color=(50, 150, 255), width=1.5, style=pg.QtCore.Qt.DotLine)
            self.plot_widget.plot(x_airy, y_airy, pen=airy_pen)

        # 3. Plot Ray Hits per Wavelength
        color_map = {
            1.100: (220, 20, 60),
            1.225: (34, 139, 34),
            1.350: (30, 144, 255)
        }

        if hasattr(trace_result, "spot_coordinates"):
            for wl, pts in trace_result.spot_coordinates.items():
                if len(pts) > 0:
                    col = color_map.get(round(wl, 3), (150, 150, 150))
                    self.plot_widget.plot(
                        pts[:, 0], pts[:, 1],
                        pen=None,
                        symbol='+',
                        symbolSize=7,
                        symbolPen=pg.mkPen(color=col, width=1.2)
                    )


class DualSpeedVisualizerWidget(QWidget):
    """
    Combined Real-Time Visualization Suite embedding SystemLayoutPlot,
    SpectrumSpreadPlot, and SpotDiagramPlot.
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget(self)
        self.layout_plot = SystemLayoutPlot(self)
        self.spectrum_plot = SpectrumSpreadPlot(self)
        self.spot_plot = SpotDiagramPlot(self, detector_diameter_mm=0.8)

        self.tab_widget.addTab(self.layout_plot, "2D Layout")
        self.tab_widget.addTab(self.spectrum_plot, "Spectrum Spread")
        self.tab_widget.addTab(self.spot_plot, "Spot Diagram")

        self.layout.addWidget(self.tab_widget)

    def refresh_fast(self, surface_stack: SurfaceStack, fast_result: FastTraceResult):
        """Update fast visualizer canvases (<50ms)."""
        self.layout_plot.update_layout(surface_stack, fast_result)
        self.spot_plot.update_spot(fast_result)

    def refresh_full(self, surface_stack: SurfaceStack, full_result: FullTraceResult):
        """Update full visualizer canvases on release/debounce."""
        self.layout_plot.update_layout(surface_stack, full_result)
        self.spectrum_plot.update_spectrum(full_result)
        self.spot_plot.update_spot(full_result)
