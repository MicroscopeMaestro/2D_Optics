"""Wavelength-resolved grating rays in the Raman spectrometer's 3D view."""

from pathlib import Path

import numpy as np
import pytest

from optics_app.engine.ray import Ray
from optics_app.engine.tracer import OpticalSystem
from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
from optics_app.gui.visualizer_3d import (
    HAS_MPL_QT,
    Visualizer3DWidget,
    collect_grating_dispersion_samples,
    diffraction_wavelength_colors,
)


MODEL = Path(__file__).resolve().parents[1] / "spectrometer_raman_50mm_100mm_manual.json"
WAVELENGTHS = (1.087, 1.225, 1.350)


def _traced_chiefs():
    system = OpticalSystem.from_json(str(MODEL))
    rays = [system.trace_ray(Ray(pos=(0, 0, -1), dir=(0, 0, 1), wavelength=wl))
            for wl in WAVELENGTHS]
    return system, rays


def test_grating_3d_samples_follow_traced_diffraction_equation():
    system, rays = _traced_chiefs()
    samples = collect_grating_dispersion_samples(system.surfaces, rays)
    assert [sample["wavelength_um"] for sample in samples] == list(WAVELENGTHS)
    assert all(sample["surface_index"] == 15 for sample in samples)
    assert all(sample["order"] == 1 for sample in samples)
    assert all(sample["lines_per_mm"] == 600 for sample in samples)
    assert all(sample["incident_angle_deg"] == pytest.approx(42.0, abs=1e-8)
               for sample in samples)

    # In the saved model the chief ray strikes the grating at 42 degrees.
    for sample in samples:
        beta = np.radians(sample["angle_deg"])
        assert np.sin(np.radians(42.0)) + np.sin(beta) == pytest.approx(
            sample["wavelength_um"] * 600.0 / 1000.0, abs=1e-7
        )
    assert [round(sample["angle_deg"], 2) for sample in samples] == [-0.97, 3.78, 8.10]
    assert len(set(diffraction_wavelength_colors(WAVELENGTHS).values())) == 3
    assert all(np.array_equal(ray.launch_pos, [0, 0, -1]) for ray in rays)


@pytest.mark.skipif(not (QT_AVAILABLE and HAS_MPL_QT), reason="Qt Matplotlib unavailable")
def test_3d_view_draws_distinct_wavelength_fan_without_false_return_segment():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    system, rays = _traced_chiefs()
    widget = Visualizer3DWidget(auto_refresh=False)
    try:
        widget.update_layout(surface_stack=system.stack, rays=rays)
        labels = [item.get_text() for item in widget.ax.texts]
        assert any("Traced grating diffraction" in label for label in labels)
        for wl in WAVELENGTHS:
            assert any(f"{wl * 1000:.0f} nm" in label for label in labels)

        colors = diffraction_wavelength_colors(WAVELENGTHS)
        for wl in WAVELENGTHS:
            plotted = [line for line in widget.ax.lines if line.get_color() == colors[wl]]
            assert plotted
            # The first point is the launch, not the final detector position.
            assert any(float(line.get_xdata()[0]) == pytest.approx(-1.0) for line in plotted)

        full_width = widget.ax.get_xlim()[1] - widget.ax.get_xlim()[0]
        assert widget.focus_grating_diffraction()
        assert widget.ax.get_xlim()[1] - widget.ax.get_xlim()[0] < full_width / 3

        widget.set_show_rays(False)
        assert not any("Traced grating diffraction" in item.get_text() for item in widget.ax.texts)
    finally:
        widget.close()
