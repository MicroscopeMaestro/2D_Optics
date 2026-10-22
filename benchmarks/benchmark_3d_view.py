"""Compare fast and detailed 3D rendering on a real project.

Run from the repository root:
    python -m benchmarks.benchmark_3d_view
"""

from __future__ import annotations

import argparse
import contextlib
import io
import statistics
import time

from optics_app.engine.lde import SurfaceStack
from optics_app.gui.qt_compat import QApplication
from optics_app.gui.visualizer_3d import Visualizer3DWidget, generate_3d_rays


def benchmark(project: str, repeats: int = 5, with_rays: bool = False) -> None:
    app = QApplication.instance() or QApplication([])
    stack = SurfaceStack.load_from_json(project)["stack"]
    if with_rays:
        with contextlib.redirect_stdout(io.StringIO()):
            rays = generate_3d_rays(stack, num_rings=1, rays_per_ring=4)
    else:
        rays = []
    widget = Visualizer3DWidget()
    # Measure one synchronous rasterization per iteration; update_layout's
    # queued draw would otherwise overlap a later timing sample.
    widget.canvas.draw_idle = lambda: None
    try:
        print(f"Project: {project} ({len(stack.surfaces)} surfaces, {len(rays)} rays)")
        for quality in ("detailed", "interactive"):
            widget.mesh_quality = quality
            build_samples = []
            draw_samples = []
            for _ in range(max(2, repeats)):
                start = time.perf_counter()
                widget.update_layout(surface_stack=stack, rays=rays)
                built = time.perf_counter()
                widget.canvas.draw()
                drawn = time.perf_counter()
                build_samples.append(built - start)
                draw_samples.append(drawn - built)
            polygons = sum(len(getattr(collection, "_segslices", [])) for collection in widget.ax.collections)
            print(
                f"{quality:11s} build={statistics.median(build_samples[1:]):.3f}s "
                f"draw={statistics.median(draw_samples[1:]):.3f}s "
                f"collections={len(widget.ax.collections)} polygons={polygons}"
            )
    finally:
        widget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", nargs="?", default="spectrometer_raman_50mm_100mm.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rays", action="store_true", help="Include a small traced 3D ray bundle")
    args = parser.parse_args()
    benchmark(args.project, args.repeats, args.rays)
