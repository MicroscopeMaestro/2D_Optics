#!/usr/bin/env python3
"""
Compare hybrid_raman_spectrometer.json with other design variations in workspace
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

files_to_check = [
    "hybrid_raman_spectrometer.json",
    "hybrid_raman_1064_full_range.json",
    "hybrid_raman_1064_full_optimized.json",
    "hybrid_raman_1064_physically_correct.json",
    "hybrid_raman_1064.json"
]

for fn in files_to_check:
    if not os.path.exists(fn):
        continue
    print("=" * 70)
    print(f"FILE: {fn}")
    print("=" * 70)
    with open(fn, 'r') as f:
        d = json.load(f)
    print(f"Surfaces: {len(d.get('surfaces', []))}")
    for i, s in enumerate(d.get('surfaces', [])):
        print(f"  [{i:2d}] {s.get('name'):<25} r={s.get('radius'):<8.2f} t={s.get('thickness'):<6.2f} g={s.get('glass'):<8} sd={s.get('semi_diameter'):<5.1f} tilt_y={s.get('tilt_y'):<5.1f} grating={s.get('grating_lines_per_mm'):<5.1f}")
