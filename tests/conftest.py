"""
Pytest configuration for 2D Optics Simulation.
Enforces headless Qt platform mode and non-interactive Matplotlib Agg backend for test execution.
"""
import os
import sys
from pathlib import Path
import matplotlib

# Ensure repository root is on sys.path
root_dir = str(Path(__file__).parent.parent.resolve())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"
matplotlib.use("Agg", force=True)

