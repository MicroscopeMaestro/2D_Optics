import sys
from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": ["os", "sys", "numpy", "matplotlib", "scipy", "PySide6", "v2", "optics_app"],
    "excludes": ["tkinter"],
    "include_files": []
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="2DOptics_V2",
    version="2.0",
    description="2D Optics V2",
    options={"build_exe": build_exe_options},
    executables=[Executable("launch_v2.py", base=base, target_name="2DOptics_V2.exe")]
)
