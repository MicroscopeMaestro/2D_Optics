"""
Zemax (.zmx) Prescription File Loader and Exporter for optics_app.engine.
"""
from typing import Optional, Union, List, Dict, Any
import os

try:
    from optics_app.io.zmx_parser import (
        parse_zmx_file, parse_zmx_string,
        export_to_zmx_string, export_to_zmx_file
    )
except ImportError:
    from v2.io.zmx_parser import (
        parse_zmx_file, parse_zmx_string,
        export_to_zmx_string, export_to_zmx_file
    )

import_zmx = parse_zmx_file

__all__ = [
    "parse_zmx_file",
    "parse_zmx_string",
    "export_to_zmx_string",
    "export_to_zmx_file",
    "import_zmx"
]
