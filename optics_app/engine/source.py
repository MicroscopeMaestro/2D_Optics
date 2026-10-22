import json
import math
from enum import Enum
from typing import Dict, Any, Optional

class SourceType(Enum):
    COLLIMATED = "Collimated"
    POINT = "Point Source"
    GAUSSIAN = "Gaussian Beam"
    FIBER = "Fiber (Extended)"
    SLIT = "Entrance Slit"

class SourceConfig:
    """
    Optical Source and Input Aperture Configuration.
    Defines input beam geometry, fiber core diameter, entrance slit dimensions, and numerical aperture (NA).
    """
    def __init__(self):
        self.source_type: SourceType = SourceType.COLLIMATED
        self.beam_radius: float = 5.0            # mm
        self.divergence_angle: float = 10.0      # degrees (half-angle)
        self.num_rays: int = 7                   # rays across diameter (1D) or grid size
        self.wavelengths: str = "0.785, 0.850, 0.935"
        
        # Input Aperture Parameters
        self.input_aperture_shape: str = "Circular"  # "Circular" (fiber) or "Rectangular" (slit)
        self.input_aperture_width: float = 0.050     # mm (50 um fiber core / slit width)
        self.input_aperture_height: float = 1.000    # mm (1.0 mm slit height for rectangular slit)
        self.numerical_aperture: float = 0.12        # Numerical Aperture (NA = sin(theta))
        
        # Ultrafast Parameters
        self.laser_mode: str = "CW"                  # "CW" or "Ultrafast"
        self.pulse_duration_fs: float = 100.0        # fs
        self.bandwidth_nm: float = 10.0              # nm (FWHM intensity)

    @property
    def fiber_core_diameter_um(self) -> float:
        return self.input_aperture_width * 1000.0

    @fiber_core_diameter_um.setter
    def fiber_core_diameter_um(self, val_um: float):
        self.input_aperture_width = val_um / 1000.0
        self.beam_radius = (val_um / 2000.0)

    def set_na(self, na: float):
        """Set numerical aperture and compute corresponding divergence half-angle."""
        self.numerical_aperture = float(np_clip_na := min(max(na, 0.001), 0.999))
        self.divergence_angle = math.degrees(math.asin(self.numerical_aperture))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "beam_radius": self.beam_radius,
            "divergence_angle": self.divergence_angle,
            "num_rays": self.num_rays,
            "wavelengths": self.wavelengths,
            "input_aperture_shape": self.input_aperture_shape,
            "input_aperture_width": self.input_aperture_width,
            "input_aperture_height": self.input_aperture_height,
            "numerical_aperture": self.numerical_aperture,
            "laser_mode": self.laser_mode,
            "pulse_duration_fs": self.pulse_duration_fs,
            "bandwidth_nm": self.bandwidth_nm
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        try:
            type_str = data.get("source_type", SourceType.COLLIMATED.value)
            self.source_type = SourceType(type_str)
        except ValueError:
            self.source_type = SourceType.COLLIMATED
            
        self.beam_radius = float(data.get("beam_radius", self.beam_radius))
        self.divergence_angle = float(data.get("divergence_angle", self.divergence_angle))
        self.num_rays = int(data.get("num_rays", self.num_rays))
        self.wavelengths = str(data.get("wavelengths", self.wavelengths))
        self.input_aperture_shape = str(data.get("input_aperture_shape", self.input_aperture_shape))
        self.input_aperture_width = float(data.get("input_aperture_width", self.input_aperture_width))
        self.input_aperture_height = float(data.get("input_aperture_height", self.input_aperture_height))
        self.numerical_aperture = float(data.get("numerical_aperture", self.numerical_aperture))
        self.laser_mode = str(data.get("laser_mode", self.laser_mode))
        self.pulse_duration_fs = float(data.get("pulse_duration_fs", self.pulse_duration_fs))
        self.bandwidth_nm = float(data.get("bandwidth_nm", self.bandwidth_nm))

    def save_to_json(self, filepath: str) -> None:
        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    def load_from_json(self, filepath: str) -> None:
        try:
            with open(filepath, 'r', encoding="utf-8") as f:
                data = json.load(f)
            self.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
