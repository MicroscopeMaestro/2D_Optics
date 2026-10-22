"""
Zemax System Configuration Data Model.
Manages Aperture, Wavelengths, Field Points, Units, Environment, and Ray Aiming sampling.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
import math
import numpy as np


class ApertureType(Enum):
    ENTRANCE_PUPIL_DIAMETER = "Entrance Pupil Diameter"
    IMAGE_SPACE_F_NUM = "Image Space F/#"
    OBJECT_SPACE_NA = "Object Space NA"
    FLOAT_BY_STOP_SIZE = "Float by Stop Size"
    PARAXIAL_WORKING_F_NUM = "Paraxial Working F/#"
    OBJECT_CONE_ANGLE = "Object Cone Angle"


class ApodizationType(Enum):
    UNIFORM = "Uniform"
    GAUSSIAN = "Gaussian"
    COSINE_CUBED = "Cosine-Cubed"


class FieldType(Enum):
    ANGLE_DEG = "Angle (deg)"
    OBJECT_HEIGHT_MM = "Object Height (mm)"
    PARAXIAL_IMAGE_HEIGHT_MM = "Paraxial Image Height (mm)"
    REAL_IMAGE_HEIGHT_MM = "Real Image Height (mm)"


class RayPatternType(Enum):
    HEXAPOLAR = "Hexapolar"
    SQUARE_GRID = "Square Grid"
    GAUSSIAN_QUADRATURE = "Gaussian Quadrature"
    RANDOM_MONTE_CARLO = "Random Monte Carlo"


@dataclass
class WavelengthEntry:
    """Individual optical wavelength specification in micrometers."""
    wavelength_um: float = 0.5876
    weight: float = 1.0
    is_primary: bool = False
    color_hex: str = "#00FF00"


@dataclass
class FieldPoint:
    """Individual optical field coordinate point."""
    x: float = 0.0
    y: float = 0.0
    weight: float = 1.0
    vignetting_factor: float = 0.0


class SystemConfig:
    """
    Complete Zemax System Explorer Configuration Container.
    """
    def __init__(self):
        # 1. Aperture & Beam Geometry
        self.beam_type: str = "Diverging (Point/Slit Source)"  # "Diverging (Point/Slit Source)" or "Collimated (Parallel Beam)"
        self.aperture_type: ApertureType = ApertureType.ENTRANCE_PUPIL_DIAMETER
        self.aperture_value: float = 25.4  # mm, NA, or degrees
        self.slit_width_um: float = 0.0    # 0.0 for point source, or 30.0, 50.0, 100.0 um for physical slit
        self.apodization_type: ApodizationType = ApodizationType.UNIFORM
        self.apodization_factor: float = 0.0
        self.telecentric_object_space: bool = False

        # 2. Wavelengths
        self.wavelengths: List[WavelengthEntry] = [
            WavelengthEntry(wavelength_um=0.48613, weight=1.0, is_primary=False, color_hex="#0000FF"),
            WavelengthEntry(wavelength_um=0.58756, weight=1.0, is_primary=True, color_hex="#00FF00"),
            WavelengthEntry(wavelength_um=0.65627, weight=1.0, is_primary=False, color_hex="#FF0000")
        ]

        # 3. Fields
        self.field_type: FieldType = FieldType.ANGLE_DEG
        self.fields: List[FieldPoint] = [
            FieldPoint(x=0.0, y=0.0, weight=1.0)
        ]

        # 4. Units & Environment
        self.lens_units: str = "Millimeters"
        self.source_units: str = "Watts"
        self.temperature_c: float = 20.0
        self.pressure_atm: float = 1.0

        # 5. Ray Aiming & Sampling
        self.ray_pattern: RayPatternType = RayPatternType.HEXAPOLAR
        self.pupil_rings: int = 6
        self.num_rays: int = 64

    @property
    def primary_wavelength_um(self) -> float:
        for w in self.wavelengths:
            if w.is_primary:
                return w.wavelength_um
        return self.wavelengths[0].wavelength_um if self.wavelengths else 0.5876

    @primary_wavelength_um.setter
    def primary_wavelength_um(self, val_um: float):
        found = False
        for w in self.wavelengths:
            if w.is_primary:
                w.wavelength_um = float(val_um)
                found = True
                break
        if not found:
            if self.wavelengths:
                self.wavelengths[0].wavelength_um = float(val_um)
                self.wavelengths[0].is_primary = True
            else:
                self.wavelengths = [WavelengthEntry(wavelength_um=float(val_um), weight=1.0, is_primary=True)]

    @property
    def wavelength_list_um(self) -> List[float]:
        return [w.wavelength_um for w in self.wavelengths]

    @wavelength_list_um.setter
    def wavelength_list_um(self, wls: List[float]):
        from optics_app.engine.system_config import WavelengthEntry
        self.wavelengths = [
            WavelengthEntry(wavelength_um=float(w), weight=1.0, is_primary=(i == 0))
            for i, w in enumerate(wls)
        ]

    @property
    def wavelength_weights(self) -> List[float]:
        return [w.weight for w in self.wavelengths]

    def set_primary_wavelength(self, index: int):
        for i, w in enumerate(self.wavelengths):
            w.is_primary = (i == index)

    def add_wavelength(self, wl_um: float, weight: float = 1.0, is_primary: bool = False):
        if is_primary:
            for w in self.wavelengths:
                w.is_primary = False
        self.wavelengths.append(WavelengthEntry(wavelength_um=wl_um, weight=weight, is_primary=is_primary))

    def remove_wavelength(self, index: int):
        if 0 <= index < len(self.wavelengths) and len(self.wavelengths) > 1:
            was_prim = self.wavelengths[index].is_primary
            self.wavelengths.pop(index)
            if was_prim and self.wavelengths:
                self.wavelengths[0].is_primary = True

    def apply_wavelength_preset(self, preset_name: str):
        """Apply standard optical dispersion wavelength presets."""
        presets = {
            "Ultrafast 800 nm (editable)": [
                (0.80000, 1.0, True, "#8B0000")
            ],
            "Visible (F, d, C)": [
                (0.48613, 1.0, False, "#0000FF"),
                (0.58756, 1.0, True, "#00FF00"),
                (0.65627, 1.0, False, "#FF0000")
            ],
            "NIR 785 nm Raman": [
                (0.78500, 1.0, True, "#FF4500"),
                (0.83000, 1.0, False, "#8B0000"),
                (0.88000, 1.0, False, "#4B0082")
            ],
            "NIR 1064 nm Laser": [
                (1.03000, 1.0, False, "#800000"),
                (1.06400, 1.0, True, "#8B0000"),
                (1.08000, 1.0, False, "#4A0000")
            ],
            "Telecom C-Band": [
                (1.53000, 1.0, False, "#333333"),
                (1.55000, 1.0, True, "#000000"),
                (1.56500, 1.0, False, "#555555")
            ],
            "HeNe Laser (632.8 nm)": [
                (0.63280, 1.0, True, "#FF0000")
            ],
            "UV (i, h, g)": [
                (0.36500, 1.0, False, "#7B68EE"),
                (0.40470, 1.0, True, "#4169E1"),
                (0.43580, 1.0, False, "#00BFFF")
            ]
        }
        if preset_name in presets:
            self.wavelengths = [
                WavelengthEntry(wavelength_um=wl, weight=wt, is_primary=prim, color_hex=c)
                for wl, wt, prim, c in presets[preset_name]
            ]

    def add_field(self, x: float, y: float, weight: float = 1.0):
        self.fields.append(FieldPoint(x=x, y=y, weight=weight))

    def remove_field(self, index: int):
        if 0 <= index < len(self.fields) and len(self.fields) > 1:
            self.fields.pop(index)

    def apply_field_preset(self, preset_name: str, max_field: float = 10.0):
        """Apply standard Zemax multi-field configurations."""
        if preset_name == "On-Axis Single Point":
            self.fields = [FieldPoint(x=0.0, y=0.0, weight=1.0)]
        elif preset_name == "3-Field Linear":
            self.fields = [
                FieldPoint(x=0.0, y=0.0, weight=1.0),
                FieldPoint(x=0.0, y=0.707 * max_field, weight=2.0),
                FieldPoint(x=0.0, y=max_field, weight=1.0)
            ]
        elif preset_name == "5-Field Cross":
            self.fields = [
                FieldPoint(x=0.0, y=0.0, weight=1.0),
                FieldPoint(x=0.0, y=max_field, weight=1.0),
                FieldPoint(x=0.0, y=-max_field, weight=1.0),
                FieldPoint(x=max_field, y=0.0, weight=1.0),
                FieldPoint(x=-max_field, y=0.0, weight=1.0)
            ]
        elif preset_name == "9-Field Grid":
            pts = []
            for y in [-max_field, 0.0, max_field]:
                for x in [-max_field, 0.0, max_field]:
                    pts.append(FieldPoint(x=x, y=y, weight=1.0))
            self.fields = pts

    def generate_input_rays(
        self,
        surfaces: List[Any],
        wavelengths: Optional[List[float]] = None,
        num_rays: Optional[int] = None,
        pupil_pattern: Optional[Union[str, RayPatternType]] = None,
        system: Optional[Any] = None,
        rings: Optional[int] = None,
        arms: Optional[int] = None,
    ) -> List[Any]:
        """
        Generates rays based on SystemConfig aperture, beam type, object thickness,
        multi-field points (Object Height mm or Angle deg), and wavelengths.
        Applies Zemax-style paraxial entrance pupil solve and ray aiming across full pupil.
        """
        from optics_app.engine.ray import Ray
        from optics_app.engine.pupil_solve import (
            calculate_entrance_pupil,
            aim_ray_to_pupil,
            aim_collimated_ray,
            generate_pupil_grid,
        )
        if not surfaces:
            return []

        wls = wavelengths or self.wavelength_list_um or [0.5876]
        d0 = float(surfaces[0].thickness) if len(surfaces) > 0 else 0.0
        r1 = float(surfaces[1].semi_diameter) if len(surfaces) > 1 and surfaces[1].semi_diameter > 0 else 12.7

        # Check if collimated input is active
        is_collimated = (self.beam_type == "Collimated (Parallel Beam)") or self.telecentric_object_space or (d0 > 1e4)

        # Paraxial entrance pupil solve cache per wavelength for rigorous chromatic ray aiming
        ep_cache: Dict[float, Tuple[float, float, int]] = {}
        def _get_ep_for_wl(target_wl: float) -> Tuple[float, float, int]:
            t_key = round(float(target_wl), 6)
            if t_key not in ep_cache:
                ep_cache[t_key] = calculate_entrance_pupil(
                    surfaces,
                    wavelength_um=float(target_wl),
                    default_epd=self.aperture_value if (self.aperture_type == ApertureType.ENTRANCE_PUPIL_DIAMETER and self.aperture_value > 0) else None,
                )
            return ep_cache[t_key]

        # Pre-seed for primary / initial reference
        z_ep, r_ep, stop_idx = _get_ep_for_wl(wls[0] if wls else 0.5876)

        # Resolve pattern
        pattern = pupil_pattern if pupil_pattern is not None else getattr(self, "ray_pattern", RayPatternType.HEXAPOLAR)
        p_str = pattern.value if hasattr(pattern, "value") else str(pattern)

        # Determine pupil coordinates (Px, Py, Weight)
        is_fan_mode = (
            p_str in ("Fan", "fan", "Tangential Fan", "Cross")
            or pupil_pattern in ("Fan", "fan", "Tangential Fan", "Cross")
            or (pupil_pattern is None and num_rays is not None)
            or (pupil_pattern is None and rings is None and arms is None)
        )
        if is_fan_mode:
            # 1D Fan covering full aperture [-1.0, 1.0]
            n_fan = num_rays if num_rays is not None else getattr(self, "num_rays", 9)
            if n_fan % 2 == 0:
                n_fan += 1  # Ensure odd count so center chief ray (Py=0.0) is sampled exactly
            pupil_pts = [(0.0, float(py), 1.0 / n_fan) for py in np.linspace(-1.0, 1.0, n_fan)]
            if not any(abs(px) < 1e-7 and abs(py) < 1e-7 for px, py, _ in pupil_pts):
                pupil_pts.append((0.0, 0.0, 1.0 / n_fan))
            if n_fan >= 3:
                pupil_pts.append((-1.0, 0.0, 1.0 / n_fan))
                pupil_pts.append((1.0, 0.0, 1.0 / n_fan))
        else:
            # Full 2D Pupil Grid (Hexapolar or Gaussian Quadrature)
            n_rings = rings if rings is not None else (
                max(1, int(round(math.sqrt(num_rays / 3.0)))) if num_rays is not None else getattr(self, "pupil_rings", 4)
            )
            n_arms = arms if arms is not None else getattr(self, "pupil_arms", 6)
            pupil_pts = generate_pupil_grid(
                pattern=pattern,
                rings=n_rings,
                arms=n_arms,
                include_chief_ray=True,
                linear_arms=True,
                max_radius=1.0,
            )

        # Slit / source extent points along Y
        slit_half_mm = (self.slit_width_um / 2000.0) if self.slit_width_um > 0 else 0.0
        slit_samples = [-slit_half_mm, 0.0, slit_half_mm] if (slit_half_mm > 0 and len(self.fields) <= 1) else [0.0]

        rays = []
        for wl in wls:
            z_ep, r_ep, stop_idx = _get_ep_for_wl(wl)
            for f_idx, field in enumerate(self.fields):
                # Determine field point coordinates in object space
                if self.field_type == FieldType.OBJECT_HEIGHT_MM:
                    fx_mm = float(field.x)
                    fy_mm = float(field.y)
                elif self.field_type == FieldType.ANGLE_DEG:
                    if is_collimated or d0 <= 0.0:
                        fx_mm = 0.0
                        fy_mm = 0.0
                    else:
                        fx_mm = -d0 * math.tan(math.radians(float(field.x)))
                        fy_mm = -d0 * math.tan(math.radians(float(field.y)))
                else:
                    fx_mm = float(field.x)
                    fy_mm = float(field.y)

                for y_slit in slit_samples:
                    obj_y = fy_mm + y_slit
                    obj_x = fx_mm

                    for px, py, wp in pupil_pts:
                        is_chief = (abs(px) < 1e-9 and abs(py) < 1e-9)
                        if is_collimated and self.field_type == FieldType.ANGLE_DEG:
                            ray = aim_collimated_ray(
                                surfaces=surfaces,
                                field_angle_deg=(float(field.x), float(field.y)),
                                pupil_norm=(float(px), float(py)),
                                wavelength_um=wl,
                                z_ep_global=z_ep,
                                r_ep=r_ep,
                            )
                        else:
                            ray = aim_ray_to_pupil(
                                surfaces=surfaces,
                                obj_point=(obj_x, obj_y, 0.0),
                                pupil_norm=(float(px), float(py)),
                                wavelength_um=wl,
                                z_ep_global=z_ep,
                                r_ep=r_ep,
                                system=system,
                            )

                        is_on_axis = (f_idx == 0 or (abs(float(field.x)) < 1e-6 and abs(float(field.y)) < 1e-6))
                        is_marginal = is_on_axis and (
                            (abs(px) < 1e-5 and abs(abs(py) - 1.0) < 1e-4) or
                            (abs(py) < 1e-5 and abs(abs(px) - 1.0) < 1e-4)
                        )

                        ray.field_idx = f_idx
                        ray.is_chief = is_chief
                        ray.is_marginal = is_marginal
                        ray.is_upper_marginal = bool(is_marginal and py > 0)
                        ray.is_lower_marginal = bool(is_marginal and py < 0)
                        ray.field_coord = (float(field.x), float(field.y))
                        ray.field_weight = float(getattr(field, "weight", 1.0))
                        ray.pupil_coord = (float(px), float(py))
                        ray.pupil_weight = float(wp)
                        rays.append(ray)
        return rays

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 2,
            "aperture": {
                "beam_type": self.beam_type,
                "type": self.aperture_type.value,
                "value": self.aperture_value,
                "slit_width_um": self.slit_width_um,
                "apodization_type": self.apodization_type.value,
                "apodization_factor": self.apodization_factor,
                "telecentric": self.telecentric_object_space
            },
            "wavelengths": [
                {"wavelength_um": w.wavelength_um, "weight": w.weight, "is_primary": w.is_primary, "color": w.color_hex}
                for w in self.wavelengths
            ],
            "fields": {
                "type": self.field_type.value,
                "points": [
                    {
                        "x": f.x,
                        "y": f.y,
                        "weight": f.weight,
                        "vignetting_factor": f.vignetting_factor,
                    }
                    for f in self.fields
                ]
            },
            "environment": {
                "lens_units": self.lens_units,
                "source_units": self.source_units,
                "temperature_c": self.temperature_c,
                "pressure_atm": self.pressure_atm
            },
            "sampling": {
                "ray_pattern": self.ray_pattern.value,
                "pupil_rings": self.pupil_rings,
                "num_rays": self.num_rays
            }
        }

    def from_dict(self, d: Dict[str, Any]):
        if not isinstance(d, dict):
            return
        ap = d.get("aperture", {})
        self.beam_type = ap.get("beam_type", self.beam_type)
        try:
            self.aperture_type = ApertureType(ap.get("type", ApertureType.ENTRANCE_PUPIL_DIAMETER.value))
        except ValueError:
            pass
        self.aperture_value = float(ap.get("value", self.aperture_value))
        self.slit_width_um = float(ap.get("slit_width_um", self.slit_width_um))
        try:
            self.apodization_type = ApodizationType(ap.get("apodization_type", ApodizationType.UNIFORM.value))
        except ValueError:
            pass
        self.apodization_factor = float(ap.get("apodization_factor", self.apodization_factor))
        self.telecentric_object_space = bool(ap.get("telecentric", self.telecentric_object_space))

        if "wavelengths" in d and isinstance(d.get("wavelengths"), list):
            wls = d["wavelengths"]
            self.wavelengths = [
                WavelengthEntry(
                    wavelength_um=float(w.get("wavelength_um", 0.5876)),
                    weight=float(w.get("weight", 1.0)),
                    is_primary=bool(w.get("is_primary", False)),
                    color_hex=str(w.get("color", "#00FF00"))
                )
                for w in wls
            ]

        fld = d.get("fields", {}) if isinstance(d.get("fields", {}), dict) else {}
        try:
            self.field_type = FieldType(fld.get("type", FieldType.ANGLE_DEG.value))
        except ValueError:
            pass
        if "points" in fld and isinstance(fld.get("points"), list):
            pts = fld["points"]
            self.fields = [
                FieldPoint(
                    x=float(p.get("x", 0.0)),
                    y=float(p.get("y", 0.0)),
                    weight=float(p.get("weight", 1.0)),
                    vignetting_factor=float(p.get("vignetting_factor", 0.0)),
                )
                for p in pts
            ]

        env = d.get("environment", {}) if isinstance(d.get("environment", {}), dict) else {}
        self.lens_units = str(env.get("lens_units", self.lens_units))
        self.source_units = str(env.get("source_units", self.source_units))
        self.temperature_c = float(env.get("temperature_c", self.temperature_c))
        self.pressure_atm = float(env.get("pressure_atm", self.pressure_atm))

        sampling = d.get("sampling", {}) if isinstance(d.get("sampling", {}), dict) else {}
        try:
            self.ray_pattern = RayPatternType(
                sampling.get("ray_pattern", self.ray_pattern.value)
            )
        except ValueError:
            pass
        self.pupil_rings = int(sampling.get("pupil_rings", self.pupil_rings))
        self.num_rays = int(sampling.get("num_rays", self.num_rays))

    def save_to_file(self, filepath: str) -> None:
        """Serialize System Explorer parameters to a standalone JSON file."""
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    def load_from_file(self, filepath: str) -> None:
        """Load and apply System Explorer parameters from a JSON file or full project file."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "system_config" in data and isinstance(data["system_config"], dict):
            data = data["system_config"]
        self.from_dict(data)
