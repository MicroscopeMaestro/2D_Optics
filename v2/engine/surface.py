"""
Optical Surface Representations (Standard, Aspheric, Grating, Stop, Detector, CAD).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict, Union
import numpy as np

try:
    from optics_app.engine.components import AOMConfig, EOMConfig
except ImportError:
    try:
        from v2.engine.components import AOMConfig, EOMConfig
    except ImportError:
        AOMConfig = Any  # type: ignore
        EOMConfig = Any  # type: ignore


class SurfaceType:
    # Standard Zemax OpticStudio Types
    STANDARD = "Standard"
    COORDINATE_BREAK = "Coordinate Break"
    ASPHERIC = "Aspheric"
    EVEN_ASPHERE = "Even Asphere"
    GRATING = "Grating"
    DIFFRACTION_GRATING = "Diffraction Grating"
    TOROIDAL = "Toroidal"
    CAD = "CAD"
    TILTSURF = "Tilted"

    # Zemax Surface Designations and Functional Roles
    MIRROR = "Mirror"
    STOP = "Stop"
    OBJECT = "Object"
    DETECTOR = "Detector"
    IMAGE = "Image"
    AIR_SPACE = "Air Space"

    @classmethod
    def normalize(cls, val: Any) -> str:
        """Normalize any Zemax code or alias string into standard canonical SurfaceType."""
        if not val:
            return cls.STANDARD
        s = str(val).strip().upper()
        if s in ("COORDBRK", "CBREAK", "COORDINATE_BREAK", "COORDINATE BREAK"):
            return cls.COORDINATE_BREAK
        if s in ("EVENASPH", "EVEN_ASPHERE", "EVEN ASPHERE", "ASPHERIC", "ASPHERE"):
            return cls.ASPHERIC
        if s in ("DIFFRACT", "DGRATING", "DIFFRACTION_GRATING", "DIFFRACTION GRATING", "GRATING"):
            return cls.GRATING
        if s in ("TOROIDAL", "TORIC"):
            return cls.TOROIDAL
        if s in ("TILTSURF", "TILTED"):
            return cls.TILTSURF
        if s in ("CAD", "GRID SAG", "GRIDSAG"):
            return cls.CAD
        if s in ("MIRROR", "MIRR", "REFL"):
            return cls.MIRROR
        if s in ("STOP", "STO", "APERTURE STOP", "APER STOP"):
            return cls.STOP
        if s in ("OBJECT", "OBJ"):
            return cls.OBJECT
        if s in ("DETECTOR", "IMAGE", "IMA", "IMJ"):
            return cls.DETECTOR
        if s in ("AIR_SPACE", "AIR SPACE", "AIRSPACE"):
            return cls.AIR_SPACE
        for attr, const_val in cls.__dict__.items():
            if isinstance(const_val, str) and const_val.upper() == s:
                return const_val
        return str(val).strip()

    @classmethod
    def to_zemax(cls, val: Any) -> str:
        """Return the standard Zemax OpticStudio .zmx TYPE token for this surface."""
        norm = cls.normalize(val)
        if norm in (cls.ASPHERIC, cls.EVEN_ASPHERE):
            return "EVENASPH"
        if norm in (cls.GRATING, cls.DIFFRACTION_GRATING):
            return "DIFFRACT"
        if norm == cls.COORDINATE_BREAK:
            return "COORDBRK"
        if norm == cls.TOROIDAL:
            return "TOROIDAL"
        if norm == cls.CAD:
            return "CAD"
        return "STANDARD"

@dataclass
class Surface:
    name: str = "Surface"
    is_active: bool = True
    surface_type: str = SurfaceType.STANDARD
    is_coord_break: bool = False
    is_stop: bool = False
    radius: float = 0.0          # 0.0 or inf denotes flat plane
    thickness: float = 0.0       # Z-distance to next surface
    glass: str = "AIR"           # Material name
    semi_diameter: float = 12.7  # Clear aperture radius (mm) or radial bound
    semi_diameter_x: float = 0.0 # Substrate physical half-width X (mm)
    semi_diameter_y: float = 0.0 # Substrate physical half-width Y (mm)
    conic: float = 0.0           # Conic constant k
    asphere_coeffs: List[float] = field(default_factory=list) # Polynomial asphere terms A4, A6, A8...
    decenter_x: float = 0.0
    decenter_y: float = 0.0
    decenter_z: float = 0.0
    tilt_x: float = 0.0          # Degrees
    tilt_y: float = 0.0          # Degrees
    tilt_z: float = 0.0          # Degrees
    global_origin: Any = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)) # Absolute 3D vertex position
    global_R: Any = field(default_factory=lambda: np.eye(3, dtype=float))                     # Absolute 3D orientation matrix
    grating_lines_per_mm: float = 0.0
    grating_order: int = 1
    grating_primary_wavelength_um: Optional[float] = None # Reference wavelength (µm) diffracted along optical axis (None = follow system)
    grating_blaze_wavelength_um: Optional[float] = None   # Blaze wavelength (µm) where peak efficiency occurs
    grating_blaze_angle_deg: Optional[float] = None       # Blaze angle (deg) of groove facets
    grating_type: str = "Blazed Reflective"               # "Blazed Reflective", "Holographic", "Transmission"
    is_transmission_grating: bool = False                 # Flag if grating is transmission rather than reflection
    cad_profile: Optional[Any] = None

    # Substrate Volume / 3D Mechanical Backing for Mirrors & Reflective Elements
    substrate_thickness: float = 6.0   # Physical 3D thickness (volume) of mirror/grating/DMD substrate (mm)
    substrate_shape: str = "Auto"      # "Auto", "Circular", or "Rectangular"

    # Variable Thickness & Optimization Attributes (Milestones M1 & M2)
    thickness_variable: bool = False
    thickness_min: float = 0.0
    thickness_max: float = 1000.0

    radius_variable: bool = False
    radius_min: float = -1000.0
    radius_max: float = 1000.0

    tilt_x_variable: bool = False
    tilt_x_min: float = -45.0
    tilt_x_max: float = 45.0

    tilt_y_variable: bool = False
    tilt_y_min: float = -45.0
    tilt_y_max: float = 45.0

    tilt_z_variable: bool = False
    tilt_z_min: float = -45.0
    tilt_z_max: float = 45.0

    decenter_x_variable: bool = False
    decenter_x_min: float = -25.0
    decenter_x_max: float = 25.0

    decenter_y_variable: bool = False
    decenter_y_min: float = -25.0
    decenter_y_max: float = 25.0

    decenter_z_variable: bool = False
    decenter_z_min: float = -25.0
    decenter_z_max: float = 25.0

    conic_variable: bool = False
    conic_min: float = -100.0
    conic_max: float = 100.0

    # Zemax Solve & Pickup Attributes
    radius_solve_type: str = "Fixed"       # "Fixed", "Variable", "Pickup", "MarginalRayNormal"
    radius_pickup_surface: int = -1        # Index of surface to pick up from
    radius_pickup_scale: float = 1.0       # Scale factor (e.g. -1.0 for symmetric)
    radius_pickup_offset: float = 0.0      # Offset added to picked up value

    thickness_solve_type: str = "Fixed"    # "Fixed", "Variable", "Pickup", "MarginalRayFocus"
    thickness_pickup_surface: int = -1
    thickness_pickup_scale: float = 1.0
    thickness_pickup_offset: float = 0.0

    # Complex Aperture & Obscuration Attributes (Requirement R5)
    extinction_coeff: float = 0.0  # Extinction coefficient k for complex refractive index n + ik
    aperture_type: str = "Circular"
    aperture_x_half: float = 0.0
    aperture_y_half: float = 0.0
    aperture_a_semi: float = 0.0
    aperture_b_semi: float = 0.0
    aperture_r_inner: float = 0.0
    aperture_r_outer: float = 0.0
    aperture_center_radius: float = 5.0
    aperture_spider_width: float = 1.0
    aperture_spider_count: int = 4
    aperture_spider_angle: float = 0.0
    jones_matrix: Optional[Any] = None  # Optional 2x2 complex Jones matrix for polarization components

    # Explicit Lens Element / Objective Group Assignment & Parameter Protection
    element_id: Optional[int] = None    # Integer ID of the lens element / objective this surface belongs to
    component_uid: str = ""             # Stable project-wide ID shared by every surface in one physical component
    group_name: str = ""                # Name of the optical assembly (e.g. "Objective", "Collimator", "Doublet 1")
    is_locked: bool = False             # Locks lens parameters (radius, thickness, glass) when set to True on catalog lenses
    catalog_metadata: Dict[str, Any] = field(default_factory=dict)  # Manufacturer identity, provenance and uncalibrated limits

    # Mechanical/actuated optical state.  State rotations are deliberately
    # separate from tilt_x/y/z: the latter positions the component housing and
    # the downstream sequential coordinate frame, while these state offsets
    # rotate only the active optical member (for example a DMD micromirror).
    mechanism_id: str = ""
    mechanism_type: str = ""
    mechanical_state: str = ""
    mechanical_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Generic Optical Component & Actuation Metadata
    galvo_config: Optional[Any] = None
    beamsplitter_config: Optional[Any] = None
    prism_config: Optional[Any] = None
    aom_config: Optional[AOMConfig] = None
    eom_config: Optional[EOMConfig] = None
    plm_config: Optional[Any] = None
    is_plm: bool = False
    is_dmd: bool = False
    mechanism: Optional[Any] = None
    is_beam_splitter: bool = False
    active_branch: str = "transmitted"
    reflected_branch_start_idx: Optional[int] = None
    component_type: str = ""

    def __post_init__(self):
        if isinstance(self.galvo_config, dict):
            try:
                from optics_app.engine.components import GalvoScannerConfig
                self.galvo_config = GalvoScannerConfig.from_dict(self.galvo_config)
            except Exception:
                try:
                    from v2.engine.components import GalvoScannerConfig
                    self.galvo_config = GalvoScannerConfig.from_dict(self.galvo_config)
                except Exception:
                    pass

        if isinstance(self.beamsplitter_config, dict):
            try:
                from optics_app.engine.components import BeamSplitterConfig
                self.beamsplitter_config = BeamSplitterConfig.from_dict(self.beamsplitter_config)
                self.is_beam_splitter = True
            except Exception:
                try:
                    from v2.engine.components import BeamSplitterConfig
                    self.beamsplitter_config = BeamSplitterConfig.from_dict(self.beamsplitter_config)
                    self.is_beam_splitter = True
                except Exception:
                    pass

        if self.beamsplitter_config is not None:
            self.is_beam_splitter = True

        if isinstance(self.prism_config, dict):
            try:
                from optics_app.engine.components import PrismConfig
                self.prism_config = PrismConfig.from_dict(self.prism_config)
            except Exception:
                try:
                    from v2.engine.components import PrismConfig
                    self.prism_config = PrismConfig.from_dict(self.prism_config)
                except Exception:
                    pass

        if isinstance(self.aom_config, dict):
            try:
                from optics_app.engine.components import AOMConfig
                self.aom_config = AOMConfig.from_dict(self.aom_config)
            except Exception:
                try:
                    from v2.engine.components import AOMConfig
                    self.aom_config = AOMConfig.from_dict(self.aom_config)
                except Exception:
                    pass

        if isinstance(self.eom_config, dict):
            try:
                from optics_app.engine.components import EOMConfig
                self.eom_config = EOMConfig.from_dict(self.eom_config)
            except Exception:
                try:
                    from v2.engine.components import EOMConfig
                    self.eom_config = EOMConfig.from_dict(self.eom_config)
                except Exception:
                    pass

        if isinstance(self.plm_config, dict):
            try:
                from optics_app.engine.plm import PLMConfig
                self.plm_config = PLMConfig.from_dict(self.plm_config)
            except Exception:
                try:
                    from v2.engine.plm import PLMConfig
                    self.plm_config = PLMConfig.from_dict(self.plm_config)
                except Exception:
                    pass

        if self.plm_config is not None:
            if getattr(self.plm_config, 'is_dmd', False) or getattr(self.plm_config, 'modulator_type', '') == 'DMD':
                self.is_dmd = True
                self.is_plm = False
            else:
                self.is_plm = True
                self.is_dmd = False

        if isinstance(self.mechanism, dict):
            try:
                from optics_app.engine.components import ActiveMechanism
                self.mechanism = ActiveMechanism.from_dict(self.mechanism)
            except Exception:
                try:
                    from v2.engine.components import ActiveMechanism
                    self.mechanism = ActiveMechanism.from_dict(self.mechanism)
                except Exception:
                    pass

        # Standard Zemax surface type & glass harmonization
        norm_type = SurfaceType.normalize(self.surface_type)
        if norm_type == SurfaceType.MIRROR:
            if not self.glass or self.glass.upper() in ("AIR", "NONE", "", "VACUUM"):
                self.glass = "MIRROR"
        elif self.glass and self.glass.upper() in ("MIRROR", "MIRR", "REFL"):
            if self.surface_type == SurfaceType.STANDARD:
                self.surface_type = SurfaceType.MIRROR
        if norm_type == SurfaceType.COORDINATE_BREAK:
            self.is_coord_break = True
        if norm_type == SurfaceType.STOP:
            self.is_stop = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Surface to dictionary, preserving components and mechanisms."""
        import dataclasses
        saved_plm = self.plm_config
        self.plm_config = None
        try:
            d = dataclasses.asdict(self)
        finally:
            self.plm_config = saved_plm
        if "cad_profile" in d:
            d["cad_profile"] = None
        if "global_origin" in d:
            del d["global_origin"]
        if "global_R" in d:
            del d["global_R"]
        if self.galvo_config is not None and hasattr(self.galvo_config, "to_dict"):
            d["galvo_config"] = self.galvo_config.to_dict()
        if self.beamsplitter_config is not None and hasattr(self.beamsplitter_config, "to_dict"):
            d["beamsplitter_config"] = self.beamsplitter_config.to_dict()
        if self.prism_config is not None and hasattr(self.prism_config, "to_dict"):
            d["prism_config"] = self.prism_config.to_dict()
        if self.aom_config is not None and hasattr(self.aom_config, "to_dict"):
            d["aom_config"] = self.aom_config.to_dict()
        if self.eom_config is not None and hasattr(self.eom_config, "to_dict"):
            d["eom_config"] = self.eom_config.to_dict()
        if self.plm_config is not None:
            d["plm_config"] = self.plm_config.to_dict() if hasattr(self.plm_config, "to_dict") else self.plm_config
        if self.mechanism is not None and hasattr(self.mechanism, "to_dict"):
            d["mechanism"] = self.mechanism.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Surface":
        clean_d = {k: v for k, v in data.items() if k not in ("global_origin", "global_R", "cad_profile") and (hasattr(cls, k) or k in cls.__dataclass_fields__)}
        if "aom_config" in clean_d and isinstance(clean_d["aom_config"], dict):
            try:
                from optics_app.engine.components import AOMConfig
                clean_d["aom_config"] = AOMConfig.from_dict(clean_d["aom_config"])
            except Exception:
                try:
                    from v2.engine.components import AOMConfig
                    clean_d["aom_config"] = AOMConfig.from_dict(clean_d["aom_config"])
                except Exception:
                    pass
        if "eom_config" in clean_d and isinstance(clean_d["eom_config"], dict):
            try:
                from optics_app.engine.components import EOMConfig
                clean_d["eom_config"] = EOMConfig.from_dict(clean_d["eom_config"])
            except Exception:
                try:
                    from v2.engine.components import EOMConfig
                    clean_d["eom_config"] = EOMConfig.from_dict(clean_d["eom_config"])
                except Exception:
                    pass
        if "plm_config" in clean_d and isinstance(clean_d["plm_config"], dict):
            try:
                from optics_app.engine.plm import PLMConfig
                clean_d["plm_config"] = PLMConfig.from_dict(clean_d["plm_config"])
            except Exception:
                try:
                    from v2.engine.plm import PLMConfig
                    clean_d["plm_config"] = PLMConfig.from_dict(clean_d["plm_config"])
                except Exception:
                    pass
        return cls(**clean_d)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Surface):
            return False
        if self is other:
            return True
        return (
            self.name == other.name and
            self.is_active == other.is_active and
            self.surface_type == other.surface_type and
            abs(self.radius - other.radius) < 1e-9 and
            abs(self.thickness - other.thickness) < 1e-9 and
            self.glass == other.glass and
            abs(self.semi_diameter - other.semi_diameter) < 1e-9 and
            abs(self.conic - other.conic) < 1e-9 and
            abs(self.decenter_x - other.decenter_x) < 1e-9 and
            abs(self.decenter_y - other.decenter_y) < 1e-9 and
            abs(getattr(self, 'decenter_z', 0.0) - getattr(other, 'decenter_z', 0.0)) < 1e-9 and
            abs(self.tilt_x - other.tilt_x) < 1e-9 and
            abs(self.tilt_y - other.tilt_y) < 1e-9 and
            abs(self.tilt_z - other.tilt_z) < 1e-9 and
            self.thickness_variable == other.thickness_variable and
            abs(self.thickness_min - other.thickness_min) < 1e-9 and
            abs(self.thickness_max - other.thickness_max) < 1e-9
        )

    def is_reflective(self) -> bool:
        """Check if this surface acts as a mirror or reflective element."""
        if self.is_air_space:
            return False
        if self.glass and self.glass.upper() in ["MIRROR", "MIRR", "REFL"]:
            return True
        if "MIRROR" in self.name.upper() or "PLM" in self.name.upper() or getattr(self, "is_plm", False) or getattr(self, "plm_config", None) is not None:
            return True
        if self.surface_type and str(self.surface_type).upper() == "MIRROR":
            return True
        if self.surface_type and str(self.surface_type).upper() == SurfaceType.GRATING.upper():
            if getattr(self, "is_transmission_grating", False):
                return False
            if str(getattr(self, "grating_type", "")).upper() == "TRANSMISSION":
                return False
            return True
        return False

    def is_mirror(self) -> bool:
        """True if this surface is reflective or acts as a mirror/grating substrate."""
        return not self.is_air_space and (self.is_reflective() or "DMD" in self.name.upper() or "PLM" in self.name.upper())

    @property
    def is_air_space(self) -> bool:
        """Spacing bookkeeping has no optical face, aperture or reflection.

        Recognize the flat AIR_after_* rows written by older projects. A lens
        rear face with AIR as its outgoing medium remains a physical surface.
        """
        if self.surface_type == SurfaceType.AIR_SPACE:
            return True
        return (
            self.surface_type == SurfaceType.STANDARD
            and str(self.name or "").upper().startswith("AIR_AFTER_")
            and self.is_air
            and (self.radius == 0.0 or np.isinf(self.radius))
            and self.conic == 0.0 and not self.asphere_coeffs
            and not self.is_coord_break
            and not self.mechanism_type and not self.mechanical_states
            and self.jones_matrix is None
        )

    @property
    def blaze_angle(self) -> float:
        """Return blaze angle in degrees, computing Littrow angle if not explicitly set."""
        if self.grating_blaze_angle_deg is not None:
            return float(self.grating_blaze_angle_deg)
        wl = self.grating_blaze_wavelength_um or self.grating_primary_wavelength_um
        lines = self.grating_lines_per_mm
        if wl and lines > 0:
            sin_b = (lines * wl * abs(self.grating_order)) / 2000.0
            if 0 < sin_b < 1.0:
                return float(np.degrees(np.arcsin(sin_b)))
        return 0.0

    @property
    def blaze_wavelength(self) -> Optional[float]:
        """Return blaze wavelength in µm."""
        return self.grating_blaze_wavelength_um or self.grating_primary_wavelength_um

    @property
    def mirror_thickness(self) -> float:
        """Mechanical substrate volume / backing thickness (mm) for mirrors, gratings, and DMDs."""
        return self.substrate_thickness

    @mirror_thickness.setter
    def mirror_thickness(self, val: float):
        self.substrate_thickness = float(val)

    def calculate_diffraction_angle(
        self,
        incident_angle_deg: Optional[float] = None,
        wavelength_um: Optional[float] = None,
        alpha_deg: Optional[float] = None
    ) -> float:
        """
        Calculate diffracted output angle beta (deg) from input incident angle alpha (deg)
        using the reflection grating equation: sin(beta) = m * lambda / d - sin(alpha).
        """
        inc_deg = incident_angle_deg if incident_angle_deg is not None else (alpha_deg if alpha_deg is not None else float(getattr(self, 'tilt_x', 0.0)))
        wl = wavelength_um or self.blaze_wavelength or 1.0
        lines = self.grating_lines_per_mm
        if lines <= 0:
            return -inc_deg
        d_um = 1000.0 / lines
        m = self.grating_order or 1
        sin_alpha = np.sin(np.radians(inc_deg))
        sin_beta = (m * wl / d_um) - sin_alpha
        if abs(sin_beta) > 1.0:
            raise ValueError(f"Evanescent diffraction order: sin(beta) = {sin_beta:.4f} > 1.0")
        return float(np.degrees(np.arcsin(sin_beta)))

    def calculate_blaze_condition(
        self,
        incident_angle_deg: Optional[float] = None,
        alpha_deg: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute the blazed reflection condition for this grating at incident angle alpha.
        At the blaze peak, specular facet reflection from groove facet angle theta_B satisfies:
        alpha + beta = 2 * theta_B
        m * lambda_B = 2 * d * sin(theta_B) * cos(alpha - theta_B)
        """
        inc_deg = incident_angle_deg if incident_angle_deg is not None else (alpha_deg if alpha_deg is not None else float(getattr(self, 'tilt_x', 0.0)))
        theta_b = self.blaze_angle
        lines = self.grating_lines_per_mm
        d_um = 1000.0 / lines if lines > 0 else 1.0
        m = self.grating_order or 1
        theta_b_rad = np.radians(theta_b)
        alpha_rad = np.radians(inc_deg)
        beta_rad = 2.0 * theta_b_rad - alpha_rad
        beta_deg = float(np.degrees(beta_rad))
        lambda_blaze_um = float((2.0 * d_um * np.sin(theta_b_rad) * np.cos(alpha_rad - theta_b_rad)) / abs(m))
        return {
            "valid": True,
            "incident_angle_deg": float(inc_deg),
            "blaze_angle_deg": float(theta_b),
            "diffracted_angle_deg": beta_deg,
            "total_fold_deg": float(inc_deg + beta_deg),
            "blaze_wavelength_um": max(0.0, lambda_blaze_um),
            "blaze_wavelength_nm": max(0.0, lambda_blaze_um * 1000.0),
        }

    def calculate_dmd_reflection_angles(
        self,
        incident_angle_deg: Optional[float] = None,
        state: Optional[str] = None,
        alpha_deg: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate Texas Instruments DLP4500 input and output angles according to active state.
        In DLP4500:
        Micromirrors tilt by alpha_tilt (+12° in ON, -12° in OFF, 0° in FLAT).
        Given incident angle theta_in relative to the substrate window normal:
        theta_out = 2 * alpha_tilt - theta_in.
        For nominal theta_in = 24.0°:
        - ON (+12°): theta_out = 2*(+12°) - 24° = 0.0° (along normal to detector)
        - OFF (-12°): theta_out = 2*(-12°) - 24° = -48.0° (steered 48° away into beam dump)
        - FLAT (0°): theta_out = 2*(0°) - 24° = -24.0° (specular 24° reflection)
        """
        inc_deg = incident_angle_deg if incident_angle_deg is not None else (alpha_deg if alpha_deg is not None else 24.0)
        from optics_app.engine.mechanisms import dmd4500_reflection_angles, active_state_name
        st = state or active_state_name(self)
        return dmd4500_reflection_angles(input_angle_deg=inc_deg, state=st)

    @property
    def is_air(self) -> bool:
        """True if the propagation medium following this surface is AIR/vacuum."""
        if self.is_air_space:
            return True
        g = (self.glass or "").strip().upper()
        if self.is_reflective() or g in ("MIRROR", "MIRR", "REFL"):
            return False
        return g in ["", "AIR", "VACUUM", "NONE"]

    @property
    def is_glass(self) -> bool:
        """True if this surface is inside a solid optical medium (lens element)."""
        return not self.is_air and not self.is_reflective()

    @property
    def global_z(self) -> float:
        return float(self.global_origin[2])

    @global_z.setter
    def global_z(self, val: float):
        self.global_origin[2] = val

    def sag(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate surface sag z(r) for spherical, conic, or polynomial asphere."""
        is_array = isinstance(r, np.ndarray)
        r_arr = np.asarray(r, dtype=float)
        if self.radius == 0.0 or np.isinf(self.radius):
            z = np.zeros_like(r_arr)
        else:
            c = 1.0 / self.radius
            r2 = r_arr**2
            arg = 1.0 - (1.0 + self.conic) * c**2 * r2
            denom = 1.0 + np.sqrt(np.maximum(0.0, arg))
            z = c * r2 / denom
        for i, a in enumerate(self.asphere_coeffs):
            z = z + a * r_arr**(2 * i + 4)
        return z if is_array else float(z)

    def normal_vector(self, x: float, y: float) -> np.ndarray:
        """
        Compute true outward unit surface normal N = (-dz/dx, -dz/dy, 1) / norm.
        """
        r = float(np.sqrt(x**2 + y**2))
        if self.radius == 0.0 or np.isinf(self.radius):
            dz_dr = 0.0
        else:
            c = 1.0 / self.radius
            r2 = r**2
            arg = 1.0 - (1.0 + self.conic) * c**2 * r2
            dz_dr = (c * r) / np.sqrt(arg) if arg > 0.0 else 0.0

        for i, a in enumerate(self.asphere_coeffs):
            power = 2 * i + 4
            dz_dr += power * a * (r**(power - 1)) if r > 0 else 0.0

        if r > 0:
            dz_dx = dz_dr * (x / r)
            dz_dy = dz_dr * (y / r)
        else:
            dz_dx = 0.0
            dz_dy = 0.0

        raw_norm = np.array([-dz_dx, -dz_dy, 1.0], dtype=float)
        norm_val = np.linalg.norm(raw_norm)
        return raw_norm / norm_val if norm_val > 0 else np.array([0.0, 0.0, 1.0])

    def check_aperture(self, x: float, y: float) -> bool:
        """
        Check if local transverse intersection point (x, y) passes through
        clear aperture and avoids central hub & spider leg obscurations.
        Returns True if passed (unvignetted), False if vignetted/blocked.
        """
        if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
            return False

        atype = str(self.aperture_type).capitalize() if self.aperture_type else "Circular"
        outer_r = self.aperture_r_outer if self.aperture_r_outer > 0 else self.semi_diameter

        if atype in ("Point", "None"):
            return True

        if atype == "Circular":
            if outer_r <= 0:
                return True
            return (x**2 + y**2) <= (outer_r**2 + 1e-9)

        elif atype == "Rectangular":
            hx = self.aperture_x_half if self.aperture_x_half > 0 else self.semi_diameter
            hy = self.aperture_y_half if self.aperture_y_half > 0 else self.semi_diameter
            if hx <= 0 and hy <= 0:
                return True
            if hx <= 0:
                return abs(y) <= (hy + 1e-9)
            if hy <= 0:
                return abs(x) <= (hx + 1e-9)
            return abs(x) <= (hx + 1e-9) and abs(y) <= (hy + 1e-9)

        elif atype == "Elliptical":
            a = self.aperture_a_semi if self.aperture_a_semi > 0 else self.semi_diameter
            b = self.aperture_b_semi if self.aperture_b_semi > 0 else self.semi_diameter
            if a <= 0 and b <= 0:
                return True
            if a <= 0 or b <= 0:
                return False
            return (x / a)**2 + (y / b)**2 <= (1.0 + 1e-9)

        elif atype == "Annular":
            r = float(np.sqrt(x**2 + y**2))
            return self.aperture_r_inner <= r <= outer_r

        elif atype == "Spider":
            r = float(np.sqrt(x**2 + y**2))
            # 1. Outer clear aperture check
            if r > outer_r:
                return False
            # 2. Central hub obscuration check
            if r <= self.aperture_center_radius:
                return False
            # 3. Spider legs check
            n_legs = self.aperture_spider_count
            w = self.aperture_spider_width
            if n_legs > 0 and w > 0:
                half_w = w / 2.0
                phi0 = np.radians(self.aperture_spider_angle)
                for k in range(n_legs):
                    angle = phi0 + 2.0 * np.pi * k / n_legs
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    d_perp = abs(x * sin_a - y * cos_a)
                    d_para = x * cos_a + y * sin_a
                    if d_perp <= half_w and self.aperture_center_radius <= d_para <= outer_r:
                        return False  # Blocked by spider leg
            return True

        # Fallback to circular semi_diameter
        return (x**2 + y**2) <= self.semi_diameter**2

    def check_vignetting(self, x: float, y: float) -> bool:
        """Alias for check_aperture."""
        return self.check_aperture(x, y)

    def get_aperture_params_str(self) -> str:
        """Format aperture parameters into human-readable LDE spreadsheet string."""
        atype = str(self.aperture_type).capitalize() if self.aperture_type else "Circular"
        if atype in ("Point", "None"):
            return "Point (Ø=0)"
        elif atype == "Rectangular":
            hx = self.aperture_x_half if self.aperture_x_half > 0 else self.semi_diameter
            hy = self.aperture_y_half if self.aperture_y_half > 0 else self.semi_diameter
            return f"X={hx}, Y={hy}"
        elif atype == "Elliptical":
            a = self.aperture_a_semi if self.aperture_a_semi > 0 else self.semi_diameter
            b = self.aperture_b_semi if self.aperture_b_semi > 0 else self.semi_diameter
            return f"A={a}, B={b}"
        elif atype == "Annular":
            rout = self.aperture_r_outer if self.aperture_r_outer > 0 else self.semi_diameter
            return f"Rin={self.aperture_r_inner}, Rout={rout}"
        elif atype == "Spider":
            return f"Rcnt={self.aperture_center_radius}, Legs={self.aperture_spider_count}, W={self.aperture_spider_width}, Ang={self.aperture_spider_angle}"
        r = self.aperture_r_outer if self.aperture_r_outer > 0 else self.semi_diameter
        return f"R={r}"

    def set_aperture_params_from_str(self, param_str: str) -> None:
        """Parse key-value string from LDE table delegate into numerical attributes."""
        if not param_str:
            return
        parts = param_str.replace(" ", "").split(",")
        kv = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                try:
                    kv[k.upper()] = float(v)
                except ValueError:
                    pass
        atype = str(self.aperture_type).capitalize() if self.aperture_type else "Circular"
        if atype in ("Point", "None"):
            self.semi_diameter = 0.0
            self.aperture_x_half = 0.0
            self.aperture_y_half = 0.0
            self.aperture_r_inner = 0.0
            self.aperture_r_outer = 0.0
        elif atype == "Circular":
            if "R" in kv:
                self.aperture_r_outer = float(kv["R"])
            elif "ROUT" in kv:
                self.aperture_r_outer = float(kv["ROUT"])
        elif atype == "Rectangular":
            if "X" in kv: self.aperture_x_half = float(kv["X"])
            if "Y" in kv: self.aperture_y_half = float(kv["Y"])
            if "R" in kv and ("X" not in kv or "Y" not in kv):
                r_val = float(kv["R"])
                if "X" not in kv: self.aperture_x_half = r_val
                if "Y" not in kv: self.aperture_y_half = r_val
            hx = self.aperture_x_half if self.aperture_x_half > 0 else self.semi_diameter
            hy = self.aperture_y_half if self.aperture_y_half > 0 else self.semi_diameter
            self.semi_diameter = float(np.hypot(hx, hy))
        elif atype == "Elliptical":
            if "A" in kv: self.aperture_a_semi = float(kv["A"])
            if "B" in kv: self.aperture_b_semi = float(kv["B"])
            if "R" in kv and ("A" not in kv or "B" not in kv):
                r_val = float(kv["R"])
                if "A" not in kv: self.aperture_a_semi = r_val
                if "B" not in kv: self.aperture_b_semi = r_val
            # Removed coupling: semi_diameter remains independent
        elif atype == "Annular":
            if "RIN" in kv: self.aperture_r_inner = float(kv["RIN"])
            if "ROUT" in kv:
                self.aperture_r_outer = float(kv["ROUT"])
                # Removed coupling: semi_diameter remains independent
            elif "R" in kv:
                self.aperture_r_outer = float(kv["R"])
                # Removed coupling: semi_diameter remains independent
        elif atype == "Spider":
            if "RCNT" in kv: self.aperture_center_radius = float(kv["RCNT"])
            if "LEGS" in kv:
                try:
                    self.aperture_spider_count = int(kv["LEGS"])
                except (ValueError, TypeError, OverflowError):
                    pass
            if "W" in kv: self.aperture_spider_width = float(kv["W"])
            if "ANG" in kv: self.aperture_spider_angle = float(kv["ANG"])
            if "R" in kv:
                self.aperture_r_outer = float(kv["R"])
                # Removed coupling: semi_diameter remains independent

    def intersect(self, ray: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Ray-surface intersection for general Surface instance."""
        if self.surface_type == SurfaceType.ASPHERIC or self.conic != 0.0 or (self.asphere_coeffs and len(self.asphere_coeffs) > 0):
            asph = AsphericSurface(radius=self.radius, conic=self.conic, asphere_coeffs=self.asphere_coeffs)
            return asph.intersect(ray)
        else:
            sph = SphericalSurface(radius=self.radius, semi_diameter=self.semi_diameter)
            return sph.intersect(ray)

class SphericalSurface:
    def __init__(self, radius: float = 100.0, semi_diameter: float = 12.7):
        self.radius = radius
        self.semi_diameter = semi_diameter

    def intersect(self, ray: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.radius == 0.0 or np.isinf(self.radius):
            # Flat plane at z=0
            if abs(ray.dir[2]) < 1e-12:
                return None
            t = (0.0 - ray.pos[2]) / ray.dir[2]
            if not np.isfinite(t) or t < -1e-9:
                return None
            hit = ray.pos + max(0.0, t) * ray.dir
            norm = np.array([0.0, 0.0, 1.0])
            return hit, norm

        R = self.radius
        c = np.array([0.0, 0.0, R], dtype=float)
        oc = ray.pos - c
        b = np.dot(ray.dir, oc)
        c_val = np.dot(oc, oc) - R**2
        disc = b**2 - c_val
        if disc < 0:
            return None
        t1 = -b - np.sqrt(disc)
        t2 = -b + np.sqrt(disc)

        # Select physically valid intersection on near lens cap (closest to z=0, avoiding opposite 2R pole)
        candidates = []
        for t_cand in (t1, t2):
            if t_cand >= -1e-12:
                hit = ray.pos + t_cand * ray.dir
                r2 = hit[0]**2 + hit[1]**2
                sd_bound = (self.semi_diameter * 2.0)**2 if self.semi_diameter > 0 else (abs(R) * 2.0)**2
                if r2 <= sd_bound and abs(hit[2]) <= abs(R):
                    candidates.append(t_cand)

        if candidates:
            t = min(candidates)
        else:
            pos_t = [t_val for t_val in (t1, t2) if t_val >= -1e-12]
            if not pos_t:
                return None
            t = min(pos_t)

        hit_pos = ray.pos + t * ray.dir

        # Gradient unit normal
        r = float(np.sqrt(hit_pos[0]**2 + hit_pos[1]**2))
        c_curv = 1.0 / R
        r2 = r**2
        rad = np.sqrt(np.maximum(1e-12, 1.0 - c_curv**2 * r2))
        dz_dr = (c_curv * r) / rad
        if r > 0:
            dz_dx = dz_dr * (hit_pos[0] / r)
            dz_dy = dz_dr * (hit_pos[1] / r)
        else:
            dz_dx = 0.0
            dz_dy = 0.0
        raw_norm = np.array([-dz_dx, -dz_dy, 1.0], dtype=float)
        norm_val = np.linalg.norm(raw_norm)
        normal = raw_norm / norm_val if norm_val > 0 else np.array([0.0, 0.0, 1.0])
        return hit_pos, normal

class AsphericSurface:
    def __init__(self, radius: float = 50.0, conic: float = -1.0, asphere_coeffs: Optional[List[float]] = None):
        self.radius = radius
        self.conic = conic
        self.asphere_coeffs = asphere_coeffs or []

    def sag(self, r: float) -> float:
        if self.radius == 0.0 or np.isinf(self.radius):
            z = 0.0
        else:
            c = 1.0 / self.radius
            r2 = r**2
            arg = 1.0 - (1.0 + self.conic) * c**2 * r2
            denom = 1.0 + np.sqrt(np.maximum(0.0, arg))
            z = c * r2 / denom
        for i, a in enumerate(self.asphere_coeffs):
            z += a * r**(2 * i + 4)
        return float(z)

    def dsag_dr(self, r: float) -> float:
        """Analytical radial derivative dz/dr of aspheric surface sag."""
        if self.radius == 0.0 or np.isinf(self.radius):
            dz_dr_conic = 0.0
        else:
            c = 1.0 / self.radius
            r2 = r**2
            arg = 1.0 - (1.0 + self.conic) * c**2 * r2
            dz_dr_conic = (c * r) / np.sqrt(arg) if arg > 0.0 else 0.0

        dz_dr_poly = 0.0
        for i, a in enumerate(self.asphere_coeffs):
            power = 2 * i + 4
            dz_dr_poly += power * a * (r**(power - 1)) if r > 0 else 0.0
        return float(dz_dr_conic + dz_dr_poly)

    def normal(self, hit_pos: np.ndarray) -> np.ndarray:
        """Calculate exact 3D gradient unit normal vector at hit position."""
        r = float(np.sqrt(hit_pos[0]**2 + hit_pos[1]**2))
        if r < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        dz_dr = self.dsag_dr(r)
        dz_dx = dz_dr * (hit_pos[0] / r)
        dz_dy = dz_dr * (hit_pos[1] / r)
        grad = np.array([-dz_dx, -dz_dy, 1.0], dtype=float)
        norm = np.linalg.norm(grad)
        return grad / norm if norm > 0 else np.array([0.0, 0.0, 1.0], dtype=float)

    def normal_vector(self, x: float, y: float) -> np.ndarray:
        return self.normal(np.array([x, y, 0.0]))

    def intersect(self, ray: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Newton-Raphson 3D ray-asphere intersection solver."""
        P0 = np.array(ray.pos, dtype=float)
        D = np.array(ray.dir, dtype=float)
        if abs(D[2]) < 1e-12 and D[0] == 0.0 and D[1] == 0.0:
            return None

        # Initial parameter t0 (intersection with flat plane at z = sag(0))
        t = (self.sag(0.0) - P0[2]) / D[2] if abs(D[2]) > 1e-12 else 0.0

        if self.radius != 0.0 and not np.isinf(self.radius):
            c = 1.0 / self.radius
            k = self.conic
            A_quad = (1.0 + k) * c * D[2]**2 + c * (D[0]**2 + D[1]**2)
            B_quad = 2.0 * (1.0 + k) * c * P0[2] * D[2] - 2.0 * D[2] + 2.0 * c * (P0[0] * D[0] + P0[1] * D[1])
            C_quad = (1.0 + k) * c * P0[2]**2 - 2.0 * P0[2] + c * (P0[0]**2 + P0[1]**2)
            
            if abs(A_quad) > 1e-12:
                disc = B_quad**2 - 4.0 * A_quad * C_quad
                if disc >= 0:
                    sq_disc = np.sqrt(disc)
                    t_c1 = (-B_quad - sq_disc) / (2.0 * A_quad)
                    t_c2 = (-B_quad + sq_disc) / (2.0 * A_quad)
                    pos_roots = [r for r in [t_c1, t_c2] if r > 1e-9]
                    if pos_roots:
                        t = min(pos_roots)
            elif abs(B_quad) > 1e-12:
                t_cand = -C_quad / B_quad
                if t_cand > 1e-9:
                    t = t_cand
            else:
                sph = SphericalSurface(radius=self.radius)
                sph_res = sph.intersect(ray)
                if sph_res is not None:
                    hit_sph = sph_res[0]
                    t_sph = float(np.linalg.norm(hit_sph - P0))
                    if D[2] < 0:
                        t_sph = -t_sph
                    if t_sph > 0:
                        t = t_sph

        max_iter = 30
        tol = 1e-9

        for _ in range(max_iter):
            P_t = P0 + t * D
            r_t = float(np.sqrt(P_t[0]**2 + P_t[1]**2))
            z_sag = self.sag(r_t)
            f_val = P_t[2] - z_sag

            if abs(f_val) < tol:
                hit_pos = P_t
                return hit_pos, self.normal(hit_pos)

            dz_dr = self.dsag_dr(r_t)
            dr_dt = (P_t[0] * D[0] + P_t[1] * D[1]) / r_t if r_t > 1e-12 else 0.0
            df_dt = D[2] - dz_dr * dr_dt

            if abs(df_dt) < 1e-12:
                break

            t_next = t - f_val / df_dt
            if abs(t_next - t) < 1e-10:
                t = t_next
                P_t = P0 + t * D
                return P_t, self.normal(P_t)
            t = t_next

        P_t = P0 + t * D
        r_t = float(np.sqrt(P_t[0]**2 + P_t[1]**2))
        if abs(P_t[2] - self.sag(r_t)) < 1e-6:
            return P_t, self.normal(P_t)
        return None
