"""
Spectrometer Data Models, Optical System State & Trace Result Containers.
Provides SpectrometerSystem, SurfaceStack, Surface, WavelengthConfig, SourceConfig,
FastTraceResult, and FullTraceResult according to V2 architecture specifications.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import os
import numpy as np

try:
    from optics_app.engine.surface import Surface, SurfaceType
    from optics_app.engine.lde import SurfaceStack
except ImportError:
    from v2.engine.surface import Surface, SurfaceType
    from v2.engine.lde import SurfaceStack


@dataclass
class WavelengthConfig:
    """Spectral line / wavelength configuration with display styling and weight."""
    wavelength_um: float
    weight: float = 1.0
    is_primary: bool = False
    color: str = "#800000"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wavelength_um": float(self.wavelength_um),
            "weight": float(self.weight),
            "is_primary": bool(self.is_primary),
            "color": str(self.color)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WavelengthConfig":
        return cls(
            wavelength_um=float(data.get("wavelength_um", 0.5876)),
            weight=float(data.get("weight", 1.0)),
            is_primary=bool(data.get("is_primary", False)),
            color=str(data.get("color", "#800000"))
        )


@dataclass
class SourceConfig:
    """Spectrometer optical source and entrance aperture definition."""
    source_type: str = "Collimated"
    beam_radius: float = 5.0
    divergence_angle: float = 10.0
    num_rays: int = 7
    wavelengths: str = "1.100, 1.225, 1.350"
    input_aperture_shape: str = "Circular"
    input_aperture_width: float = 0.03
    input_aperture_height: float = 1.0
    numerical_aperture: float = 0.12

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "beam_radius": float(self.beam_radius),
            "divergence_angle": float(self.divergence_angle),
            "num_rays": int(self.num_rays),
            "wavelengths": str(self.wavelengths),
            "input_aperture_shape": self.input_aperture_shape,
            "input_aperture_width": float(self.input_aperture_width),
            "input_aperture_height": float(self.input_aperture_height),
            "numerical_aperture": float(self.numerical_aperture)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceConfig":
        return cls(
            source_type=str(data.get("source_type", "Collimated")),
            beam_radius=float(data.get("beam_radius", 5.0)),
            divergence_angle=float(data.get("divergence_angle", 10.0)),
            num_rays=int(data.get("num_rays", 7)),
            wavelengths=str(data.get("wavelengths", "1.100, 1.225, 1.350")),
            input_aperture_shape=str(data.get("input_aperture_shape", "Circular")),
            input_aperture_width=float(data.get("input_aperture_width", 0.03)),
            input_aperture_height=float(data.get("input_aperture_height", 1.0)),
            numerical_aperture=float(data.get("numerical_aperture", 0.12))
        )


@dataclass
class FastTraceResult:
    """Lightweight trace result for interactive slider feedback (<50ms)."""
    ray_paths: List[np.ndarray] = field(default_factory=list)
    spot_coordinates: Dict[float, np.ndarray] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    wavelengths: List[float] = field(default_factory=list)
    vignetted_count: int = 0
    total_rays: int = 0


@dataclass
class FullTraceResult:
    """Comprehensive high-density trace result for release/committed analysis."""
    ray_paths: List[np.ndarray] = field(default_factory=list)
    spot_coordinates: Dict[float, np.ndarray] = field(default_factory=dict)
    rms_radius: float = 0.0
    geo_radius: float = 0.0
    airy_radius: float = 0.0
    dispersion_spread: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    elapsed_ms: float = 0.0
    wavelengths: List[float] = field(default_factory=list)
    vignetted_count: int = 0
    total_rays: int = 0
    spot_diagrams_by_wl: Dict[float, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class SpectrometerSystem:
    """
    Core Spectrometer System model encapsulating SurfaceStack, spectral config,
    grating/slit/detector parameters, and dynamic optical element operations.
    """
    surface_stack: SurfaceStack = field(default_factory=SurfaceStack)
    wavelengths: List[float] = field(default_factory=lambda: [1.100, 1.225, 1.350])
    primary_wavelength: float = 1.225
    slit_width: float = 30.0  # um
    grating_surface_idx: int = 3
    detector_surface_idx: int = 16
    detector_pixel_size_um: float = 800.0  # 0.8 mm detector
    system_config: Optional[Dict[str, Any]] = None
    source_config: Optional[Union[SourceConfig, Dict[str, Any]]] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    analysis_config: Optional[Dict[str, Any]] = None
    name: str = "NIR Fingerprint Spectrometer"
    version: int = 5

    def __post_init__(self):
        self.auto_detect_indices()

    @property
    def surfaces(self) -> List[Surface]:
        return self.surface_stack.surfaces

    @property
    def components(self) -> List[Any]:
        return getattr(self.surface_stack, "components", [])

    @components.setter
    def components(self, val: List[Any]) -> None:
        self.surface_stack.components = val

    @property
    def stack(self) -> SurfaceStack:
        return self.surface_stack

    def __len__(self) -> int:
        return len(self.surface_stack.surfaces)

    def __getitem__(self, idx: int) -> Surface:
        return self.surface_stack.surfaces[idx]

    def auto_detect_indices(self) -> None:
        """Automatically identify key optical components in the surface stack."""
        surfs = self.surface_stack.surfaces
        if not surfs:
            return

        # 1. Grating Surface
        grating_found = False
        for i, s in enumerate(surfs):
            if s.is_air_space:
                continue
            if s.surface_type == SurfaceType.GRATING or getattr(s, "grating_lines_per_mm", 0.0) > 0 or "GRATING" in s.name.upper():
                self.grating_surface_idx = i
                grating_found = True
                break
        if not grating_found and len(surfs) > 3:
            self.grating_surface_idx = 3

        # 2. Detector Surface
        detector_found = False
        for i in reversed(range(len(surfs))):
            s = surfs[i]
            if s.surface_type == SurfaceType.DETECTOR or "DETECTOR" in s.name.upper() or "IMA" in s.name.upper():
                self.detector_surface_idx = i
                detector_found = True
                break
        if not detector_found:
            self.detector_surface_idx = len(surfs) - 1

        # 3. Slit width & detector dimensions
        if self.system_config and isinstance(self.system_config, dict):
            ap = self.system_config.get("aperture", {})
            if "slit_width_um" in ap:
                self.slit_width = float(ap["slit_width_um"])

        if 0 <= self.detector_surface_idx < len(surfs):
            det_surf = surfs[self.detector_surface_idx]
            if det_surf.semi_diameter > 0:
                self.detector_pixel_size_um = float(det_surf.semi_diameter * 2000.0)

    @classmethod
    def load_from_json(cls, filepath: str) -> "SpectrometerSystem":
        """
        Load spectrometer prescription JSON file without loss of precision.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Spectrometer JSON file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpectrometerSystem":
        """Instantiate SpectrometerSystem from dictionary data."""
        stack = SurfaceStack()
        surfaces_raw = data.get("surfaces", [])

        for d in surfaces_raw:
            st = d.get("surface_type")
            if isinstance(st, str):
                for val in [SurfaceType.OBJECT, SurfaceType.STANDARD, SurfaceType.MIRROR, SurfaceType.ASPHERIC, SurfaceType.STOP, SurfaceType.GRATING, SurfaceType.DETECTOR, SurfaceType.CAD, SurfaceType.AIR_SPACE]:
                    if val.upper() == st.upper():
                        d["surface_type"] = val
                        break
            clean_d = {k: v for k, v in d.items() if k not in ("global_origin", "global_R", "cad_profile")}
            try:
                surface = Surface.from_dict(clean_d) if hasattr(Surface, "from_dict") else Surface(**clean_d)
                if surface.is_air_space:
                    surface.surface_type = SurfaceType.AIR_SPACE
                stack.add_surface(surface)
            except Exception:
                s = Surface()
                for k, v in clean_d.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
                stack.add_surface(s)

        wavelengths = data.get("wavelengths", [1.100, 1.225, 1.350])
        primary_wl = data.get("primary_wavelength", wavelengths[len(wavelengths)//2] if wavelengths else 1.225)
        sys_cfg = data.get("system_config")
        if sys_cfg and isinstance(sys_cfg, dict) and "wavelengths" in sys_cfg:
            for w in sys_cfg["wavelengths"]:
                if isinstance(w, dict) and w.get("is_primary"):
                    primary_wl = float(w.get("wavelength_um", primary_wl))
                    break
        stack.wavelengths = wavelengths
        stack.primary_wavelength = float(primary_wl)
        stack._update_accumulated_positions()

        sys_cfg = data.get("system_config")
        src_cfg = data.get("source_config")
        opt_cfg = data.get("optimizer_config")
        ana_cfg = data.get("analysis_config")
        if isinstance(data.get("viewer_settings"), dict):
            stack.viewer_settings = data["viewer_settings"]
        if data.get("optical_routes") is not None:
            from optics_app.engine.routes import OpticalRouteSet
            stack.optical_routes = OpticalRouteSet.from_dict(data.get("optical_routes"))
        if data.get("two_photon_config") is not None:
            from optics_app.analysis.two_photon import TwoPhotonMicroscopeConfig
            stack.two_photon_config = TwoPhotonMicroscopeConfig.from_dict(data.get("two_photon_config"))
        if data.get("nonlinear_imaging_config") is not None:
            from optics_app.analysis.nonlinear_imaging import NonlinearImagingConfig
            stack.nonlinear_imaging_config = NonlinearImagingConfig.from_dict(data.get("nonlinear_imaging_config"))
        if data.get("srs_microscope_config") is not None:
            try:
                from optics_app.analysis.ultrafast_nonlinear import ThorlabsSRSMicroscopeConfig
            except ImportError:
                from v2.analysis.ultrafast_nonlinear import ThorlabsSRSMicroscopeConfig
            stack.srs_microscope_config = ThorlabsSRSMicroscopeConfig.from_dict(data.get("srs_microscope_config"))

        components_raw = data.get("components", [])
        if components_raw:
            try:
                from optics_app.engine.components import OpticalComponent
            except ImportError:
                from v2.engine.components import OpticalComponent
            stack.components = []
            for cd in components_raw:
                try:
                    comp = OpticalComponent.from_dict(cd)
                    stack.components.append(comp)
                except Exception:
                    stack.components.append(cd)
            if hasattr(stack, "bind_components_to_surfaces"):
                stack.bind_components_to_surfaces()

        system = cls(
            surface_stack=stack,
            wavelengths=list(wavelengths),
            primary_wavelength=float(primary_wl),
            system_config=sys_cfg,
            source_config=src_cfg,
            optimizer_config=opt_cfg,
            analysis_config=ana_cfg,
            name=str(data.get("name", "Spectrometer")),
            version=int(data.get("version", 5))
        )
        return system

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete system preserving schema version."""
        import dataclasses
        surfaces_data = []
        for s in self.surface_stack.surfaces:
            sd = s.to_dict() if hasattr(s, "to_dict") else dataclasses.asdict(s)
            sd["cad_profile"] = None
            if "global_origin" in sd:
                del sd["global_origin"]
            if "global_R" in sd:
                del sd["global_R"]
            surfaces_data.append(sd)

        components_data = self.surface_stack.extract_components_dict() if hasattr(self.surface_stack, "extract_components_dict") else []

        data: Dict[str, Any] = {
            "version": getattr(self, "version", 5),
            "name": self.name,
            "surfaces": surfaces_data,
            "components": components_data,
            "wavelengths": self.wavelengths,
            "primary_wavelength": self.primary_wavelength
        }

        if self.system_config is not None:
            data["system_config"] = self.system_config
        if self.source_config is not None:
            if hasattr(self.source_config, "to_dict"):
                data["source_config"] = self.source_config.to_dict()
            else:
                data["source_config"] = self.source_config
        if self.optimizer_config is not None:
            data["optimizer_config"] = self.optimizer_config
        if self.analysis_config is not None:
            data["analysis_config"] = self.analysis_config
        viewer_settings = getattr(self.surface_stack, "viewer_settings", None)
        if isinstance(viewer_settings, dict):
            data["viewer_settings"] = viewer_settings
        routes = getattr(self.surface_stack, "optical_routes", None)
        if routes is not None:
            data["optical_routes"] = routes.to_dict() if hasattr(routes, "to_dict") else routes
        two_photon = getattr(self.surface_stack, "two_photon_config", None)
        if two_photon is not None:
            data["two_photon_config"] = two_photon.to_dict() if hasattr(two_photon, "to_dict") else two_photon
        nonlinear = getattr(self.surface_stack, "nonlinear_imaging_config", None)
        if nonlinear is not None:
            data["nonlinear_imaging_config"] = nonlinear.to_dict() if hasattr(nonlinear, "to_dict") else nonlinear
        srs_cfg = getattr(self.surface_stack, "srs_microscope_config", None)
        if srs_cfg is not None:
            data["srs_microscope_config"] = srs_cfg.to_dict() if hasattr(srs_cfg, "to_dict") else srs_cfg

        return data

    def to_json(self, filepath: Optional[str] = None, indent: int = 4) -> str:
        """Serialize spectrometer system state to JSON string or save to file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, source: Union[str, Dict[str, Any]]) -> "SpectrometerSystem":
        """Deserialize SpectrometerSystem from JSON string, dictionary, or file path."""
        if isinstance(source, dict):
            return cls.from_dict(source)
        if isinstance(source, str):
            if os.path.exists(source):
                return cls.load_from_json(source)
            data = json.loads(source)
            return cls.from_dict(data)
        raise ValueError(f"Unsupported source type for from_json: {type(source)}")

    def save_to_json(self, filepath: str) -> None:
        """Save spectrometer system state to JSON file."""
        self.to_json(filepath=filepath, indent=4)

    # Dynamic Optical Element Manipulations
    def insert_element(self, after_surface_idx: int, element: Union[SurfaceStack, List[Surface]]) -> List[int]:
        """Insert optical element surfaces after specified surface index."""
        indices = self.surface_stack.insert_element(after_surface_idx, element)
        self.auto_detect_indices()
        return indices

    def insert_catalog_lens(self, after_surface_idx: int, part_number: str, air_gap_after: float = 20.0) -> List[int]:
        """Insert a Thorlabs catalog singlet/doublet lens after specified surface index."""
        indices = self.surface_stack.insert_catalog_lens(after_surface_idx, part_number, air_gap_after=air_gap_after)
        self.auto_detect_indices()
        return indices

    def delete_element(self, element_idx: Union[int, str] = None, target: Any = None, **kwargs) -> List[int]:
        """Delete optical element and merge its glass thickness into preceding air gap."""
        if element_idx is None:
            element_idx = target
        if element_idx is None:
            return []
        indices = self.surface_stack.delete_element(element_idx)
        self.auto_detect_indices()
        return indices

    delete_lens_element = delete_element

    def flip_element(self, element_idx: Union[int, str]) -> List[int]:
        """Flip optical element front-to-back, inverting curvatures and shifting glass thicknesses."""
        indices = self.surface_stack.flip_element(element_idx)
        self.auto_detect_indices()
        return indices

    def rotate_element(
        self,
        element_idx: Union[int, str] = None,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        tilt_z: float = 0.0,
        pivot: Union[str, Tuple, List, np.ndarray, float] = "front_vertex",
        front_surface_index: Optional[int] = None,
        **kwargs
    ) -> None:
        """Perform 3D rigid body rotation of element around specified pivot point."""
        if element_idx is None:
            element_idx = front_surface_index
        if element_idx is None:
            return
        self.surface_stack.rotate_element(element_idx, tilt_x=tilt_x, tilt_y=tilt_y, tilt_z=tilt_z, pivot=pivot, **kwargs)
        self.auto_detect_indices()

    rotate_lens_element = rotate_element

    def replace_element(self, element_idx: Union[int, str], new_element: Union[SurfaceStack, List[Surface]]) -> List[int]:
        """Replace active lens element with a new element in sequence."""
        indices = self.surface_stack.replace_element(element_idx, new_element)
        self.auto_detect_indices()
        return indices

    def displace_element(
        self,
        element_idx: Union[int, str],
        decenter_x: float = 0.0,
        decenter_y: float = 0.0,
        decenter_z: float = 0.0
    ) -> None:
        """Displace/translate rigid optical element in 3D."""
        self.surface_stack.displace_element(element_idx, decenter_x=decenter_x, decenter_y=decenter_y, decenter_z=decenter_z)
        self.auto_detect_indices()

    def tilt_element(
        self,
        element_idx: Union[int, str],
        angle_deg: float = 0.0,
        axis: str = "x",
        tilt_x: Optional[float] = None,
        tilt_y: Optional[float] = None,
        tilt_z: Optional[float] = None,
        pivot: Union[str, Tuple, List, np.ndarray, float] = "front_vertex"
    ) -> None:
        """Tilt rigid optical element around chosen axis or 3D angles."""
        self.surface_stack.tilt_element(
            element_idx,
            angle_deg=angle_deg,
            axis=axis,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
            pivot=pivot
        )
        self.auto_detect_indices()

    def offset_element(
        self,
        element_idx: Union[int, str],
        reference_surface_idx: Union[int, str],
        offset_bias_mm: float,
        axis: str = "z"
    ) -> None:
        """Move/offset rigid optical element relative to reference surface with an offset bias."""
        self.surface_stack.offset_element(
            element_idx,
            reference_surface_idx=reference_surface_idx,
            offset_bias_mm=offset_bias_mm,
            axis=axis
        )
        self.auto_detect_indices()

    move_to_surface_offset = offset_element

    def replace_element_with_catalog(self, element_idx: Union[int, str], part_number: str) -> List[int]:
        """Replace optical element with a Thorlabs catalog component."""
        indices = self.surface_stack.replace_element_with_catalog(element_idx, part_number)
        self.auto_detect_indices()
        return indices
