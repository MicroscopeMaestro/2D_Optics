"""
Lens Data Editor (LDE) Sequential Surface Stack Management.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import uuid
try:
    from optics_app.engine.surface import Surface, SurfaceType
except ImportError:
    from v2.engine.surface import Surface, SurfaceType


def build_rotation_matrix(tilt_x_deg: float, tilt_y_deg: float, tilt_z_deg: float) -> np.ndarray:
    """Compute composite 3D Euler rotation matrix R = Rz(rz) @ Ry(ry) @ Rx(rx)."""
    rx = np.radians(tilt_x_deg)
    ry = np.radians(tilt_y_deg)
    rz = np.radians(tilt_z_deg)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rx), -np.sin(rx)],
        [0.0, np.sin(rx), np.cos(rx)]
    ], dtype=float)

    Ry = np.array([
        [np.cos(ry), 0.0, np.sin(ry)],
        [0.0, 1.0, 0.0],
        [-np.sin(ry), 0.0, np.cos(ry)]
    ], dtype=float)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    return Rz @ Ry @ Rx


class SurfaceStack:
    def __init__(self):
        self.surfaces: List[Surface] = []
        self.components: List[Any] = []

    @classmethod
    def create_clean_system(cls, object_dist: float = 100.0, stop_dist: float = 50.0, stop_semi_diameter: float = 10.0) -> 'SurfaceStack':
        """
        Create a clean default optical system with Object, Stop, and Image surfaces,
        matching Zemax OpticStudio default project conventions.
        - Surface 0: S0 (OBJ) [Object plane]
        - Surface 1: S1 (STO) [Aperture Stop]
        - Surface 2: S2 (IMA) [Image / Detector plane]
        """
        stack = cls()
        s0 = Surface(
            name="S0 (OBJ)",
            surface_type=SurfaceType.OBJECT,
            radius=0.0,
            thickness=float(object_dist),
            glass="AIR",
            semi_diameter=25.0,
            is_stop=False
        )
        s1 = Surface(
            name="S1 (STO)",
            surface_type=SurfaceType.STOP,
            radius=0.0,
            thickness=float(stop_dist),
            glass="AIR",
            semi_diameter=float(stop_semi_diameter),
            is_stop=True
        )
        s2 = Surface(
            name="S2 (IMA)",
            surface_type=SurfaceType.DETECTOR,
            radius=0.0,
            thickness=0.0,
            glass="AIR",
            semi_diameter=25.0,
            is_stop=False
        )
        stack.add_surface(s0)
        stack.add_surface(s1)
        stack.add_surface(s2)
        stack._update_accumulated_positions()
        return stack

    @classmethod
    def create_default(cls, object_dist: float = 100.0, stop_dist: float = 50.0, stop_semi_diameter: float = 10.0) -> 'SurfaceStack':
        """Alias for create_clean_system."""
        return cls.create_clean_system(object_dist=object_dist, stop_dist=stop_dist, stop_semi_diameter=stop_semi_diameter)


    def add_component(self, component: Any) -> None:
        """Add an optical component to the stack."""
        self.components.append(component)

    def extract_components_dict(self) -> List[Dict[str, Any]]:
        """
        Extract serialized component dictionaries.
        If self.components is populated, serialize them.
        Also inspect surfaces and synthesize components for any
        galvos, beamsplitters, prisms, or active mechanisms not already represented.
        """
        extracted = []
        seen_uids = set()
        seen_surface_names = set()

        if self.components:
            for c in self.components:
                if hasattr(c, "to_dict"):
                    c_dict = c.to_dict()
                elif isinstance(c, dict):
                    c_dict = dict(c)
                else:
                    c_dict = {}
                extracted.append(c_dict)
                uid = c_dict.get("component_uid")
                if uid:
                    seen_uids.add(uid)
                for suid in c_dict.get("surface_uids", []):
                    seen_surface_names.add(suid)

        for idx, s in enumerate(self.surfaces):
            c_uid = getattr(s, "component_uid", "") or f"comp_surf_{idx}"
            s_name = getattr(s, "name", f"surf_{idx}")

            galvo_cfg = getattr(s, "galvo_config", None)
            bs_cfg = getattr(s, "beamsplitter_config", None)
            prism_cfg = getattr(s, "prism_config", None)
            aom_cfg = getattr(s, "aom_config", None)
            eom_cfg = getattr(s, "eom_config", None)
            plm_cfg = getattr(s, "plm_config", None)
            is_bs = getattr(s, "is_beam_splitter", False)
            mech = getattr(s, "mechanism", None)

            # Check if this surface or component is already represented
            already_represented = (
                (c_uid in seen_uids)
                or (s_name in seen_surface_names)
                or (getattr(s, "component_uid", "") and getattr(s, "component_uid", "") in seen_surface_names)
            )

            if mech is not None and not already_represented:
                seen_uids.add(c_uid)
                seen_surface_names.add(s_name)
                extracted.append({
                    "component_uid": c_uid,
                    "name": s.name,
                    "component_type": "ActiveMechanism",
                    "surface_uids": [s_name],
                    "mechanism": mech.to_dict() if hasattr(mech, "to_dict") else (dict(mech) if isinstance(mech, dict) else {}),
                })
            elif not self.components:
                if c_uid in seen_uids:
                    continue
                if galvo_cfg is not None:
                    seen_uids.add(c_uid)
                    seen_surface_names.add(s_name)
                    extracted.append({
                        "component_uid": c_uid,
                        "name": s.name,
                        "component_type": "galvo",
                        "surface_uids": [s_name],
                        "galvo_config": galvo_cfg.to_dict() if hasattr(galvo_cfg, "to_dict") else (dict(galvo_cfg) if isinstance(galvo_cfg, dict) else {}),
                    })
                elif bs_cfg is not None or is_bs:
                    seen_uids.add(c_uid)
                    seen_surface_names.add(s_name)
                    extracted.append({
                        "component_uid": c_uid,
                        "name": s.name,
                        "component_type": "beamsplitter",
                        "surface_uids": [s_name],
                        "beamsplitter_config": bs_cfg.to_dict() if (bs_cfg and hasattr(bs_cfg, "to_dict")) else (bs_cfg or {}),
                    })
                elif prism_cfg is not None:
                    seen_uids.add(c_uid)
                    seen_surface_names.add(s_name)
                    extracted.append({
                        "component_uid": c_uid,
                        "name": s.name,
                        "component_type": "prism",
                        "surface_uids": [s_name],
                        "prism_config": prism_cfg.to_dict() if hasattr(prism_cfg, "to_dict") else (dict(prism_cfg) if isinstance(prism_cfg, dict) else {}),
                    })
                elif aom_cfg is not None:
                    seen_uids.add(c_uid)
                    seen_surface_names.add(s_name)
                    extracted.append({
                        "component_uid": c_uid,
                        "name": s.name,
                        "component_type": "aom",
                        "surface_uids": [s_name],
                        "aom_config": aom_cfg.to_dict() if hasattr(aom_cfg, "to_dict") else (dict(aom_cfg) if isinstance(aom_cfg, dict) else {}),
                    })
                elif eom_cfg is not None:
                    seen_uids.add(c_uid)
                    seen_surface_names.add(s_name)
                    extracted.append({
                        "component_uid": c_uid,
                        "name": s.name,
                        "component_type": "eom",
                        "surface_uids": [s_name],
                        "eom_config": eom_cfg.to_dict() if hasattr(eom_cfg, "to_dict") else (dict(eom_cfg) if isinstance(eom_cfg, dict) else {}),
                    })
                elif plm_cfg is not None:
                    seen_uids.add(c_uid)
                    seen_surface_names.add(s_name)
                    extracted.append({
                        "component_uid": c_uid,
                        "name": s.name,
                        "component_type": "plm",
                        "surface_uids": [s_name],
                        "plm_config": plm_cfg.to_dict() if hasattr(plm_cfg, "to_dict") else (dict(plm_cfg) if isinstance(plm_cfg, dict) else {}),
                    })
        return extracted

    def bind_components_to_surfaces(self) -> None:
        """
        Bind loaded component configurations to their corresponding surfaces.
        """
        if not self.components:
            return
        for comp in self.components:
            c_uid = comp.get("component_uid") if isinstance(comp, dict) else getattr(comp, "component_uid", "")
            c_type = comp.get("component_type") if isinstance(comp, dict) else getattr(comp, "component_type", "")
            s_uids = comp.get("surface_uids", []) if isinstance(comp, dict) else getattr(comp, "surface_uids", [])
            galvo = comp.get("galvo_config") if isinstance(comp, dict) else getattr(comp, "galvo_config", None)
            bs = comp.get("beamsplitter_config") if isinstance(comp, dict) else getattr(comp, "beamsplitter_config", None)
            prism = comp.get("prism_config") if isinstance(comp, dict) else getattr(comp, "prism_config", None)
            aom = comp.get("aom_config") if isinstance(comp, dict) else getattr(comp, "aom_config", None)
            eom = comp.get("eom_config") if isinstance(comp, dict) else getattr(comp, "eom_config", None)
            plm = comp.get("plm_config") if isinstance(comp, dict) else getattr(comp, "plm_config", None)

            mech = None
            raw_mech = comp.get("mechanism") if isinstance(comp, dict) else getattr(comp, "mechanism", None)
            if raw_mech is not None:
                if isinstance(raw_mech, dict):
                    try:
                        from optics_app.engine.components import ActiveMechanism
                        mech = ActiveMechanism.from_dict(raw_mech)
                    except Exception:
                        try:
                            from v2.engine.components import ActiveMechanism
                            mech = ActiveMechanism.from_dict(raw_mech)
                        except Exception:
                            mech = raw_mech
                else:
                    mech = raw_mech
            elif isinstance(comp, dict) and ("channels" in comp or "mechanism_id" in comp):
                try:
                    from optics_app.engine.components import ActiveMechanism
                    mech = ActiveMechanism.from_dict(comp)
                except Exception:
                    try:
                        from v2.engine.components import ActiveMechanism
                        mech = ActiveMechanism.from_dict(comp)
                    except Exception:
                        mech = comp
            elif hasattr(comp, "channels"):
                mech = comp

            mech_target_uids = []
            if mech is not None and hasattr(mech, "channels") and mech.channels:
                for ch in mech.channels.values():
                    t_uid = getattr(ch, "target_surface_uid", "")
                    if t_uid:
                        mech_target_uids.append(t_uid)

            for s in self.surfaces:
                matches = (
                    (c_uid and s.component_uid == c_uid)
                    or (s.name in s_uids)
                    or (s.component_uid and s.component_uid in s_uids)
                    or (s.name in mech_target_uids)
                    or (s.component_uid and s.component_uid in mech_target_uids)
                )
                if matches:
                    if c_uid and not s.component_uid:
                        s.component_uid = c_uid
                    if c_type and not getattr(s, "component_type", ""):
                        s.component_type = c_type
                    if galvo is not None and getattr(s, "galvo_config", None) is None:
                        s.galvo_config = galvo
                        s.mechanism_type = "GALVO"
                    if bs is not None and getattr(s, "beamsplitter_config", None) is None:
                        s.beamsplitter_config = bs
                        s.is_beam_splitter = True
                    if prism is not None and getattr(s, "prism_config", None) is None:
                        s.prism_config = prism
                    if aom is not None and getattr(s, "aom_config", None) is None:
                        if isinstance(aom, dict):
                            try:
                                from optics_app.engine.components import AOMConfig
                                aom = AOMConfig.from_dict(aom)
                            except Exception:
                                try:
                                    from v2.engine.components import AOMConfig
                                    aom = AOMConfig.from_dict(aom)
                                except Exception:
                                    pass
                        s.aom_config = aom
                        s.mechanism_type = "AOM"
                    if eom is not None and getattr(s, "eom_config", None) is None:
                        if isinstance(eom, dict):
                            try:
                                from optics_app.engine.components import EOMConfig
                                eom = EOMConfig.from_dict(eom)
                            except Exception:
                                try:
                                    from v2.engine.components import EOMConfig
                                    eom = EOMConfig.from_dict(eom)
                                except Exception:
                                    pass
                        s.eom_config = eom
                        s.mechanism_type = "EOM"
                    if plm is not None and getattr(s, "plm_config", None) is None:
                        if isinstance(plm, dict):
                            try:
                                from optics_app.engine.plm import PLMConfig
                                plm = PLMConfig.from_dict(plm)
                            except Exception:
                                try:
                                    from v2.engine.plm import PLMConfig
                                    plm = PLMConfig.from_dict(plm)
                                except Exception:
                                    pass
                        s.plm_config = plm
                        s.is_plm = True
                        s.mechanism_type = "PLM"
                    if mech is not None:
                        s.mechanism = mech
                        s.component_type = "ActiveMechanism"

    def save_to_json(
        self,
        filepath: str,
        wavelengths: Optional[List[float]] = None,
        system_config: Optional[Any] = None,
        optimizer_config: Optional[Any] = None,
        analysis_config: Optional[Any] = None,
        version: Optional[int] = None
    ) -> None:
        import json
        import dataclasses
        surfaces_data = [s.to_dict() if hasattr(s, "to_dict") else dataclasses.asdict(s) for s in self.surfaces]
        for d in surfaces_data:
            if "cad_profile" in d:
                d["cad_profile"] = None
            if "global_origin" in d:
                del d["global_origin"]
            if "global_R" in d:
                del d["global_R"]
                
        sys_cfg = system_config if system_config is not None else getattr(self, 'system_config', None)
        if wavelengths:
            project_wavelengths = list(wavelengths)
        elif sys_cfg is not None:
            if isinstance(sys_cfg, dict):
                project_wavelengths = [float(w.get("wavelength_um", 0.0)) for w in sys_cfg.get("wavelengths", [])]
            else:
                project_wavelengths = list(getattr(sys_cfg, "wavelength_list_um", []) or [])
        else:
            project_wavelengths = []
        if not project_wavelengths:
            project_wavelengths = [float(getattr(self, "primary_wavelength", 0.58756))]
        primary = float(getattr(
            self, "primary_wavelength",
            getattr(sys_cfg, "primary_wavelength_um", project_wavelengths[0]),
        ))

        components_data = self.extract_components_dict()
        if version is not None:
            v_num = int(version)
        elif not components_data and getattr(self, "two_photon_config", None) is not None:
            v_num = 4
        else:
            v_num = 5

        data = {
            "version": v_num,
            "surfaces": surfaces_data,
            "components": components_data,
            "wavelengths": project_wavelengths,
            "primary_wavelength": primary,
        }

        # 1. System Explorer / System Config
        if sys_cfg is not None:
            if hasattr(sys_cfg, 'to_dict'):
                data['system_config'] = sys_cfg.to_dict()
            elif isinstance(sys_cfg, dict):
                data['system_config'] = sys_cfg

        # 2. Source Configuration
        if hasattr(self, 'source_config') and self.source_config is not None:
            if hasattr(self.source_config, 'to_dict'):
                data['source_config'] = self.source_config.to_dict()
            elif isinstance(self.source_config, dict):
                data['source_config'] = self.source_config

        # 3. Optimizer Configuration
        opt_cfg = optimizer_config if optimizer_config is not None else getattr(self, 'optimizer_config', None)
        if opt_cfg is not None:
            if hasattr(opt_cfg, 'to_dict'):
                data['optimizer_config'] = opt_cfg.to_dict()
            elif isinstance(opt_cfg, dict):
                data['optimizer_config'] = opt_cfg

        # 4. Analysis Configuration
        if analysis_config is not None:
            data['analysis_config'] = analysis_config

        viewer_settings = getattr(self, 'viewer_settings', None)
        if isinstance(viewer_settings, dict):
            data['viewer_settings'] = viewer_settings

        # 5. Optional branched-instrument configuration.  Legacy projects omit
        # these keys and continue to behave as ordinary sequential stacks.
        routes = getattr(self, "optical_routes", None)
        if routes is not None:
            data["optical_routes"] = routes.to_dict() if hasattr(routes, "to_dict") else routes
        two_photon = getattr(self, "two_photon_config", None)
        if two_photon is not None:
            data["two_photon_config"] = two_photon.to_dict() if hasattr(two_photon, "to_dict") else two_photon
        nonlinear = getattr(self, "nonlinear_imaging_config", None)
        if nonlinear is not None:
            data["nonlinear_imaging_config"] = nonlinear.to_dict() if hasattr(nonlinear, "to_dict") else nonlinear

        spectrometer = getattr(self, "spectrometer_calibration", None)
        if spectrometer is not None:
            data["spectrometer_calibration"] = spectrometer

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from_json(cls, filepath: str) -> Dict[str, Any]:
        import json
        stack = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        wavelengths = [0.450, 0.546, 0.650]
        surfaces_data = []
        if isinstance(data, dict):
            surfaces_data = data.get("surfaces", [])
            wavelengths = data.get("wavelengths", wavelengths)
            if "source_config" in data:
                stack.source_config = data["source_config"]
            if "system_config" in data:
                stack.system_config = data["system_config"]
            if "optimizer_config" in data:
                stack.optimizer_config = data["optimizer_config"]
            if "analysis_config" in data:
                stack.analysis_config = data["analysis_config"]
            if isinstance(data.get("viewer_settings"), dict):
                stack.viewer_settings = data["viewer_settings"]
            if "optical_routes" in data:
                from optics_app.engine.routes import OpticalRouteSet
                stack.optical_routes = OpticalRouteSet.from_dict(data["optical_routes"])
            if "two_photon_config" in data:
                from optics_app.analysis.two_photon import TwoPhotonMicroscopeConfig
                stack.two_photon_config = TwoPhotonMicroscopeConfig.from_dict(data["two_photon_config"])
            if "nonlinear_imaging_config" in data:
                from optics_app.analysis.nonlinear_imaging import NonlinearImagingConfig
                stack.nonlinear_imaging_config = NonlinearImagingConfig.from_dict(data["nonlinear_imaging_config"])
            if "spectrometer_calibration" in data:
                stack.spectrometer_calibration = data["spectrometer_calibration"]
            if "components" in data:
                components_raw = data.get("components", [])
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
        elif isinstance(data, list):
            # Legacy format (just a list of surfaces)
            surfaces_data = data
            
        for d in surfaces_data:
            st = d.get("surface_type")
            
            # Stop support: ensure is_stop flag and STOP surface type are preserved
            name_u = str(d.get("name", "")).upper()
            if (isinstance(st, str) and st.upper() == "STOP") or d.get("is_stop") or ("STOP" in name_u and not name_u.startswith("AIR")):
                d["is_stop"] = True
                if isinstance(st, str) and st.upper() == "STOP":
                    d["surface_type"] = SurfaceType.STOP
                    st = SurfaceType.STOP

            if isinstance(st, str):
                for val in [SurfaceType.OBJECT, SurfaceType.STANDARD, SurfaceType.STOP, SurfaceType.MIRROR, SurfaceType.ASPHERIC, SurfaceType.GRATING, SurfaceType.DETECTOR, SurfaceType.CAD, SurfaceType.AIR_SPACE]:
                    if val.upper() == st.upper():
                        d["surface_type"] = val
                        break
            # Clean non-field keys
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
            
        if isinstance(data, dict) and "primary_wavelength" in data:
            stack.primary_wavelength = float(data["primary_wavelength"])
        elif wavelengths:
            stack.primary_wavelength = wavelengths[len(wavelengths) // 2]
        stack.consolidate_dmd_surfaces()
        stack.bind_components_to_surfaces()
        stack._update_accumulated_positions()
            
        return {
            "stack": stack,
            "components": stack.components,
            "wavelengths": wavelengths,
            "system_config": data.get("system_config") if isinstance(data, dict) else None,
            "source_config": data.get("source_config") if isinstance(data, dict) else None,
            "optimizer_config": data.get("optimizer_config") if isinstance(data, dict) else None,
            "analysis_config": data.get("analysis_config") if isinstance(data, dict) else None,
            "viewer_settings": data.get("viewer_settings") if isinstance(data, dict) else None,
            "optical_routes": data.get("optical_routes") if isinstance(data, dict) else None,
            "two_photon_config": data.get("two_photon_config") if isinstance(data, dict) else None,
            "nonlinear_imaging_config": data.get("nonlinear_imaging_config") if isinstance(data, dict) else None,
            "raw_data": data
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize SurfaceStack into a dictionary representation."""
        import dataclasses
        surfaces_data = [s.to_dict() if hasattr(s, "to_dict") else dataclasses.asdict(s) for s in self.surfaces]
        for d in surfaces_data:
            if "cad_profile" in d:
                d["cad_profile"] = None
            if "global_origin" in d:
                del d["global_origin"]
            if "global_R" in d:
                del d["global_R"]
        components_data = self.extract_components_dict()
        res = {
            "version": 5,
            "surfaces": surfaces_data,
            "components": components_data,
            "wavelengths": list(getattr(self, "wavelengths", [0.450, 0.546, 0.650])),
            "primary_wavelength": float(getattr(self, "primary_wavelength", 0.58756)),
        }
        if hasattr(self, "system_config") and self.system_config is not None:
            res["system_config"] = self.system_config.to_dict() if hasattr(self.system_config, "to_dict") else self.system_config
        if hasattr(self, "source_config") and self.source_config is not None:
            res["source_config"] = self.source_config.to_dict() if hasattr(self.source_config, "to_dict") else self.source_config
        if hasattr(self, "optimizer_config") and self.optimizer_config is not None:
            res["optimizer_config"] = self.optimizer_config.to_dict() if hasattr(self.optimizer_config, "to_dict") else self.optimizer_config
        if hasattr(self, "analysis_config") and self.analysis_config is not None:
            res["analysis_config"] = self.analysis_config
        if hasattr(self, "optical_routes") and self.optical_routes is not None:
            routes = self.optical_routes
            res["optical_routes"] = routes.to_dict() if hasattr(routes, "to_dict") else routes
        if hasattr(self, "two_photon_config") and self.two_photon_config is not None:
            tpc = self.two_photon_config
            res["two_photon_config"] = tpc.to_dict() if hasattr(tpc, "to_dict") else tpc
        if hasattr(self, "nonlinear_imaging_config") and self.nonlinear_imaging_config is not None:
            nic = self.nonlinear_imaging_config
            res["nonlinear_imaging_config"] = nic.to_dict() if hasattr(nic, "to_dict") else nic
        if hasattr(self, "srs_microscope_config") and self.srs_microscope_config is not None:
            srs = self.srs_microscope_config
            res["srs_microscope_config"] = srs.to_dict() if hasattr(srs, "to_dict") else srs
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurfaceStack":
        """Instantiate a SurfaceStack from a dictionary."""
        stack = cls()
        surfaces_raw = data.get("surfaces", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for d in surfaces_raw:
            st = d.get("surface_type")
            if isinstance(st, str) and st.upper() == "STOP":
                d["surface_type"] = SurfaceType.STANDARD
                d["is_stop"] = True
                st = SurfaceType.STANDARD
            if isinstance(st, str):
                for val in [SurfaceType.OBJECT, SurfaceType.STANDARD, SurfaceType.MIRROR, SurfaceType.ASPHERIC, SurfaceType.GRATING, SurfaceType.DETECTOR, SurfaceType.CAD, SurfaceType.AIR_SPACE]:
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

        if isinstance(data, dict):
            if "components" in data:
                components_raw = data.get("components", [])
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

            if "wavelengths" in data:
                stack.wavelengths = list(data["wavelengths"])
            if "primary_wavelength" in data:
                stack.primary_wavelength = float(data["primary_wavelength"])
            if "system_config" in data:
                stack.system_config = data["system_config"]
            if "source_config" in data:
                stack.source_config = data["source_config"]
            if "optimizer_config" in data:
                stack.optimizer_config = data["optimizer_config"]
            if "analysis_config" in data:
                stack.analysis_config = data["analysis_config"]
            if "optical_routes" in data and data["optical_routes"] is not None:
                try:
                    from optics_app.engine.routes import OpticalRouteSet
                    stack.optical_routes = OpticalRouteSet.from_dict(data["optical_routes"])
                except Exception:
                    stack.optical_routes = data["optical_routes"]
            if "two_photon_config" in data and data["two_photon_config"] is not None:
                try:
                    from optics_app.analysis.two_photon import TwoPhotonMicroscopeConfig
                    stack.two_photon_config = TwoPhotonMicroscopeConfig.from_dict(data["two_photon_config"])
                except Exception:
                    stack.two_photon_config = data["two_photon_config"]
            if "nonlinear_imaging_config" in data and data["nonlinear_imaging_config"] is not None:
                try:
                    from optics_app.analysis.nonlinear_imaging import NonlinearImagingConfig
                    stack.nonlinear_imaging_config = NonlinearImagingConfig.from_dict(data["nonlinear_imaging_config"])
                except Exception:
                    stack.nonlinear_imaging_config = data["nonlinear_imaging_config"]
            if "srs_microscope_config" in data and data["srs_microscope_config"] is not None:
                try:
                    from optics_app.analysis.ultrafast_nonlinear import ThorlabsSRSMicroscopeConfig
                    stack.srs_microscope_config = ThorlabsSRSMicroscopeConfig.from_dict(data["srs_microscope_config"])
                except Exception:
                    stack.srs_microscope_config = data["srs_microscope_config"]

        stack.consolidate_dmd_surfaces()
        stack.bind_components_to_surfaces()
        stack._update_accumulated_positions()
        return stack

    def to_json(self, filepath: Optional[str] = None, indent: int = 4) -> str:
        """Serialize SurfaceStack to JSON string or save to file if filepath provided."""
        import json
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, source: Union[str, Dict[str, Any]]) -> "SurfaceStack":
        """Deserialize SurfaceStack from JSON string, dictionary, or file path."""
        import json
        import os
        if isinstance(source, dict):
            return cls.from_dict(source)
        if isinstance(source, str):
            if os.path.exists(source):
                with open(source, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(source)
            return cls.from_dict(data)
        raise ValueError(f"Unsupported source type for from_json: {type(source)}")

    def consolidate_dmd_surfaces(self) -> int:
        """
        Merge any redundant post-DMD coordinate break surfaces into the DMD mirror itself.
        If a DMD surface is immediately followed by a coordinate break with zero tilts and
        zero decenters, its thickness is transferred to the DMD surface and the redundant
        surface is removed so light reflects off the single expected DMD face.
        Returns the number of removed redundant surfaces.
        """
        from optics_app.engine.mechanisms import is_dmd
        removed = 0
        i = 0
        while i < len(self.surfaces) - 1:
            s_cur = self.surfaces[i]
            s_next = self.surfaces[i + 1]
            if is_dmd(s_cur) and (s_cur.is_reflective() or str(getattr(s_cur, "surface_type", "")).upper() == "MIRROR"):
                is_cb = getattr(s_next, "is_coord_break", False) or (str(getattr(s_next, "surface_type", "")).upper() in ("COORDINATE BREAK", "COORDINATE_BREAK", "COORDBRK", "CBREAK"))
                name_match = "CB" in str(getattr(s_next, "name", "")).upper() and "DMD" in str(getattr(s_next, "name", "")).upper()
                zero_tilt = abs(float(getattr(s_next, "tilt_x", 0.0))) < 1e-4 and abs(float(getattr(s_next, "tilt_y", 0.0))) < 1e-4 and abs(float(getattr(s_next, "tilt_z", 0.0))) < 1e-4
                zero_decenter = abs(float(getattr(s_next, "decenter_x", 0.0))) < 1e-4 and abs(float(getattr(s_next, "decenter_y", 0.0))) < 1e-4 and abs(float(getattr(s_next, "decenter_z", 0.0))) < 1e-4
                if (is_cb or name_match) and zero_tilt and zero_decenter:
                    extra_thick = float(getattr(s_next, "thickness", 0.0))
                    s_cur.thickness = float(getattr(s_cur, "thickness", 0.0)) + extra_thick
                    self.surfaces.pop(i + 1)
                    removed += 1
                    continue
            i += 1
        if removed > 0:
            self._update_accumulated_positions()
        return removed

    def add_surface(self, surface: Surface) -> None:
        self.surfaces.append(surface)
        self._update_accumulated_positions()

    def ensure_component_uids(self) -> None:
        """Assign one stable UID to each physical component without changing legacy grouping."""
        previous_key: Optional[Tuple[str, Any]] = None
        previous_uid = ""
        for index, surf in enumerate(self.surfaces):
            uid = str(getattr(surf, "component_uid", "") or "").strip()
            element_id = getattr(surf, "element_id", None)
            group_name = str(getattr(surf, "group_name", "") or "").strip()
            if element_id is not None:
                key: Tuple[str, Any] = ("element", element_id)
            elif group_name:
                key = ("group", group_name.upper())
            else:
                key = ("surface", index)
            if not uid:
                uid = previous_uid if key == previous_key and previous_uid else uuid.uuid4().hex
            surf.component_uid = uid
            previous_key = key
            previous_uid = uid

    def insert_surface(self, index: int, surface: Optional[Surface] = None) -> int:
        """
        Insert a single surface sequentially at the specified index,
        maintaining coordinate continuity and inheriting appropriate geometry from previous optical elements.
        """
        if index < 0:
            index = 0
        if index > len(self.surfaces):
            index = len(self.surfaces)

        if surface is None:
            if index > 0 and index - 1 < len(self.surfaces):
                prev_surf = self.surfaces[index - 1]
                sd = float(prev_surf.semi_diameter) if prev_surf.semi_diameter > 0 else 12.7
                prev_thick = float(prev_surf.thickness)
                if prev_surf.is_glass:
                    surface = Surface(
                        name=f"S{index}",
                        surface_type=SurfaceType.STANDARD,
                        radius=0.0,
                        thickness=10.0,
                        glass="AIR",
                        semi_diameter=sd,
                        group_name=getattr(prev_surf, 'group_name', '')
                    )
                else:
                    if prev_thick > 4.0:
                        half_thick = prev_thick / 2.0
                        prev_surf.thickness = half_thick
                        new_thick = half_thick
                    else:
                        new_thick = 10.0
                    surface = Surface(
                        name=f"S{index}",
                        surface_type=SurfaceType.STANDARD,
                        radius=50.0,
                        thickness=new_thick,
                        glass="N-BK7",
                        semi_diameter=sd
                    )
            elif len(self.surfaces) > 0:
                first_surf = self.surfaces[0]
                sd = float(first_surf.semi_diameter) if first_surf.semi_diameter > 0 else 12.7
                surface = Surface(
                    name=f"S{index}",
                    surface_type=SurfaceType.STANDARD,
                    radius=0.0,
                    thickness=10.0,
                    glass="AIR",
                    semi_diameter=sd
                )
            else:
                surface = Surface(
                    name="S0",
                    surface_type=SurfaceType.OBJECT,
                    radius=0.0,
                    thickness=50.0,
                    glass="AIR",
                    semi_diameter=12.7
                )

        self.surfaces.insert(index, surface)
        self._update_accumulated_positions()
        return index

    def remove_surface(self, index: int) -> None:
        if 0 <= index < len(self.surfaces):
            self.surfaces.pop(index)
            self._update_accumulated_positions()

    def delete_lens_element(self, target: Any) -> List[int]:
        """
        Delete an entire multi-surface or singlet lens element from the optical stack.
        Target can be a surface index (int) belonging to the lens, or a group name (str).
        Merges preceding and trailing air gaps to preserve downstream component 3D alignment.
        Returns list of deleted surface indices.
        """
        from optics_app.engine.element_colors import compute_surface_groups
        if not self.surfaces:
            return []

        groups = compute_surface_groups(self.surfaces)
        matched_indices = []

        if isinstance(target, int):
            if 0 <= target < len(groups) and groups[target] and groups[target].get("group_type") == "LENS":
                elem_id = groups[target].get("element_id")
                g_name = groups[target].get("group_name")
                for i, grp in enumerate(groups):
                    if grp and grp.get("group_type") == "LENS" and (grp.get("element_id") == elem_id or grp.get("group_name") == g_name):
                        matched_indices.append(i)
            elif 0 <= target < len(self.surfaces):
                matched_indices = [target]
        elif isinstance(target, str):
            target_str = target.strip().upper()
            for i, grp in enumerate(groups):
                if grp and grp.get("group_type") == "LENS":
                    g_name = str(grp.get("group_name", "")).strip().upper()
                    if g_name == target_str or target_str in g_name:
                        matched_indices.append(i)

        if not matched_indices:
            return []

        matched_indices = sorted(list(set(matched_indices)))
        
        # Calculate thickness to merge into preceding surface
        removed_thickness = sum(self.surfaces[k].thickness for k in matched_indices)
        first_idx = matched_indices[0]
        if first_idx > 0:
            self.surfaces[first_idx - 1].thickness += removed_thickness

        # Pop surfaces in reverse order
        for idx in reversed(matched_indices):
            self.surfaces.pop(idx)

        self._update_accumulated_positions()
        return matched_indices

    def insert_singlet(
        self,
        after_surface_index: int,
        name: str = "Singlet",
        radius_front: float = 50.0,
        thickness: float = 5.0,
        glass: str = "N-BK7",
        radius_back: float = -50.0,
        semi_diameter: float = 12.7,
        air_gap_after: float = 10.0
    ) -> List[int]:
        """
        Insert a singlet lens (2 optical surfaces: front glass and back air) after after_surface_index.
        Returns list of newly inserted surface indices.
        """
        insert_idx = after_surface_index + 1
        s_front = Surface(
            name=f"{name}_Front",
            group_name=name,
            surface_type=SurfaceType.STANDARD,
            radius=radius_front,
            thickness=thickness,
            glass=glass,
            semi_diameter=semi_diameter,
            thickness_variable=False,
            radius_variable=False
        )
        s_back = Surface(
            name=f"{name}_Back",
            group_name=name,
            surface_type=SurfaceType.STANDARD,
            radius=radius_back,
            thickness=air_gap_after,
            glass="AIR",
            semi_diameter=semi_diameter,
            thickness_variable=True,
            radius_variable=False
        )
        self.surfaces.insert(insert_idx, s_front)
        self.surfaces.insert(insert_idx + 1, s_back)
        self._update_accumulated_positions()
        return [insert_idx, insert_idx + 1]

    def insert_doublet(
        self,
        after_surface_index: int,
        name: str = "Doublet",
        r1: float = 34.0,
        t1: float = 6.0,
        glass1: str = "N-BK7",
        r2: float = -27.8,
        t2: float = 2.0,
        glass2: str = "SF5",
        r3: float = -115.0,
        semi_diameter: float = 12.7,
        air_gap_after: float = 10.0
    ) -> List[int]:
        """
        Insert an achromatic doublet lens (3 surfaces: front, cemented interface, back) after after_surface_index.
        Returns list of newly inserted surface indices.
        """
        insert_idx = after_surface_index + 1
        s1 = Surface(name=f"{name}_Front", group_name=name, radius=r1, thickness=t1, glass=glass1, semi_diameter=semi_diameter, thickness_variable=False)
        s2 = Surface(name=f"{name}_Mid", group_name=name, radius=r2, thickness=t2, glass=glass2, semi_diameter=semi_diameter, thickness_variable=False)
        s3 = Surface(name=f"{name}_Back", group_name=name, radius=r3, thickness=air_gap_after, glass="AIR", semi_diameter=semi_diameter, thickness_variable=True)

        self.surfaces.insert(insert_idx, s1)
        self.surfaces.insert(insert_idx + 1, s2)
        self.surfaces.insert(insert_idx + 2, s3)
        self._update_accumulated_positions()
        return [insert_idx, insert_idx + 1, insert_idx + 2]

    def replace_lens(
        self,
        surface_index: int,
        new_surfaces: List[Surface],
        surfaces_to_remove: int = 2
    ) -> List[int]:
        """
        Replace an existing lens element starting at surface_index with a new set of surfaces.
        """
        # Remove old surfaces
        for _ in range(min(surfaces_to_remove, len(self.surfaces) - surface_index)):
            if surface_index < len(self.surfaces):
                self.surfaces.pop(surface_index)

        # Insert new surfaces
        for k, s in enumerate(new_surfaces):
            self.surfaces.insert(surface_index + k, s)

        self._update_accumulated_positions()
        return list(range(surface_index, surface_index + len(new_surfaces)))

    @staticmethod
    def _build_catalog_surfaces(
        lens_info: Dict[str, Any],
        key: str,
        reverse: bool = False,
        air_gap_after: Optional[float] = None,
        use_catalog_bfl: bool = False,
        sd_override: Optional[float] = None,
    ) -> List[Surface]:
        raw_surfs = lens_info["surfaces"]
        sd = float(sd_override) if sd_override is not None else float(lens_info.get("semi_diameter", 12.7))
        bfl = float(lens_info.get("back_focal_length_mm", raw_surfs[-1].get("thickness", 10.0)))
        final_gap = float(air_gap_after) if air_gap_after is not None else (bfl if use_catalog_bfl else float(raw_surfs[-1].get("thickness", 10.0)))

        sec = str(lens_info.get("section", "")).lower()

        if not reverse:
            surfs = []
            for idx, s_data in enumerate(raw_surfs):
                s_name = f"{key}_S{idx+1}"
                thic = final_gap if idx == len(raw_surfs) - 1 else float(s_data["thickness"])
                stype_input = str(s_data.get("surface_type", "")).upper()
                glass_str = str(s_data.get("glass", "")).upper()

                if stype_input == "STOP":
                    stype = SurfaceType.STOP
                elif stype_input == "DETECTOR":
                    stype = SurfaceType.DETECTOR
                elif stype_input == "GRATING" or sec in ("diffraction_gratings", "gratings"):
                    stype = SurfaceType.GRATING
                elif stype_input == "MIRROR" or glass_str in ("MIRROR", "MIRR", "REFL") or sec in ("parabolic_mirrors", "spherical_mirrors", "concave_mirrors", "mirrors"):
                    stype = SurfaceType.MIRROR
                elif "conic" in s_data or "asphere_coeffs" in s_data:
                    stype = SurfaceType.ASPHERIC
                else:
                    stype = SurfaceType.STANDARD

                is_bs = bool(s_data.get("is_beam_splitter", False) or (idx == 1 and "beamsplitter" in sec))
                bs_cfg = s_data.get("beamsplitter_config") or (lens_info.get("beamsplitter_config") if is_bs else None)

                s = Surface(
                    name=s_name,
                    group_name=key,
                    surface_type=stype,
                    radius=float(s_data["radius"]),
                    thickness=thic,
                    glass=str(s_data["glass"]),
                    semi_diameter=sd,
                    conic=float(s_data.get("conic", 0.0)),
                    asphere_coeffs=list(s_data.get("asphere_coeffs", [])),
                    thickness_variable=(s_data["glass"].upper() in ["AIR", ""]),
                    substrate_thickness=float(s_data.get("thickness", lens_info.get("center_thickness_mm", 6.0))),
                    grating_lines_per_mm=float(s_data.get("grating_lines_per_mm", lens_info.get("grooves_per_mm", 0.0))),
                    grating_order=int(s_data.get("grating_order", 1)),
                    grating_primary_wavelength_um=s_data.get("grating_primary_wavelength_um", lens_info.get("blaze_wavelength_um", None)),
                    grating_blaze_wavelength_um=s_data.get("grating_blaze_wavelength_um", lens_info.get("blaze_wavelength_um", None)),
                    grating_blaze_angle_deg=float(s_data.get("grating_blaze_angle_deg", lens_info.get("blaze_angle_deg", 0.0))),
                    grating_type=str(s_data.get("grating_type", "Blazed Reflective" if not lens_info.get("is_transmission", False) else "Transmission")),
                    is_transmission_grating=bool(s_data.get("is_transmission_grating", lens_info.get("is_transmission", False))),
                    tilt_x=float(s_data.get("tilt_x", 0.0)),
                    tilt_y=float(s_data.get("tilt_y", 0.0)),
                    tilt_z=float(s_data.get("tilt_z", 0.0)),
                    decenter_x=float(s_data.get("decenter_x", 0.0)),
                    decenter_y=float(s_data.get("decenter_y", 0.0)),
                    is_beam_splitter=is_bs,
                    beamsplitter_config=bs_cfg,
                )
                surfs.append(s)
            return surfs
        else:
            # Flipped / Reversed element orientation
            surfs = []
            if len(raw_surfs) == 3:
                # Doublet (S1 glass1, S2 glass2, S3 air) -> (-R3 glass2, -R2 glass1, -R1 air)
                s1_data, s2_data, s3_data = raw_surfs
                r1, t1, g1 = float(s1_data["radius"]), float(s1_data["thickness"]), str(s1_data["glass"])
                r2, t2, g2 = float(s2_data["radius"]), float(s2_data["thickness"]), str(s2_data["glass"])
                r3 = float(s3_data["radius"])

                surfs.append(Surface(
                    name=f"{key}_rev_S1",
                    group_name=f"{key}_rev",
                    surface_type=SurfaceType.STANDARD,
                    radius=-r3 if r3 != 0.0 else 0.0,
                    thickness=t2,
                    glass=g2,
                    semi_diameter=sd,
                    thickness_variable=False,
                ))
                surfs.append(Surface(
                    name=f"{key}_rev_S2",
                    group_name=f"{key}_rev",
                    surface_type=SurfaceType.STANDARD,
                    radius=-r2 if r2 != 0.0 else 0.0,
                    thickness=t1,
                    glass=g1,
                    semi_diameter=sd,
                    thickness_variable=False,
                ))
                surfs.append(Surface(
                    name=f"{key}_rev_S3",
                    group_name=f"{key}_rev",
                    surface_type=SurfaceType.STANDARD,
                    radius=-r1 if r1 != 0.0 else 0.0,
                    thickness=final_gap,
                    glass="AIR",
                    semi_diameter=sd,
                    thickness_variable=True,
                ))
            elif len(raw_surfs) == 2:
                # Singlet (S1 glass1, S2 air) -> (-R2 glass1, -R1 air)
                s1_data, s2_data = raw_surfs
                r1, t1, g1 = float(s1_data["radius"]), float(s1_data["thickness"]), str(s1_data["glass"])
                r2 = float(s2_data["radius"])
                conic1 = float(s1_data.get("conic", 0.0))
                coeffs1 = list(s1_data.get("asphere_coeffs", []))
                stype1 = SurfaceType.ASPHERIC if (conic1 != 0.0 or coeffs1) else SurfaceType.STANDARD

                surfs.append(Surface(
                    name=f"{key}_rev_S1",
                    group_name=f"{key}_rev",
                    surface_type=SurfaceType.STANDARD,
                    radius=-r2 if r2 != 0.0 else 0.0,
                    thickness=t1,
                    glass=g1,
                    semi_diameter=sd,
                    thickness_variable=False,
                ))
                surfs.append(Surface(
                    name=f"{key}_rev_S2",
                    group_name=f"{key}_rev",
                    surface_type=stype1,
                    radius=-r1 if r1 != 0.0 else 0.0,
                    thickness=final_gap,
                    glass="AIR",
                    semi_diameter=sd,
                    conic=conic1,
                    asphere_coeffs=coeffs1,
                    thickness_variable=True,
                ))
            elif len(raw_surfs) == 1:
                # Single-surface element (e.g. Parabolic Mirror, Grating, Flat Mirror)
                s_data = raw_surfs[0]
                r = float(s_data.get("radius", 0.0))
                conic = float(s_data.get("conic", 0.0))
                coeffs = list(s_data.get("asphere_coeffs", []))
                stype_input = str(s_data.get("surface_type", "")).upper()
                glass_str = str(s_data.get("glass", "")).upper()

                if stype_input == "GRATING" or sec in ("diffraction_gratings", "gratings"):
                    stype = SurfaceType.GRATING
                elif stype_input == "MIRROR" or glass_str in ("MIRROR", "MIRR", "REFL") or sec in ("parabolic_mirrors", "spherical_mirrors", "concave_mirrors", "mirrors"):
                    stype = SurfaceType.MIRROR
                elif conic != 0.0 or coeffs:
                    stype = SurfaceType.ASPHERIC
                else:
                    stype = SurfaceType.STANDARD

                s = Surface(
                    name=f"{key}_rev_S1",
                    group_name=f"{key}_rev",
                    surface_type=stype,
                    radius=-r if r != 0.0 else 0.0,
                    thickness=final_gap,
                    glass=str(s_data.get("glass", "AIR")),
                    semi_diameter=sd,
                    conic=conic,
                    asphere_coeffs=coeffs,
                    thickness_variable=(glass_str in ["AIR", ""]),
                    substrate_thickness=float(s_data.get("thickness", lens_info.get("center_thickness_mm", 6.0))),
                    grating_lines_per_mm=float(s_data.get("grating_lines_per_mm", lens_info.get("grooves_per_mm", 0.0))),
                    grating_order=int(s_data.get("grating_order", 1)),
                    grating_primary_wavelength_um=s_data.get("grating_primary_wavelength_um", lens_info.get("blaze_wavelength_um", None)),
                    grating_blaze_wavelength_um=s_data.get("grating_blaze_wavelength_um", lens_info.get("blaze_wavelength_um", None)),
                    grating_blaze_angle_deg=float(s_data.get("grating_blaze_angle_deg", lens_info.get("blaze_angle_deg", 0.0))),
                    grating_type=str(s_data.get("grating_type", "Blazed Reflective" if not lens_info.get("is_transmission", False) else "Transmission")),
                    is_transmission_grating=bool(s_data.get("is_transmission_grating", lens_info.get("is_transmission", False))),
                )
                surfs.append(s)
            else:
                for idx, s_data in enumerate(reversed(raw_surfs)):
                    r = float(s_data["radius"])
                    surfs.append(Surface(
                        name=f"{key}_rev_S{idx+1}",
                        group_name=f"{key}_rev",
                        radius=-r if r != 0.0 else 0.0,
                        thickness=float(s_data["thickness"]),
                        glass=str(s_data["glass"]),
                        semi_diameter=sd,
                    ))
                surfs[-1].thickness = final_gap
                surfs[-1].glass = "AIR"
            return surfs

    def insert_catalog_lens(
        self,
        after_surface_index: int,
        part_number: str,
        air_gap_after: Optional[float] = None,
        reverse: bool = False,
    ) -> List[int]:
        """
        Insert a standard commercial Thorlabs catalog lens immediately after after_surface_index.
        Supports normal and reversed/flipped orientation.
        """
        catalog = self._get_lens_catalog()
        key = part_number.upper().strip()
        if key not in catalog:
            raise KeyError(f"Catalog lens '{part_number}' not found in database. Available: {list(catalog.keys())}")

        lens_info = catalog[key]
        surfs = self._build_catalog_surfaces(
            lens_info=lens_info,
            key=key,
            reverse=reverse,
            air_gap_after=air_gap_after,
        )

        insert_idx = after_surface_index + 1
        for k, s in enumerate(surfs):
            self.surfaces.insert(insert_idx + k, s)
        self._update_accumulated_positions()
        return list(range(insert_idx, insert_idx + len(surfs)))

    def replace_with_catalog_lens(
        self,
        surface_index: int,
        part_number: str,
        surfaces_to_remove: Optional[int] = None,
        reverse: bool = False,
        air_gap_after: Optional[float] = None,
        use_catalog_bfl: bool = False,
        preserve_aperture: bool = False,
        auto_adjust_focus: bool = False,
    ) -> List[int]:
        """
        Replace the lens starting at surface_index with a catalog lens.
        Automatically detects whether old lens was a singlet (2 surfaces) or doublet (3 surfaces).
        Supports choosing reverse orientation, air gap after lens, autofocus correction, and aperture preservation.
        """
        # This legacy entry point used to delete forward from the clicked row.
        # Clicking a cemented or rear surface therefore left part of the old
        # lens behind. All catalog replacements now use the same atomic
        # whole-element resolver as the Component Editor and AI tools.
        return self.replace_lens_from_catalog(
            target=surface_index,
            part_number=part_number,
            reverse=reverse,
            air_gap_after=air_gap_after,
            use_catalog_bfl=use_catalog_bfl,
            auto_adjust_focus=auto_adjust_focus,
            preserve_aperture=preserve_aperture,
        )

    def replace_lens_element(
        self,
        target: Any,
        new_surfaces: List[Surface],
        preserve_trailing_air_gap: bool = True,
        air_gap_after: Optional[float] = None,
        preserve_aperture: bool = False,
    ) -> List[int]:
        """
        Replace an entire multi-surface or singlet lens element with a new set of surfaces.
        Target can be a group name (str) or a surface index (int) belonging to the lens.
        Preserves preceding and following air gaps by default, or applies custom trailing air gap.
        Returns list of newly inserted surface indices.
        """
        if not self.surfaces or not new_surfaces:
            return []
        matched_indices = self.find_lens_element_span(target)
        if not matched_indices:
            raise ValueError(
                f"Target '{target}' is not a surface belonging to a physical lens; no surfaces were replaced."
            )

        first_idx = matched_indices[0]
        n_remove = len(matched_indices)

        # Preserve the trailing air gap
        old_trailing_air_gap = 10.0
        last_surf_idx = matched_indices[-1]
        if 0 <= last_surf_idx < len(self.surfaces):
            old_trailing_air_gap = self.surfaces[last_surf_idx].thickness

        # Preserve the lens pose and the trailing propagation gap. Outside
        # surfaces are not edited; replace_lens performs one contiguous splice.
        old_front = self.surfaces[first_idx]
        pose_fields = (
            "decenter_x", "decenter_y", "decenter_z",
            "tilt_x", "tilt_y", "tilt_z", "is_active",
        )
        preserved_pose = {field: getattr(old_front, field) for field in pose_fields}
        preserved_element_id = getattr(old_front, "element_id", None)
        preserved_component_uid = str(getattr(old_front, "component_uid", "") or "")
        old_semi_diameter = getattr(old_front, "semi_diameter", 12.7)

        for surface in new_surfaces:
            for field, value in preserved_pose.items():
                if getattr(surface, field, 0.0) == 0.0 or not hasattr(surface, field):
                    setattr(surface, field, value)
            surface.element_id = preserved_element_id
            surface.component_uid = preserved_component_uid
            surface.is_locked = True
            if preserve_aperture:
                surface.semi_diameter = old_semi_diameter

        if air_gap_after is not None:
            new_surfaces[-1].thickness = float(air_gap_after)
        elif preserve_trailing_air_gap:
            new_surfaces[-1].thickness = old_trailing_air_gap

        return self.replace_lens(first_idx, new_surfaces, surfaces_to_remove=n_remove)

    def replace_lens_from_zmx(self, target: Any, file_path: str, reverse: bool = False, tilt_x: float = 0.0, tilt_y: float = 0.0, tilt_z: float = 0.0) -> List[int]:
        """Replace target lens element with optical surfaces from a Zemax (.zmx) file."""
        import os
        from optics_app.io.zmx_parser import parse_zmx_file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Zemax file not found: {file_path}")

        imported_stack = parse_zmx_file(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Filter out dummy object plane (S0) and dummy detector (last) if they are standard air planes
        imported_surfs = list(imported_stack.surfaces)
        if len(imported_surfs) >= 3:
            if imported_surfs[0].surface_type == SurfaceType.OBJECT or imported_surfs[0].name.upper().startswith("OBJ"):
                imported_surfs = imported_surfs[1:]
            if len(imported_surfs) >= 2 and (imported_surfs[-1].surface_type == SurfaceType.DETECTOR or imported_surfs[-1].name.upper().startswith("IMA")):
                imported_surfs = imported_surfs[:-1]

        # Optional reversal of surfaces
        if reverse:
            n_surfs = len(imported_surfs)
            reversed_surfs = []
            if n_surfs >= 2 and imported_surfs[-1].glass.upper() in ("AIR", ""):
                old_props = [
                    {
                        "name": s.name,
                        "surface_type": s.surface_type,
                        "radius": s.radius,
                        "thickness": s.thickness,
                        "glass": s.glass,
                        "semi_diameter": s.semi_diameter,
                        "conic": s.conic,
                        "asphere_coeffs": list(s.asphere_coeffs),
                    }
                    for s in imported_surfs
                ]
                for k in range(n_surfs):
                    old_geom = old_props[n_surfs - 1 - k]
                    if k < n_surfs - 1:
                        old_mat = old_props[n_surfs - 2 - k]
                        thic = old_mat["thickness"]
                        mat = old_mat["glass"]
                    else:
                        thic = old_props[-1]["thickness"]
                        mat = "AIR"

                    new_s = Surface(
                        name=f"{base_name}_rev_S{k+1}",
                        surface_type=old_geom["surface_type"],
                        radius=-old_geom["radius"] if old_geom["radius"] != 0.0 else 0.0,
                        thickness=thic,
                        glass=mat,
                        semi_diameter=old_geom["semi_diameter"],
                        conic=old_geom["conic"],
                        asphere_coeffs=list(old_geom["asphere_coeffs"]),
                        group_name=base_name
                    )
                    reversed_surfs.append(new_s)
                imported_surfs = reversed_surfs
            else:
                for s in reversed(imported_surfs):
                    new_s = Surface(
                        name=s.name,
                        surface_type=s.surface_type,
                        radius=-s.radius if s.radius != 0 else 0.0,
                        thickness=s.thickness,
                        glass=s.glass,
                        semi_diameter=s.semi_diameter,
                        conic=s.conic,
                        asphere_coeffs=list(s.asphere_coeffs),
                        group_name=base_name
                    )
                    reversed_surfs.append(new_s)
                imported_surfs = reversed_surfs
        else:
            for s in imported_surfs:
                s.group_name = base_name

        if tilt_x != 0.0 or tilt_y != 0.0 or tilt_z != 0.0:
            for s in imported_surfs:
                s.tilt_x = tilt_x
                s.tilt_y = tilt_y
                s.tilt_z = tilt_z

        return self.replace_lens_element(target, imported_surfs)

    def replace_lens_from_custom_file(self, target: Any, file_path: str) -> List[int]:
        """Replace target lens element with surfaces from a custom JSON/text file."""
        import os, json
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Custom file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        base_name = data.get("name") or os.path.splitext(os.path.basename(file_path))[0]
        surf_list = data.get("surfaces", [])
        if not surf_list and isinstance(data, list):
            surf_list = data

        new_surfs = []
        for idx, s_data in enumerate(surf_list):
            stype = SurfaceType.ASPHERIC if ("conic" in s_data or "asphere_coeffs" in s_data) else SurfaceType.STANDARD
            s = Surface(
                name=str(s_data.get("name", f"{base_name}_S{idx+1}")),
                group_name=base_name,
                surface_type=stype,
                radius=float(s_data.get("radius", 0.0)),
                thickness=float(s_data.get("thickness", 10.0)),
                glass=str(s_data.get("glass", "AIR")),
                semi_diameter=float(s_data.get("semi_diameter", 12.7)),
                conic=float(s_data.get("conic", 0.0)),
                asphere_coeffs=list(s_data.get("asphere_coeffs", [])),
                tilt_x=float(s_data.get("tilt_x", 0.0)),
                tilt_y=float(s_data.get("tilt_y", 0.0)),
                tilt_z=float(s_data.get("tilt_z", 0.0))
            )
            new_surfs.append(s)

        # Filter out standalone object/image if full system was passed
        if len(new_surfs) >= 3:
            if new_surfs[0].surface_type == SurfaceType.OBJECT or new_surfs[0].name.upper().startswith("OBJ"):
                new_surfs = new_surfs[1:]
            if len(new_surfs) >= 2 and (new_surfs[-1].surface_type == SurfaceType.DETECTOR or new_surfs[-1].name.upper().startswith("IMA")):
                new_surfs = new_surfs[:-1]

        return self.replace_lens_element(target, new_surfs)

    def find_lens_element_span(self, target: Any) -> List[int]:
        """Find the contiguous surface indices belonging to the specified lens element."""
        from optics_app.engine.element_colors import compute_surface_groups
        if not self.surfaces:
            return []
        groups = compute_surface_groups(self.surfaces)

        def contiguous_lens_span(seed: int) -> List[int]:
            if not (0 <= seed < len(self.surfaces)):
                return []
            # Resolve from optical media rather than the table's current visual
            # grouping. This makes every surface of the lens (front, cemented
            # interfaces, and rear AIR boundary) select the same element.
            if self.surfaces[seed].is_glass:
                glass_seed = seed
            elif seed > 0 and self.surfaces[seed - 1].is_glass:
                glass_seed = seed - 1
            elif (
                self.surfaces[seed].is_reflective()
                or self.surfaces[seed].surface_type in (SurfaceType.MIRROR, SurfaceType.GRATING)
                or "MIRROR" in str(getattr(self.surfaces[seed], "name", "")).upper()
                or "GRATING" in str(getattr(self.surfaces[seed], "name", "")).upper()
            ):
                return [seed]
            else:
                return []
            first = glass_seed
            while first > 0 and self.surfaces[first - 1].is_glass:
                first -= 1
            last = glass_seed
            while last + 1 < len(self.surfaces) and self.surfaces[last + 1].is_glass:
                last += 1
            if last + 1 < len(self.surfaces):
                rear = self.surfaces[last + 1]
                rear_type = str(getattr(rear, "surface_type", "")).upper()
                rear_glass = str(getattr(rear, "glass", "") or "").upper().strip()
                if rear_glass in ("", "AIR", "VACUUM") and not any(
                    marker in rear_type for marker in ("OBJECT", "DETECTOR", "IMAGE", "STOP", "GRATING", "MIRROR")
                ):
                    last += 1
            return list(range(first, last + 1))

        if isinstance(target, int):
            span = contiguous_lens_span(target)
            if span:
                return span
        elif isinstance(target, str):
            target_name = target.strip().upper()
            exact_surface_indices = [
                idx for idx, surface in enumerate(self.surfaces)
                if str(getattr(surface, "name", "") or "").strip().upper() == target_name
            ]
            if len(exact_surface_indices) == 1:
                return contiguous_lens_span(exact_surface_indices[0])

            spans: List[List[int]] = []
            for index, group in enumerate(groups):
                if group and str(group.get("group_name", "")).strip().upper() == target_name:
                    span = contiguous_lens_span(index)
                    if span and span not in spans:
                        spans.append(span)
            if len(spans) > 1:
                raise ValueError(
                    f"Lens name '{target}' is ambiguous ({len(spans)} physical lenses). "
                    "Select a surface belonging to the intended lens."
                )
            if spans:
                return spans[0]
            for idx, s in enumerate(self.surfaces):
                if s.name and s.name.strip().upper() == target_name:
                    return contiguous_lens_span(idx)
        return []

    def calculate_autofocus_gap(
        self,
        target: Any,
        new_part_number: str,
        reverse: bool = False
    ) -> float:
        """
        Compute the automatic focal length corrected trailing air gap when replacing a lens.
        If downstream surface is a detector/image plane, adjusts trailing distance to the new BFL.
        Otherwise shifts existing trailing distance by ΔEFL = EFL_new - EFL_old.
        """
        catalog = self._get_lens_catalog()
        key = new_part_number.upper().strip()
        if key not in catalog:
            return 10.0

        lens_info = catalog[key]
        new_efl = float(lens_info.get("focal_length_mm", lens_info.get("nominal_focal_length_mm", 50.0)))
        new_bfl = float(lens_info.get("back_focal_length_mm", 43.3))

        span = self.find_lens_element_span(target)
        if not span:
            return max(0.1, new_bfl)

        old_rear_idx = span[-1]
        old_trailing_gap = self.surfaces[old_rear_idx].thickness if old_rear_idx < len(self.surfaces) else 10.0

        # Check if downstream surface is a detector
        is_detector = False
        if old_rear_idx + 1 < len(self.surfaces):
            nxt = self.surfaces[old_rear_idx + 1]
            if getattr(nxt, "surface_type", None) == SurfaceType.DETECTOR or "DET" in getattr(nxt, "name", "").upper():
                is_detector = True

        # Try to detect old element's catalog EFL
        old_efl = None
        old_front = self.surfaces[span[0]]
        old_gname = str(getattr(old_front, "group_name", "")).strip().upper()
        if old_gname in catalog:
            old_efl = float(catalog[old_gname].get("focal_length_mm", 50.0))

        if is_detector or old_efl is None:
            return max(0.1, new_bfl)
        else:
            delta_efl = new_efl - old_efl
            return max(0.1, old_trailing_gap + delta_efl)

    def replace_lens_from_catalog(
        self,
        target: Any,
        part_number: str,
        reverse: bool = False,
        air_gap_after: Optional[float] = None,
        use_catalog_bfl: bool = False,
        auto_adjust_focus: bool = False,
        preserve_aperture: bool = False,
        tilt_x: Optional[float] = None,
        tilt_y: Optional[float] = None,
        tilt_z: Optional[float] = None,
    ) -> List[int]:
        """
        Replace target lens element with a standard Thorlabs catalog lens.
        Allows choosing reverse orientation, custom air gap, catalog BFL, automatic focal length correction, aperture, and tilts.
        """
        catalog = self._get_lens_catalog()
        key = part_number.upper().strip()
        if key not in catalog:
            raise KeyError(f"Catalog lens '{part_number}' not found in database.")

        lens_info = catalog[key]
        bfl = float(lens_info.get("back_focal_length_mm", lens_info["surfaces"][-1].get("thickness", 10.0)))
        
        if auto_adjust_focus:
            effective_gap = self.calculate_autofocus_gap(target, key, reverse=reverse)
        elif air_gap_after is not None:
            effective_gap = float(air_gap_after)
        elif use_catalog_bfl:
            effective_gap = bfl
        else:
            effective_gap = None

        new_surfs = self._build_catalog_surfaces(
            lens_info=lens_info,
            key=key,
            reverse=reverse,
            air_gap_after=effective_gap,
            use_catalog_bfl=(use_catalog_bfl or auto_adjust_focus),
        )

        if tilt_x is not None:
            new_surfs[0].tilt_x = float(tilt_x)
        if tilt_y is not None:
            new_surfs[0].tilt_y = float(tilt_y)
        if tilt_z is not None:
            new_surfs[0].tilt_z = float(tilt_z)

        return self.replace_lens_element(
            target=target,
            new_surfaces=new_surfs,
            preserve_trailing_air_gap=(not use_catalog_bfl and not auto_adjust_focus and air_gap_after is None),
            air_gap_after=effective_gap,
            preserve_aperture=preserve_aperture,
        )

    @staticmethod
    def _get_lens_catalog() -> Dict[str, Any]:
        catalog = {
            "A230-C": {
                "name": "Thorlabs A230-C (f=4.51mm, NA=0.55 Aspheric Lens)",
                "semi_diameter": 3.15,
                "surfaces": [
                    {
                        "radius": 3.478709,
                        "thickness": 2.94,
                        "glass": "S-NPH1_MOLD",
                        "conic": -0.1263039,
                        "asphere_coeffs": [0.0, -0.0012606, -0.000109, 3.2255631e-7, -7.8343862e-7],
                    },
                    {
                        "radius": 0.0,
                        "thickness": 10.0,
                        "glass": "AIR"
                    }
                ]
            },
            "ACL25416U": {
                "name": "Thorlabs ACL25416U (f=16mm, NA=0.79 Aspheric Condenser)",
                "semi_diameter": 12.7,
                "back_focal_length_mm": 7.3,
                "surfaces": [
                    {"radius": 8.32, "thickness": 14.0, "glass": "B270", "conic": -0.6265, "asphere_coeffs": [1.8e-6]},
                    {"radius": 0.0, "thickness": 7.3, "glass": "AIR"}
                ]
            },
            "AC254-030-B": {
                "name": "Thorlabs AC254-030-B (f=30mm)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 18.5, "thickness": 7.5, "glass": "N-BK7"},
                    {"radius": -14.8, "thickness": 2.0, "glass": "SF5"},
                    {"radius": -57.2, "thickness": 10.0, "glass": "AIR"}
                ]
            },
            "AC254-040-B": {
                "name": "Thorlabs AC254-040-B (f=40mm)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 25.1, "thickness": 6.8, "glass": "N-BK7"},
                    {"radius": -20.2, "thickness": 2.0, "glass": "SF5"},
                    {"radius": -84.0, "thickness": 10.0, "glass": "AIR"}
                ]
            },
            "AC254-050-A": {
                "name": "Thorlabs AC254-050-A (f=50mm, 400-700nm)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 34.0, "thickness": 6.0, "glass": "N-BK7"},
                    {"radius": -27.8, "thickness": 2.0, "glass": "SF5"},
                    {"radius": -115.0, "thickness": 10.0, "glass": "AIR"}
                ]
            },
            "AC254-050-B": {
                "name": "Thorlabs AC254-050-B (f=50mm, 650-1050nm)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 28.3, "thickness": 6.0, "glass": "N-LAK22"},
                    {"radius": -21.6, "thickness": 2.0, "glass": "N-SF6HT"},
                    {"radius": -105.0, "thickness": 10.0, "glass": "AIR"}
                ]
            },
            "AC254-050-C": {
                "name": "Thorlabs AC254-050-C (f=50mm, 1050-1700nm NIR)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 28.3, "thickness": 6.0, "glass": "N-LAK22"},
                    {"radius": -21.6, "thickness": 2.0, "glass": "N-SF6HT"},
                    {"radius": -105.0, "thickness": 10.0, "glass": "AIR"}
                ]
            },
            "AC254-075-B": {
                "name": "Thorlabs AC254-075-B (f=75mm)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 45.6, "thickness": 4.5, "glass": "N-LAK22"},
                    {"radius": -32.8, "thickness": 2.0, "glass": "N-SF6HT"},
                    {"radius": -162.0, "thickness": 10.0, "glass": "AIR"}
                ]
            },
            "AC254-075-C": {
                "name": "Thorlabs AC254-075-C (f=75mm, 1050-1700nm NIR)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 45.6, "thickness": 4.5, "glass": "N-LAK22"},
                    {"radius": -32.8, "thickness": 2.0, "glass": "N-SF6HT"},
                    {"radius": -162.0, "thickness": 10.0, "glass": "AIR"}
                ]
            },
            "AC254-100-C": {
                "name": "Thorlabs AC254-100-C (f=100mm, 1050-1700nm NIR)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 60.1, "thickness": 4.0, "glass": "N-LAK22"},
                    {"radius": -43.2, "thickness": 2.0, "glass": "N-SF6HT"},
                    {"radius": -215.0, "thickness": 10.0, "glass": "AIR"}
                ]
            },
            "AC254-100-B": {
                "name": "Thorlabs AC254-100-B (f=100mm)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 61.2, "thickness": 5.0, "glass": "N-BK7"},
                    {"radius": -44.5, "thickness": 2.0, "glass": "SF5"},
                    {"radius": -198.0, "thickness": 10.0, "glass": "AIR"}
                ]
            },
            "AC254-150-B": {
                "name": "Thorlabs AC254-150-B (f=150mm)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 92.5, "thickness": 4.5, "glass": "N-BK7"},
                    {"radius": -66.8, "thickness": 2.0, "glass": "SF5"},
                    {"radius": -298.0, "thickness": 10.0, "glass": "AIR"}
                ]
            },
            "ACL25416U-B": {
                "name": "Thorlabs ACL25416U-B (f=16mm, NA=0.68 Asphere)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 8.32, "thickness": 12.0, "glass": "B270", "conic": -0.6265, "asphere_coeffs": [1.8e-6]},
                    {"radius": 0.0, "thickness": 8.5, "glass": "AIR"}
                ]
            },
            "LA1951": {
                "name": "Thorlabs LA1951 (Plano-Convex f=25.4mm)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 13.1, "thickness": 8.8, "glass": "N-BK7"},
                    {"radius": 0.0, "thickness": 10.0, "glass": "AIR"}
                ]
            },
            "LA1131": {
                "name": "Thorlabs LA1131 (Plano-Convex f=50mm)",
                "semi_diameter": 12.7,
                "surfaces": [
                    {"radius": 25.8, "thickness": 5.3, "glass": "N-BK7"},
                    {"radius": 0.0, "thickness": 10.0, "glass": "AIR"}
                ]
            }
        }
        try:
            from optics_app.catalogs.thorlabs import get_lens_catalog
            full_cat = get_lens_catalog()
            catalog.update(full_cat)
            return catalog
        except ImportError:
            try:
                from v2.catalogs.thorlabs import get_lens_catalog
                full_cat = get_lens_catalog()
                catalog.update(full_cat)
                return catalog
            except Exception:
                return catalog

    def get_active_sequence(self) -> List[Surface]:
        """Return all surfaces up to the first disabled surface."""
        active = []
        for surf in self.surfaces:
            if not getattr(surf, 'is_active', True):
                break
            active.append(surf)
        return active

    def set_zemax_defaults(self, var_air_gaps: bool = True, fix_lenses: bool = True) -> None:
        """
        Apply Zemax OpticStudio solve/variable conventions:
        - Lens elements (solid glass center thicknesses & radii) are Fixed (non-variable by default).
        - Air gaps / propagation distances between components are Variable degrees of freedom.
        """
        for s in self.surfaces:
            if getattr(s, 'is_glass', False):
                if fix_lenses:
                    s.thickness_variable = False
                    s.radius_variable = False
            else:
                if var_air_gaps:
                    s.thickness_variable = True
                    if s.thickness_min <= 0:
                        s.thickness_min = 0.5
                    if s.thickness_max <= s.thickness_min:
                        s.thickness_max = 500.0

    def set_all_lenses_fixed(self) -> None:
        """Lock all lens element thicknesses and radii as fixed catalog values."""
        for s in self.surfaces:
            if getattr(s, 'is_glass', False):
                s.thickness_variable = False
                s.radius_variable = False

    def set_air_gaps_variable(self, is_variable: bool = True) -> None:
        """Mark all propagation air gaps between optical elements as variable."""
        for s in self.surfaces:
            if getattr(s, 'is_air', True):
                s.thickness_variable = is_variable
                if is_variable and s.thickness_min <= 0:
                    s.thickness_min = 0.5

    def _get_element_indices(self, target: Any) -> List[int]:
        """
        Identify all constituent surface indices for an optical element (singlet, doublet,
        triplet, catalog lens, mirror, etc.) from an index (int) or element name (str).
        """
        if not self.surfaces:
            return []

        if isinstance(target, int):
            if target < 0 or target >= len(self.surfaces):
                return []

            target_surf = self.surfaces[target]
            grp_name = getattr(target_surf, "group_name", "")
            elem_id = getattr(target_surf, "element_id", None)

            # 1. An integer row always resolves to its local physical glass
            # region first. This keeps adjacent instances of the same catalog
            # part separate even if they share a display name or legacy ID.
            if target_surf.is_glass:
                front_idx = target
                while front_idx > 0 and self.surfaces[front_idx - 1].is_glass:
                    prev_s = self.surfaces[front_idx - 1]
                    if elem_id is not None and prev_s.element_id != elem_id:
                        break
                    if grp_name and prev_s.group_name != grp_name:
                        break
                    front_idx -= 1
                indices = [front_idx]
                k = front_idx + 1
                while k < len(self.surfaces):
                    s_k = self.surfaces[k]
                    if elem_id is not None and s_k.element_id != elem_id:
                        break
                    if grp_name and s_k.group_name != grp_name:
                        break
                    indices.append(k)
                    if not s_k.is_glass:
                        break
                    k += 1
                return indices

            if target > 0 and self.surfaces[target - 1].is_glass:
                prev_s = self.surfaces[target - 1]
                p_elem_id = getattr(prev_s, "element_id", None)
                p_grp_name = getattr(prev_s, "group_name", "")
                if grp_name and p_grp_name and grp_name != p_grp_name:
                    pass
                else:
                    front_idx = target - 1
                    while front_idx > 0 and self.surfaces[front_idx - 1].is_glass:
                        pp_s = self.surfaces[front_idx - 1]
                        if p_elem_id is not None and pp_s.element_id != p_elem_id:
                            break
                        if p_grp_name and pp_s.group_name != p_grp_name:
                            break
                        front_idx -= 1
                    indices = []
                    k = front_idx
                    while k <= target:
                        indices.append(k)
                        k += 1
                    return indices

            # 2. Explicit grouping remains useful for non-glass assemblies.
            if elem_id is not None or (grp_name and grp_name.strip()):
                matched = []
                for i, s in enumerate(self.surfaces):
                    if (elem_id is not None and s.element_id == elem_id) or (grp_name and s.group_name == grp_name):
                        matched.append(i)
                if matched and target in matched:
                    return matched

            return [target]

        elif isinstance(target, str):
            target_str = target.strip().upper()
            matched = []
            for i, s in enumerate(self.surfaces):
                g_name = str(getattr(s, "group_name", "")).strip().upper()
                s_name = str(getattr(s, "name", "")).strip().upper()
                if g_name == target_str or s_name == target_str or (target_str in g_name) or (target_str in s_name):
                    matched.append(i)
            if matched:
                return sorted(list(set(matched)))

        return [target] if isinstance(target, int) and 0 <= target < len(self.surfaces) else []

    def insert_element(
        self,
        after_surface_idx: int,
        element: Union["SurfaceStack", List[Surface]]
    ) -> List[int]:
        """
        Insert an optical element (SurfaceStack or list of Surface objects)
        immediately after after_surface_idx.
        Returns list of newly inserted surface indices.
        """
        if isinstance(element, SurfaceStack):
            new_surfs = list(element.surfaces)
        else:
            new_surfs = list(element)

        if len(new_surfs) >= 3:
            if new_surfs[0].surface_type == SurfaceType.OBJECT or new_surfs[0].name.upper().startswith("OBJ"):
                new_surfs = new_surfs[1:]
            if len(new_surfs) >= 2 and (new_surfs[-1].surface_type == SurfaceType.DETECTOR or new_surfs[-1].name.upper().startswith("IMA")):
                new_surfs = new_surfs[:-1]

        if not new_surfs:
            return []

        insert_idx = max(0, min(after_surface_idx + 1, len(self.surfaces)))
        for k, s in enumerate(new_surfs):
            self.surfaces.insert(insert_idx + k, s)

        self._update_accumulated_positions()
        return list(range(insert_idx, insert_idx + len(new_surfs)))

    def delete_element(self, element_idx: Union[int, str] = None, target: Any = None, **kwargs) -> List[int]:
        """
        Delete an entire multi-surface or singlet lens element from the optical stack.
        Preserves downstream optical spacing by adding deleted thickness to preceding gap.
        """
        if element_idx is None:
            element_idx = target
        if element_idx is None:
            return []
        if not self.surfaces:
            return []

        matched_indices = self._get_element_indices(element_idx)
        if not matched_indices:
            return []

        matched_indices = sorted(list(set(matched_indices)))
        removed_thickness = sum(self.surfaces[k].thickness for k in matched_indices)
        first_idx = matched_indices[0]

        if first_idx > 0:
            self.surfaces[first_idx - 1].thickness += removed_thickness

        for idx in reversed(matched_indices):
            self.surfaces.pop(idx)

        self._update_accumulated_positions()
        return matched_indices

    delete_lens_element = delete_element

    def flip_element(self, element_idx: Union[int, str]) -> List[int]:
        """
        Flips a singlet, doublet, or multi-surface lens element front-to-back:
        - Reverses surface ordering.
        - Inverts curvatures (R -> -R).
        - Shifts glass thicknesses and materials forward.
        - Preserves trailing air gap thickness.
        Returns list of modified surface indices.
        """
        indices = self._get_element_indices(element_idx)
        if not indices:
            return []

        n = len(indices)
        if n == 1:
            s = self.surfaces[indices[0]]
            if s.radius == 0.0:
                return []
            s.radius = -s.radius
            self._update_accumulated_positions()
            return indices

        old_props = [
            {
                "radius": float(self.surfaces[idx].radius),
                "conic": float(self.surfaces[idx].conic),
                "asphere_coeffs": list(self.surfaces[idx].asphere_coeffs),
                "semi_diameter": float(self.surfaces[idx].semi_diameter),
                "surface_type": self.surfaces[idx].surface_type,
                "thickness": float(self.surfaces[idx].thickness),
                "glass": str(self.surfaces[idx].glass),
                "name": str(self.surfaces[idx].name),
            }
            for idx in indices
        ]

        for k in range(n):
            old_geom = old_props[n - 1 - k]
            surf = self.surfaces[indices[k]]

            surf.radius = -old_geom["radius"] if old_geom["radius"] != 0.0 else 0.0
            surf.conic = old_geom["conic"]
            surf.asphere_coeffs = list(old_geom["asphere_coeffs"])
            surf.semi_diameter = old_geom["semi_diameter"]
            surf.surface_type = old_geom["surface_type"]

            if k < n - 1:
                old_mat = old_props[n - 2 - k]
                surf.thickness = old_mat["thickness"]
                surf.glass = old_mat["glass"]
            else:
                surf.thickness = old_props[-1]["thickness"]
                surf.glass = "AIR"

        self._update_accumulated_positions()
        return indices

    def rotate_element(
        self,
        element_idx: Union[int, str] = None,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        tilt_z: float = 0.0,
        pivot: Union[str, Tuple[float, float, float], List[float], np.ndarray, float] = "front_vertex",
        front_surface_index: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Rotates all surfaces belonging to a multi-surface lens element as a rigid body
        around a designated pivot point without creating wedge distortion.
        """
        if element_idx is None:
            element_idx = front_surface_index
        if element_idx is None:
            return
        from optics_app.engine.tracer import build_rotation_matrix

        indices = self._get_element_indices(element_idx)
        if not indices:
            return

        front_idx = indices[0]
        n = len(indices)

        z_positions = [0.0]
        for idx in range(n - 1):
            s_idx = indices[idx]
            z_positions.append(z_positions[-1] + self.surfaces[s_idx].thickness)

        total_glass_thickness = z_positions[-1] if n > 1 else self.surfaces[front_idx].thickness

        if isinstance(pivot, str):
            p_str = pivot.lower().strip()
            if p_str in ("front_vertex", "front", "vertex"):
                p_piv = np.array([0.0, 0.0, 0.0], dtype=float)
            elif p_str in ("center_of_mass", "center", "com", "mid", "middle"):
                p_piv = np.array([0.0, 0.0, total_glass_thickness / 2.0], dtype=float)
            elif p_str in ("rear_vertex", "back_vertex", "rear", "back"):
                p_piv = np.array([0.0, 0.0, total_glass_thickness], dtype=float)
            else:
                p_piv = np.array([0.0, 0.0, 0.0], dtype=float)
        elif isinstance(pivot, (list, tuple, np.ndarray)):
            if len(pivot) == 3:
                p_piv = np.array(pivot, dtype=float)
            elif len(pivot) == 1:
                p_piv = np.array([0.0, 0.0, float(pivot[0])], dtype=float)
            else:
                p_piv = np.array([0.0, 0.0, 0.0], dtype=float)
        elif isinstance(pivot, (int, float)):
            p_piv = np.array([0.0, 0.0, float(pivot)], dtype=float)
        else:
            p_piv = np.array([0.0, 0.0, 0.0], dtype=float)

        R_elem = build_rotation_matrix(tilt_x, tilt_y, tilt_z)

        for j, s_idx in enumerate(indices):
            surf = self.surfaces[s_idx]
            p_local = np.array([0.0, 0.0, z_positions[j]], dtype=float)
            p_rotated = p_piv + R_elem @ (p_local - p_piv)

            surf.tilt_x = float(tilt_x)
            surf.tilt_y = float(tilt_y)
            surf.tilt_z = float(tilt_z)
            surf.decenter_x = float(p_rotated[0])
            surf.decenter_y = float(p_rotated[1])
            surf.decenter_z = float(p_rotated[2] - z_positions[j])

        self._update_accumulated_positions()

    rotate_lens_element = rotate_element

    def decenter_lens_element(
        self,
        front_surface_index: Union[int, str],
        decenter_x: float = 0.0,
        decenter_y: float = 0.0,
        decenter_z: float = 0.0,
        tilt_x: Optional[float] = None,
        tilt_y: Optional[float] = None,
        tilt_z: Optional[float] = None,
        pivot: Union[str, Tuple[float, float, float], List[float], np.ndarray, float] = "front_vertex"
    ) -> None:
        """Translates/decenters all surfaces in an optical element."""
        from optics_app.engine.tracer import build_rotation_matrix

        indices = self._get_element_indices(front_surface_index)
        if not indices:
            return

        if tilt_x is None and tilt_y is None and tilt_z is None:
            for s_idx in indices:
                surf = self.surfaces[s_idx]
                surf.decenter_x = float(decenter_x)
                surf.decenter_y = float(decenter_y)
                surf.decenter_z = float(decenter_z)
        else:
            front_surf = self.surfaces[indices[0]]
            tx = float(tilt_x) if tilt_x is not None else float(front_surf.tilt_x)
            ty = float(tilt_y) if tilt_y is not None else float(front_surf.tilt_y)
            tz = float(tilt_z) if tilt_z is not None else float(front_surf.tilt_z)

            z_positions = [0.0]
            for idx in range(len(indices) - 1):
                s_idx = indices[idx]
                z_positions.append(z_positions[-1] + self.surfaces[s_idx].thickness)
            total_glass = z_positions[-1] if len(indices) > 1 else front_surf.thickness

            if isinstance(pivot, str):
                p_str = pivot.lower().strip()
                if p_str in ("center_of_mass", "center", "com", "mid"):
                    p_piv = np.array([0.0, 0.0, total_glass / 2.0], dtype=float)
                elif p_str in ("rear_vertex", "back_vertex", "rear", "back"):
                    p_piv = np.array([0.0, 0.0, total_glass], dtype=float)
                else:
                    p_piv = np.array([0.0, 0.0, 0.0], dtype=float)
            else:
                p_piv = np.array([0.0, 0.0, 0.0], dtype=float)

            R_elem = build_rotation_matrix(tx, ty, tz)
            disp_vec = np.array([float(decenter_x), float(decenter_y), float(decenter_z)], dtype=float)

            for j, s_idx in enumerate(indices):
                surf = self.surfaces[s_idx]
                p_local = np.array([0.0, 0.0, z_positions[j]], dtype=float)
                p_trans = p_piv + R_elem @ (p_local - p_piv) + disp_vec
                surf.tilt_x = tx
                surf.tilt_y = ty
                surf.tilt_z = tz
                surf.decenter_x = float(p_trans[0])
                surf.decenter_y = float(p_trans[1])
                surf.decenter_z = float(p_trans[2] - z_positions[j])

        self._update_accumulated_positions()

    displace_element = decenter_lens_element

    def tilt_element(
        self,
        element_idx: Union[int, str],
        angle_deg: float = 0.0,
        axis: str = "x",
        tilt_x: Optional[float] = None,
        tilt_y: Optional[float] = None,
        tilt_z: Optional[float] = None,
        pivot: Union[str, Tuple[float, float, float], List[float], np.ndarray, float] = "front_vertex"
    ) -> None:
        """
        Tilt a rigid optical element by specified angle(s) around a chosen axis or 3D angles.
        """
        ax = str(axis).lower().strip()
        tx = tilt_x if tilt_x is not None else (angle_deg if ax == "x" else 0.0)
        ty = tilt_y if tilt_y is not None else (angle_deg if ax == "y" else 0.0)
        tz = tilt_z if tilt_z is not None else (angle_deg if ax == "z" else 0.0)
        self.rotate_element(element_idx, tilt_x=tx, tilt_y=ty, tilt_z=tz, pivot=pivot)

    def offset_element(
        self,
        element_idx: Union[int, str],
        reference_surface_idx: Union[int, str],
        offset_bias_mm: float,
        axis: str = "z"
    ) -> None:
        """
        Move/offset an element relative to another surface with an offset bias.
        For axis='z': adjusts the air gap / thickness between reference_surface_idx and element_idx
        to match offset_bias_mm.
        For axis='x' or 'y': sets decenter relative to reference surface.
        """
        elem_indices = self._get_element_indices(element_idx)
        ref_indices = self._get_element_indices(reference_surface_idx)
        if not elem_indices or not ref_indices:
            return

        elem_front = elem_indices[0]
        ref_idx = ref_indices[-1]

        ax = str(axis).lower().strip()
        if ax == "z":
            if ref_idx < elem_front:
                self.surfaces[elem_front - 1].thickness = float(offset_bias_mm)
            elif ref_idx > elem_front:
                self.surfaces[ref_idx - 1].thickness = float(offset_bias_mm)
            else:
                for idx in elem_indices:
                    self.surfaces[idx].decenter_z = float(offset_bias_mm)
        elif ax == "x":
            ref_dx = float(self.surfaces[ref_idx].decenter_x)
            for idx in elem_indices:
                self.surfaces[idx].decenter_x = ref_dx + float(offset_bias_mm)
        elif ax == "y":
            ref_dy = float(self.surfaces[ref_idx].decenter_y)
            for idx in elem_indices:
                self.surfaces[idx].decenter_y = ref_dy + float(offset_bias_mm)

        self._update_accumulated_positions()

    move_to_surface_offset = offset_element

    def replace_element(
        self,
        element_idx: Union[int, str],
        new_element: Union["SurfaceStack", List[Surface]]
    ) -> List[int]:
        """Replace element in sequence, preserving trailing air gap."""
        if isinstance(new_element, SurfaceStack):
            new_surfs = list(new_element.surfaces)
        else:
            new_surfs = list(new_element)

        if len(new_surfs) >= 3:
            if new_surfs[0].surface_type == SurfaceType.OBJECT or new_surfs[0].name.upper().startswith("OBJ"):
                new_surfs = new_surfs[1:]
            if len(new_surfs) >= 2 and (new_surfs[-1].surface_type == SurfaceType.DETECTOR or new_surfs[-1].name.upper().startswith("IMA")):
                new_surfs = new_surfs[:-1]

        return self.replace_lens_element(element_idx, new_surfs)

    def replace_element_with_catalog(
        self,
        element_idx: Union[int, str],
        part_number: str
    ) -> List[int]:
        """Replace target optical element with a Thorlabs catalog component."""
        catalog = self._get_lens_catalog()
        key = part_number.upper().strip()
        if key not in catalog:
            raise KeyError(f"Catalog lens '{part_number}' not found in database.")

        lens_info = catalog[key]
        surfs = self._build_catalog_surfaces(lens_info=lens_info, key=key)
        for s in surfs:
            s.is_locked = True

        return self.replace_lens_element(element_idx, surfs)


    def _update_accumulated_positions(self) -> None:
        from optics_app.engine.tracer import build_rotation_matrix

        self.ensure_component_uids()
        
        origin = np.array([0.0, 0.0, 0.0], dtype=float)
        frame_R = np.eye(3, dtype=float)
        
        for s_idx, surf in enumerate(self.surfaces):
            # Shift the reference origin by the decenters (which are in the reference frame's coordinate space)
            decenter_vec = frame_R @ np.array([surf.decenter_x, surf.decenter_y, getattr(surf, 'decenter_z', 0.0)], dtype=float)
            surf.global_origin = origin + decenter_vec
            
            R_local_tilt = build_rotation_matrix(surf.tilt_x, surf.tilt_y, surf.tilt_z)
            surf.global_R = frame_R @ R_local_tilt
            
            is_cb = getattr(surf, 'is_coord_break', False) or (str(getattr(surf, 'surface_type', '')).upper() in ("COORDINATE BREAK", "COORDINATE_BREAK", "COORDBRK", "CBREAK"))
            if is_cb:
                frame_R = surf.global_R
                origin = surf.global_origin
                z_axis_out = frame_R[:, 2]
                origin = origin + z_axis_out * surf.thickness
                continue

            z_axis_in = frame_R[:, 2]
            
            next_is_cb = False
            if s_idx + 1 < len(self.surfaces):
                next_s = self.surfaces[s_idx + 1]
                next_is_cb = getattr(next_s, 'is_coord_break', False) or (str(getattr(next_s, 'surface_type', '')).upper() in ("COORDINATE BREAK", "COORDINATE_BREAK", "COORDBRK", "CBREAK"))

            if surf.is_reflective():
                normal = surf.global_R[:, 2]
                if np.dot(z_axis_in, normal) > 0:
                    normal = -normal
                
                if next_is_cb:
                    # Specular fold of the frame before the coordinate break applies its relative tilt
                    z_axis_out = z_axis_in - 2.0 * np.dot(z_axis_in, normal) * normal
                    z_axis_out /= np.linalg.norm(z_axis_out)
                    x_axis_in = frame_R[:, 0]
                    x_axis_out = x_axis_in - np.dot(x_axis_in, z_axis_out) * z_axis_out
                    x_norm = np.linalg.norm(x_axis_out)
                    if x_norm > 1e-6:
                        x_axis_out /= x_norm
                    else:
                        y_axis_in = frame_R[:, 1]
                        y_axis_out = y_axis_in - np.dot(y_axis_in, z_axis_out) * z_axis_out
                        y_axis_out /= np.linalg.norm(y_axis_out)
                        x_axis_out = np.cross(y_axis_out, z_axis_out)
                    frame_R = np.column_stack((x_axis_out, np.cross(z_axis_out, x_axis_out), z_axis_out))
                else:
                    # Check if this is a grating or a TI-PLM with diffractive carrier axis folding
                    is_grating = hasattr(surf, 'surface_type') and getattr(surf, 'surface_type', None) == SurfaceType.GRATING
                    lines = getattr(surf, 'grating_lines_per_mm', 0)
                    try:
                        from optics_app.engine.plm import is_plm, ensure_plm_config
                    except ImportError:
                        from v2.engine.plm import is_plm, ensure_plm_config
                    surf_is_plm = is_plm(surf)
                    
                    if surf_is_plm:
                        plm_cfg = ensure_plm_config(surf)
                        gcfg = plm_cfg.grating
                        ref_lpmm = float(gcfg.reference_lines_per_mm) if gcfg.reference_lines_per_mm is not None else (float(gcfg.lines_per_mm) if gcfg.enabled else 0.0)
                        if gcfg.fold_optical_axis and ref_lpmm > 0:
                            from optics_app.engine.grating import grating_diffract_3d
                            wl = float(plm_cfg.design_wavelength_um or getattr(self, 'primary_wavelength', 1.225))
                            psi_rad = np.radians(float(gcfg.orientation_deg))
                            local_disp = np.array([np.sin(psi_rad), np.cos(psi_rad), 0.0], dtype=float)
                            dir_out, evan = grating_diffract_3d(
                                ray_dir=z_axis_in,
                                grating_normal=normal,
                                line_vector=surf.global_R @ local_disp,
                                line_density_lines_per_mm=ref_lpmm,
                                order=int(gcfg.reference_order if gcfg.reference_lines_per_mm is not None else gcfg.diffraction_order),
                                wavelength_um=wl,
                                is_reflection=True
                            )
                            if not evan and dir_out is not None:
                                z_axis_out = dir_out
                            else:
                                z_axis_out = z_axis_in - 2.0 * np.dot(z_axis_in, normal) * normal
                        else:
                            z_axis_out = z_axis_in - 2.0 * np.dot(z_axis_in, normal) * normal
                    elif is_grating and lines > 0:
                        from optics_app.engine.grating import grating_diffract_3d
                        custom_wl = getattr(surf, 'grating_primary_wavelength_um', None) or getattr(surf, 'grating_blaze_wavelength_um', None)
                        wl = float(custom_wl) if (custom_wl is not None and custom_wl > 0) else getattr(self, 'primary_wavelength', 0.5876)
                        local_disp = np.array([0.0, 1.0, 0.0]) if abs(getattr(surf, 'tilt_x', 0.0)) > abs(getattr(surf, 'tilt_y', 0.0)) else np.array([1.0, 0.0, 0.0])
                        dir_out, evan = grating_diffract_3d(
                            ray_dir=z_axis_in,
                            grating_normal=normal,
                            line_vector=surf.global_R @ local_disp,
                            line_density_lines_per_mm=lines,
                            order=getattr(surf, 'grating_order', 1),
                            wavelength_um=wl,
                            is_reflection=surf.is_reflective()
                        )
                        if not evan and dir_out is not None:
                            z_axis_out = dir_out
                        else:
                            # Fallback to specular
                            z_axis_out = z_axis_in - 2.0 * np.dot(z_axis_in, normal) * normal
                    else:
                        z_axis_out = z_axis_in - 2.0 * np.dot(z_axis_in, normal) * normal
                        
                    z_axis_out /= np.linalg.norm(z_axis_out)
                    
                    # We need a new right-handed coordinate frame for the reflected path.
                    x_axis_in = frame_R[:, 0]
                    x_axis_out = x_axis_in - np.dot(x_axis_in, z_axis_out) * z_axis_out
                    x_norm = np.linalg.norm(x_axis_out)
                    if x_norm > 1e-6:
                        x_axis_out /= x_norm
                    else:
                        y_axis_in = frame_R[:, 1]
                        y_axis_out = y_axis_in - np.dot(y_axis_in, z_axis_out) * z_axis_out
                        y_axis_out /= np.linalg.norm(y_axis_out)
                        x_axis_out = np.cross(y_axis_out, z_axis_out)
                    
                    # Enforce right-handed coordinate system (Y = Z x X)
                    frame_R = np.column_stack((x_axis_out, np.cross(z_axis_out, x_axis_out), z_axis_out))
            else:
                z_axis_out = z_axis_in
                
            origin = origin + z_axis_out * surf.thickness

    def get_total_system_length(self) -> float:
        return sum(s.thickness for s in self.surfaces)

    def propagate_ray(self, ray_x: float = 0.0, ray_y: float = 0.0, ray_z: float = 0.0,
                      dir_x: float = 0.0, dir_y: float = 0.0, dir_z: float = 1.0) -> List[Dict[str, Any]]:
        """
        Propagate ray through the surface stack via OpticalSystem 3D vector ray tracing.
        Returns per-surface ray history list.
        """
        from optics_app.engine.ray import Ray
        from optics_app.engine.tracer import OpticalSystem

        v_dir = np.array([dir_x, dir_y, dir_z], dtype=float)
        v_norm = np.linalg.norm(v_dir)
        if v_norm > 0:
            v_dir = v_dir / v_norm
        else:
            v_dir = np.array([0.0, 0.0, 1.0], dtype=float)

        ray = Ray(pos=np.array([ray_x, ray_y, ray_z], dtype=float), dir=v_dir)
        sys = OpticalSystem(self)
        traced = sys.trace_ray(ray)
        return traced.history

class LensDataEditor:
    def __init__(self, stack: Optional[SurfaceStack] = None):
        self.stack = stack if stack is not None else SurfaceStack()

    def add_surface(self, surface: Surface) -> None:
        self.stack.add_surface(surface)

    def remove_surface(self, index: int) -> None:
        self.stack.remove_surface(index)

    def get_surface(self, index: int) -> Optional[Surface]:
        if 0 <= index < len(self.stack.surfaces):
            return self.stack.surfaces[index]
        return None
