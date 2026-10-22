"""
Unit and Integration Tests for AOM and EOM Modulator Engine Components (Milestone 1).

Verifies:
1. AOMConfig dataclass: creation, defaults, fields, to_dict(), from_dict(), copy(), mutation isolation.
2. EOMConfig dataclass: creation, defaults, fields, to_dict(), from_dict(), copy(), mutation isolation.
3. Surface integration: aom_config and eom_config optional fields, __post_init__ auto-parsing,
   clean serialization and deserialization via to_dict() and from_dict().
4. OpticalComponent registration: AOM and EOM registered as component types, factories,
   and active mechanism channels.
5. SurfaceStack and LDE persistence: to_dict/from_dict and to_json/from_json preserving
   AOM and EOM configurations without loss.
6. SpectrometerSystem persistence: Schema v5 JSON round-trip with active modulators.
7. Dual-package parity across optics_app/ and v2/.
"""

import json
import os
import tempfile
import unittest
import numpy as np

# optics_app imports
from optics_app.engine.components import (
    AOMConfig,
    EOMConfig,
    OpticalComponent,
    ActiveMechanism,
    MechanismChannel,
    ComponentType,
    MechanismType,
    create_aom_component,
    create_eom_component,
    create_aom_mechanism,
    create_eom_mechanism,
)
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.models import SpectrometerSystem

# v2 imports for cross-package parity
import v2.engine.components as v2_comp
import v2.engine.surface as v2_surf
import v2.engine.lde as v2_lde
import v2.engine.models as v2_models


class TestAOMConfig(unittest.TestCase):
    """Test AOMConfig dataclass definition, serialization, and copying."""

    def test_default_initialization(self):
        cfg = AOMConfig()
        self.assertEqual(cfg.material, "TeO2_slow")
        self.assertAlmostEqual(cfg.rf_frequency_mhz, 80.0)
        self.assertAlmostEqual(cfg.rf_power_w, 1.0)
        self.assertAlmostEqual(cfg.active_aperture_mm, 2.0)
        self.assertAlmostEqual(cfg.interaction_length_mm, 10.0)
        self.assertAlmostEqual(cfg.bragg_angle_mrad, 0.0)
        self.assertAlmostEqual(cfg.diffraction_efficiency, 0.0)

    def test_custom_initialization(self):
        cfg = AOMConfig(
            material="FusedSilica",
            rf_frequency_mhz=120.0,
            rf_power_w=2.5,
            active_aperture_mm=3.5,
            interaction_length_mm=15.0,
            bragg_angle_mrad=12.4,
            diffraction_efficiency=0.88,
        )
        self.assertEqual(cfg.material, "FusedSilica")
        self.assertAlmostEqual(cfg.rf_frequency_mhz, 120.0)
        self.assertAlmostEqual(cfg.rf_power_w, 2.5)
        self.assertAlmostEqual(cfg.active_aperture_mm, 3.5)
        self.assertAlmostEqual(cfg.interaction_length_mm, 15.0)
        self.assertAlmostEqual(cfg.bragg_angle_mrad, 12.4)
        self.assertAlmostEqual(cfg.diffraction_efficiency, 0.88)

    def test_to_dict_and_from_dict_roundtrip(self):
        cfg = AOMConfig(
            material="TeO2_long",
            rf_frequency_mhz=200.0,
            rf_power_w=1.8,
            active_aperture_mm=1.5,
            interaction_length_mm=8.0,
            bragg_angle_mrad=8.5,
            diffraction_efficiency=0.92,
        )
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["material"], "TeO2_long")
        self.assertEqual(d["rf_frequency_mhz"], 200.0)
        self.assertEqual(d["rf_power_w"], 1.8)
        self.assertEqual(d["active_aperture_mm"], 1.5)
        self.assertEqual(d["interaction_length_mm"], 8.0)
        self.assertEqual(d["bragg_angle_mrad"], 8.5)
        self.assertEqual(d["diffraction_efficiency"], 0.92)

        restored = AOMConfig.from_dict(d)
        self.assertEqual(restored.material, cfg.material)
        self.assertAlmostEqual(restored.rf_frequency_mhz, cfg.rf_frequency_mhz)
        self.assertAlmostEqual(restored.rf_power_w, cfg.rf_power_w)
        self.assertAlmostEqual(restored.active_aperture_mm, cfg.active_aperture_mm)
        self.assertAlmostEqual(restored.interaction_length_mm, cfg.interaction_length_mm)
        self.assertAlmostEqual(restored.bragg_angle_mrad, cfg.bragg_angle_mrad)
        self.assertAlmostEqual(restored.diffraction_efficiency, cfg.diffraction_efficiency)

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "material": "Quartz",
            "rf_frequency_mhz": 80.0,
            "rf_power_w": 1.0,
            "active_aperture_mm": 2.0,
            "interaction_length_mm": 10.0,
            "bragg_angle_mrad": 5.0,
            "diffraction_efficiency": 0.75,
            "extra_unrecognized_key": "some_value",
            "another_field": 12345,
        }
        cfg = AOMConfig.from_dict(d)
        self.assertEqual(cfg.material, "Quartz")
        self.assertAlmostEqual(cfg.diffraction_efficiency, 0.75)

    def test_copy_isolation(self):
        original = AOMConfig(
            material="TeO2_slow",
            rf_frequency_mhz=80.0,
            rf_power_w=1.0,
            active_aperture_mm=2.0,
            interaction_length_mm=10.0,
            bragg_angle_mrad=6.2,
            diffraction_efficiency=0.85,
        )
        cloned = original.copy()
        self.assertIsNot(original, cloned)
        self.assertEqual(cloned.material, original.material)
        self.assertAlmostEqual(cloned.rf_power_w, original.rf_power_w)

        # Mutate clone, ensure original unaffected
        cloned.rf_power_w = 3.0
        cloned.material = "FusedSilica"
        self.assertAlmostEqual(original.rf_power_w, 1.0)
        self.assertEqual(original.material, "TeO2_slow")


class TestEOMConfig(unittest.TestCase):
    """Test EOMConfig dataclass definition, serialization, and copying."""

    def test_default_initialization(self):
        cfg = EOMConfig()
        self.assertEqual(cfg.crystal, "LiNbO3")
        self.assertAlmostEqual(cfg.v_pi_volts, 200.0)
        self.assertAlmostEqual(cfg.applied_voltage_volts, 0.0)
        self.assertAlmostEqual(cfg.bias_voltage_volts, 0.0)
        self.assertAlmostEqual(cfg.modulation_freq_mhz, 10.0)
        self.assertAlmostEqual(cfg.extinction_ratio, 1000.0)
        self.assertAlmostEqual(cfg.transmission, 0.0)

    def test_custom_initialization(self):
        cfg = EOMConfig(
            crystal="RTP",
            v_pi_volts=240.0,
            applied_voltage_volts=120.0,
            bias_voltage_volts=30.0,
            modulation_freq_mhz=20.0,
            extinction_ratio=2500.0,
            transmission=0.95,
        )
        self.assertEqual(cfg.crystal, "RTP")
        self.assertAlmostEqual(cfg.v_pi_volts, 240.0)
        self.assertAlmostEqual(cfg.applied_voltage_volts, 120.0)
        self.assertAlmostEqual(cfg.bias_voltage_volts, 30.0)
        self.assertAlmostEqual(cfg.modulation_freq_mhz, 20.0)
        self.assertAlmostEqual(cfg.extinction_ratio, 2500.0)
        self.assertAlmostEqual(cfg.transmission, 0.95)

    def test_to_dict_and_from_dict_roundtrip(self):
        cfg = EOMConfig(
            crystal="DKDP",
            v_pi_volts=3200.0,
            applied_voltage_volts=1600.0,
            bias_voltage_volts=100.0,
            modulation_freq_mhz=5.0,
            extinction_ratio=5000.0,
            transmission=0.98,
        )
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["crystal"], "DKDP")
        self.assertEqual(d["v_pi_volts"], 3200.0)
        self.assertEqual(d["applied_voltage_volts"], 1600.0)
        self.assertEqual(d["bias_voltage_volts"], 100.0)
        self.assertEqual(d["modulation_freq_mhz"], 5.0)
        self.assertEqual(d["extinction_ratio"], 5000.0)
        self.assertEqual(d["transmission"], 0.98)

        restored = EOMConfig.from_dict(d)
        self.assertEqual(restored.crystal, cfg.crystal)
        self.assertAlmostEqual(restored.v_pi_volts, cfg.v_pi_volts)
        self.assertAlmostEqual(restored.applied_voltage_volts, cfg.applied_voltage_volts)
        self.assertAlmostEqual(restored.bias_voltage_volts, cfg.bias_voltage_volts)
        self.assertAlmostEqual(restored.modulation_freq_mhz, cfg.modulation_freq_mhz)
        self.assertAlmostEqual(restored.extinction_ratio, cfg.extinction_ratio)
        self.assertAlmostEqual(restored.transmission, cfg.transmission)

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "crystal": "BBO",
            "v_pi_volts": 250.0,
            "applied_voltage_volts": 50.0,
            "bias_voltage_volts": 0.0,
            "modulation_freq_mhz": 15.0,
            "extinction_ratio": 2000.0,
            "transmission": 0.91,
            "unknown_metadata": "dummy",
        }
        cfg = EOMConfig.from_dict(d)
        self.assertEqual(cfg.crystal, "BBO")
        self.assertAlmostEqual(cfg.v_pi_volts, 250.0)

    def test_copy_isolation(self):
        original = EOMConfig(
            crystal="LiNbO3",
            v_pi_volts=180.0,
            applied_voltage_volts=90.0,
            bias_voltage_volts=0.0,
            modulation_freq_mhz=10.0,
            extinction_ratio=1500.0,
            transmission=0.50,
        )
        cloned = original.copy()
        self.assertIsNot(original, cloned)
        self.assertEqual(cloned.crystal, original.crystal)
        self.assertAlmostEqual(cloned.applied_voltage_volts, original.applied_voltage_volts)

        # Mutate clone, ensure original unaffected
        cloned.applied_voltage_volts = 180.0
        cloned.crystal = "RTP"
        self.assertAlmostEqual(original.applied_voltage_volts, 90.0)
        self.assertEqual(original.crystal, "LiNbO3")


class TestSurfaceModulatorIntegration(unittest.TestCase):
    """Test Surface dataclass integration with aom_config and eom_config."""

    def test_surface_with_aom_config(self):
        aom_cfg = AOMConfig(
            material="TeO2_slow",
            rf_frequency_mhz=80.0,
            rf_power_w=1.2,
            active_aperture_mm=2.5,
            interaction_length_mm=12.0,
            bragg_angle_mrad=7.1,
            diffraction_efficiency=0.86,
        )
        surf = Surface(
            name="AOM Crystal",
            surface_type=SurfaceType.STANDARD,
            thickness=12.0,
            glass="AIR",
            aom_config=aom_cfg,
        )
        self.assertIs(surf.aom_config, aom_cfg)
        self.assertIsNone(surf.eom_config)

        # Serialization to dict
        sd = surf.to_dict()
        self.assertIn("aom_config", sd)
        self.assertIsInstance(sd["aom_config"], dict)
        self.assertEqual(sd["aom_config"]["material"], "TeO2_slow")
        self.assertAlmostEqual(sd["aom_config"]["rf_power_w"], 1.2)

        # Deserialization from dict
        restored = Surface.from_dict(sd)
        self.assertIsInstance(restored.aom_config, AOMConfig)
        self.assertEqual(restored.aom_config.material, "TeO2_slow")
        self.assertAlmostEqual(restored.aom_config.rf_power_w, 1.2)
        self.assertAlmostEqual(restored.aom_config.diffraction_efficiency, 0.86)

    def test_surface_with_eom_config(self):
        eom_cfg = EOMConfig(
            crystal="RTP",
            v_pi_volts=240.0,
            applied_voltage_volts=120.0,
            bias_voltage_volts=10.0,
            modulation_freq_mhz=25.0,
            extinction_ratio=3000.0,
            transmission=0.94,
        )
        surf = Surface(
            name="EOM Pockels Cell",
            surface_type=SurfaceType.STANDARD,
            thickness=15.0,
            glass="AIR",
            eom_config=eom_cfg,
        )
        self.assertIs(surf.eom_config, eom_cfg)
        self.assertIsNone(surf.aom_config)

        # Serialization to dict
        sd = surf.to_dict()
        self.assertIn("eom_config", sd)
        self.assertIsInstance(sd["eom_config"], dict)
        self.assertEqual(sd["eom_config"]["crystal"], "RTP")
        self.assertAlmostEqual(sd["eom_config"]["v_pi_volts"], 240.0)

        # Deserialization from dict
        restored = Surface.from_dict(sd)
        self.assertIsInstance(restored.eom_config, EOMConfig)
        self.assertEqual(restored.eom_config.crystal, "RTP")
        self.assertAlmostEqual(restored.eom_config.applied_voltage_volts, 120.0)
        self.assertAlmostEqual(restored.eom_config.extinction_ratio, 3000.0)

    def test_surface_dict_input_auto_conversion(self):
        """Verify Surface(__post_init__) automatically converts dict configs to typed dataclasses."""
        raw_aom = {
            "material": "FusedSilica",
            "rf_frequency_mhz": 100.0,
            "rf_power_w": 2.0,
            "active_aperture_mm": 3.0,
            "interaction_length_mm": 10.0,
            "bragg_angle_mrad": 4.5,
            "diffraction_efficiency": 0.70,
        }
        raw_eom = {
            "crystal": "LiNbO3",
            "v_pi_volts": 190.0,
            "applied_voltage_volts": 95.0,
            "bias_voltage_volts": 0.0,
            "modulation_freq_mhz": 12.0,
            "extinction_ratio": 1200.0,
            "transmission": 0.88,
        }
        surf = Surface(
            name="ModulatorCombo",
            aom_config=raw_aom,
            eom_config=raw_eom,
        )
        self.assertIsInstance(surf.aom_config, AOMConfig)
        self.assertEqual(surf.aom_config.material, "FusedSilica")
        self.assertIsInstance(surf.eom_config, EOMConfig)
        self.assertEqual(surf.eom_config.crystal, "LiNbO3")


class TestModulatorComponentHierarchy(unittest.TestCase):
    """Test OpticalComponent and factory integration for AOM and EOM."""

    def test_component_type_and_mechanism_type_registration(self):
        self.assertEqual(ComponentType.AOM, "aom")
        self.assertEqual(ComponentType.EOM, "eom")
        self.assertEqual(MechanismType.AOM, "aom")
        self.assertEqual(MechanismType.EOM, "eom")

    def test_create_aom_component_factory(self):
        comp, surf = create_aom_component(
            name="AOM_Test",
            material="TeO2_slow",
            rf_frequency_mhz=80.0,
            rf_power_w=1.5,
            active_aperture_mm=2.0,
            interaction_length_mm=10.0,
            bragg_angle_mrad=6.0,
            diffraction_efficiency=0.85,
        )
        self.assertEqual(comp.component_type, "aom")
        self.assertEqual(comp.name, "AOM_Test")
        self.assertIsInstance(comp.aom_config, AOMConfig)
        self.assertEqual(comp.aom_config.material, "TeO2_slow")
        self.assertAlmostEqual(comp.aom_config.rf_power_w, 1.5)

        self.assertIsInstance(surf, Surface)
        self.assertEqual(surf.component_uid, comp.component_uid)
        self.assertEqual(surf.mechanism_type, "AOM")
        self.assertIsInstance(surf.aom_config, AOMConfig)
        self.assertAlmostEqual(surf.aom_config.rf_frequency_mhz, 80.0)

    def test_create_eom_component_factory(self):
        comp, surf = create_eom_component(
            name="EOM_Test",
            crystal="LiNbO3",
            v_pi_volts=200.0,
            applied_voltage_volts=100.0,
            bias_voltage_volts=15.0,
            modulation_freq_mhz=10.0,
            extinction_ratio=2000.0,
            transmission=0.92,
            crystal_length_mm=14.0,
        )
        self.assertEqual(comp.component_type, "eom")
        self.assertEqual(comp.name, "EOM_Test")
        self.assertIsInstance(comp.eom_config, EOMConfig)
        self.assertEqual(comp.eom_config.crystal, "LiNbO3")
        self.assertAlmostEqual(comp.eom_config.v_pi_volts, 200.0)

        self.assertIsInstance(surf, Surface)
        self.assertEqual(surf.component_uid, comp.component_uid)
        self.assertEqual(surf.mechanism_type, "EOM")
        self.assertIsInstance(surf.eom_config, EOMConfig)
        self.assertAlmostEqual(surf.thickness, 14.0)

    def test_create_active_mechanisms(self):
        aom_mech = create_aom_mechanism(
            mechanism_id="AOM_RF",
            target_surface_uid="surf_aom_01",
            min_rf_power_w=0.0,
            max_rf_power_w=3.0,
            current_rf_power_w=1.2,
        )
        self.assertEqual(aom_mech.mechanism_type, "aom")
        self.assertIn("rf_power_w", aom_mech.channels)
        self.assertAlmostEqual(aom_mech.channels["rf_power_w"].current_value, 1.2)
        self.assertEqual(aom_mech.channels["rf_power_w"].unit, "W")

        eom_mech = create_eom_mechanism(
            mechanism_id="EOM_V",
            target_surface_uid="surf_eom_01",
            min_voltage_v=-250.0,
            max_voltage_v=250.0,
            current_voltage_v=50.0,
        )
        self.assertEqual(eom_mech.mechanism_type, "eom")
        self.assertIn("applied_voltage_volts", eom_mech.channels)
        self.assertAlmostEqual(eom_mech.channels["applied_voltage_volts"].current_value, 50.0)
        self.assertEqual(eom_mech.channels["applied_voltage_volts"].unit, "V")

    def test_optical_component_serialization(self):
        aom_cfg = AOMConfig(material="TeO2_slow", rf_frequency_mhz=80.0, rf_power_w=1.0)
        eom_cfg = EOMConfig(crystal="RTP", v_pi_volts=240.0, applied_voltage_volts=0.0)

        comp = OpticalComponent(
            name="ModulatorBox",
            component_type="generic",
            aom_config=aom_cfg,
            eom_config=eom_cfg,
        )
        d = comp.to_dict()
        self.assertIn("aom_config", d)
        self.assertIn("eom_config", d)
        self.assertEqual(d["aom_config"]["material"], "TeO2_slow")
        self.assertEqual(d["eom_config"]["crystal"], "RTP")

        restored = OpticalComponent.from_dict(d)
        self.assertIsInstance(restored.aom_config, AOMConfig)
        self.assertEqual(restored.aom_config.material, "TeO2_slow")
        self.assertIsInstance(restored.eom_config, EOMConfig)
        self.assertEqual(restored.eom_config.crystal, "RTP")


class TestSurfaceStackAndLDESerialization(unittest.TestCase):
    """Test SurfaceStack and LDE serialization/deserialization with AOM and EOM."""

    def test_surface_stack_roundtrip_with_modulators(self):
        stack = SurfaceStack()
        s_obj = Surface(name="Object", thickness=15.0, glass="AIR")
        stack.add_surface(s_obj)

        comp_aom, surf_aom = create_aom_component(
            name="AOM_Attenuator",
            material="TeO2_slow",
            rf_frequency_mhz=80.0,
            rf_power_w=1.2,
            active_aperture_mm=2.5,
            interaction_length_mm=10.0,
            bragg_angle_mrad=6.5,
            diffraction_efficiency=0.82,
        )
        stack.add_surface(surf_aom)
        stack.add_component(comp_aom)

        comp_eom, surf_eom = create_eom_component(
            name="EOM_PulsePicker",
            crystal="DKDP",
            v_pi_volts=3200.0,
            applied_voltage_volts=1600.0,
            bias_voltage_volts=50.0,
            modulation_freq_mhz=10.0,
            extinction_ratio=4000.0,
            transmission=0.96,
        )
        stack.add_surface(surf_eom)
        stack.add_component(comp_eom)

        s_det = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, glass="AIR")
        stack.add_surface(s_det)

        # 1. to_dict / from_dict roundtrip
        stack_dict = stack.to_dict()
        self.assertIsInstance(stack_dict, dict)
        self.assertIn("surfaces", stack_dict)
        self.assertIn("components", stack_dict)

        restored_stack = SurfaceStack.from_dict(stack_dict)
        self.assertEqual(len(restored_stack.surfaces), 4)
        self.assertEqual(len(restored_stack.components), 2)

        # Verify AOM on restored stack
        aom_surface = next(s for s in restored_stack.surfaces if "AOM" in s.name)
        self.assertIsInstance(aom_surface.aom_config, AOMConfig)
        self.assertEqual(aom_surface.aom_config.material, "TeO2_slow")
        self.assertAlmostEqual(aom_surface.aom_config.rf_power_w, 1.2)
        self.assertAlmostEqual(aom_surface.aom_config.diffraction_efficiency, 0.82)

        # Verify EOM on restored stack
        eom_surface = next(s for s in restored_stack.surfaces if "EOM" in s.name)
        self.assertIsInstance(eom_surface.eom_config, EOMConfig)
        self.assertEqual(eom_surface.eom_config.crystal, "DKDP")
        self.assertAlmostEqual(eom_surface.eom_config.v_pi_volts, 3200.0)
        self.assertAlmostEqual(eom_surface.eom_config.extinction_ratio, 4000.0)

        # 2. to_json / from_json string roundtrip
        json_str = stack.to_json()
        self.assertIsInstance(json_str, str)
        restored_from_json = SurfaceStack.from_json(json_str)
        self.assertEqual(len(restored_from_json.surfaces), 4)
        aom_surf_json = next(s for s in restored_from_json.surfaces if "AOM" in s.name)
        self.assertEqual(aom_surf_json.aom_config.material, "TeO2_slow")

    def test_surface_stack_file_json_roundtrip(self):
        stack = SurfaceStack()
        comp_aom, surf_aom = create_aom_component(
            name="AOM_Device",
            material="FusedSilica",
            rf_frequency_mhz=100.0,
            rf_power_w=2.0,
            active_aperture_mm=3.0,
            interaction_length_mm=12.0,
        )
        stack.add_surface(surf_aom)
        stack.add_component(comp_aom)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "stack_with_aom.json")
            stack.to_json(filepath=file_path)
            self.assertTrue(os.path.exists(file_path))

            loaded_stack = SurfaceStack.from_json(file_path)
            self.assertEqual(len(loaded_stack.surfaces), 1)
            loaded_aom = loaded_stack.surfaces[0]
            self.assertIsInstance(loaded_aom.aom_config, AOMConfig)
            self.assertEqual(loaded_aom.aom_config.material, "FusedSilica")
            self.assertAlmostEqual(loaded_aom.aom_config.rf_power_w, 2.0)

    def test_spectrometer_system_json_roundtrip(self):
        """Verify full SpectrometerSystem Schema v5 JSON round-trip with AOM and EOM."""
        stack = SurfaceStack()
        s0 = Surface(name="Source", thickness=25.0, glass="AIR")
        stack.add_surface(s0)

        comp_aom, surf_aom = create_aom_component(
            name="PowerAOM",
            material="TeO2_slow",
            rf_frequency_mhz=80.0,
            rf_power_w=1.4,
            active_aperture_mm=2.0,
            interaction_length_mm=10.0,
            bragg_angle_mrad=6.2,
            diffraction_efficiency=0.85,
        )
        stack.add_surface(surf_aom)
        stack.add_component(comp_aom)

        comp_eom, surf_eom = create_eom_component(
            name="ModulatorEOM",
            crystal="LiNbO3",
            v_pi_volts=180.0,
            applied_voltage_volts=90.0,
            bias_voltage_volts=0.0,
            modulation_freq_mhz=10.0,
            extinction_ratio=1500.0,
            transmission=0.50,
        )
        stack.add_surface(surf_eom)
        stack.add_component(comp_eom)

        s_det = Surface(name="DetectorPlane", surface_type=SurfaceType.DETECTOR, glass="AIR")
        stack.add_surface(s_det)

        system = SpectrometerSystem(surface_stack=stack, name="UltrafastModulatorSystem")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "modulator_system.json")
            system.save_to_json(path)

            # Check JSON contents directly
            with open(path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
            self.assertEqual(raw_json["version"], 5)
            self.assertEqual(len(raw_json["surfaces"]), 4)
            self.assertEqual(len(raw_json["components"]), 2)

            # Load back via SpectrometerSystem.load_from_json
            loaded_sys = SpectrometerSystem.load_from_json(path)
            self.assertEqual(loaded_sys.name, "UltrafastModulatorSystem")
            self.assertEqual(len(loaded_sys.surfaces), 4)
            self.assertEqual(len(loaded_sys.components), 2)

            # Verify AOM reconstructed properly
            aom_surf = next(s for s in loaded_sys.surfaces if "AOM" in s.name)
            self.assertIsInstance(aom_surf.aom_config, AOMConfig)
            self.assertEqual(aom_surf.aom_config.material, "TeO2_slow")
            self.assertAlmostEqual(aom_surf.aom_config.rf_power_w, 1.4)
            self.assertAlmostEqual(aom_surf.aom_config.diffraction_efficiency, 0.85)

            # Verify EOM reconstructed properly
            eom_surf = next(s for s in loaded_sys.surfaces if "EOM" in s.name)
            self.assertIsInstance(eom_surf.eom_config, EOMConfig)
            self.assertEqual(eom_surf.eom_config.crystal, "LiNbO3")
            self.assertAlmostEqual(eom_surf.eom_config.v_pi_volts, 180.0)
            self.assertAlmostEqual(eom_surf.eom_config.applied_voltage_volts, 90.0)


class TestDualPackageParity(unittest.TestCase):
    """Verify exact parity between optics_app/ and v2/ implementations."""

    def test_aom_config_parity(self):
        cfg_v1 = AOMConfig(
            material="TeO2_slow",
            rf_frequency_mhz=80.0,
            rf_power_w=1.2,
            active_aperture_mm=2.5,
            interaction_length_mm=12.0,
            bragg_angle_mrad=6.5,
            diffraction_efficiency=0.85,
        )
        cfg_v2 = v2_comp.AOMConfig(
            material="TeO2_slow",
            rf_frequency_mhz=80.0,
            rf_power_w=1.2,
            active_aperture_mm=2.5,
            interaction_length_mm=12.0,
            bragg_angle_mrad=6.5,
            diffraction_efficiency=0.85,
        )
        self.assertEqual(cfg_v1.to_dict(), cfg_v2.to_dict())

        # Cross-package from_dict
        v2_from_v1 = v2_comp.AOMConfig.from_dict(cfg_v1.to_dict())
        self.assertEqual(v2_from_v1.material, cfg_v1.material)
        self.assertAlmostEqual(v2_from_v1.rf_power_w, cfg_v1.rf_power_w)

        v1_from_v2 = AOMConfig.from_dict(cfg_v2.to_dict())
        self.assertEqual(v1_from_v2.material, cfg_v2.material)
        self.assertAlmostEqual(v1_from_v2.rf_power_w, cfg_v2.rf_power_w)

    def test_eom_config_parity(self):
        cfg_v1 = EOMConfig(
            crystal="RTP",
            v_pi_volts=240.0,
            applied_voltage_volts=120.0,
            bias_voltage_volts=10.0,
            modulation_freq_mhz=20.0,
            extinction_ratio=3000.0,
            transmission=0.92,
        )
        cfg_v2 = v2_comp.EOMConfig(
            crystal="RTP",
            v_pi_volts=240.0,
            applied_voltage_volts=120.0,
            bias_voltage_volts=10.0,
            modulation_freq_mhz=20.0,
            extinction_ratio=3000.0,
            transmission=0.92,
        )
        self.assertEqual(cfg_v1.to_dict(), cfg_v2.to_dict())

        # Cross-package from_dict
        v2_from_v1 = v2_comp.EOMConfig.from_dict(cfg_v1.to_dict())
        self.assertEqual(v2_from_v1.crystal, cfg_v1.crystal)
        self.assertAlmostEqual(v2_from_v1.v_pi_volts, cfg_v1.v_pi_volts)

        v1_from_v2 = EOMConfig.from_dict(cfg_v2.to_dict())
        self.assertEqual(v1_from_v2.crystal, cfg_v2.crystal)
        self.assertAlmostEqual(v1_from_v2.v_pi_volts, cfg_v2.v_pi_volts)

    def test_surface_and_stack_cross_package_roundtrip(self):
        # Create in optics_app
        stack_v1 = SurfaceStack()
        comp_aom, surf_aom = create_aom_component(name="AOM_V1", rf_power_w=1.5)
        comp_eom, surf_eom = create_eom_component(name="EOM_V1", applied_voltage_volts=100.0)
        stack_v1.add_surface(surf_aom)
        stack_v1.add_surface(surf_eom)
        stack_v1.add_component(comp_aom)
        stack_v1.add_component(comp_eom)

        v1_dict = stack_v1.to_dict()

        # Load in v2
        stack_v2 = v2_lde.SurfaceStack.from_dict(v1_dict)
        self.assertEqual(len(stack_v2.surfaces), 2)
        self.assertEqual(len(stack_v2.components), 2)

        aom_v2_surf = stack_v2.surfaces[0]
        self.assertEqual(aom_v2_surf.mechanism_type, "AOM")
        self.assertAlmostEqual(aom_v2_surf.aom_config.rf_power_w, 1.5)

        eom_v2_surf = stack_v2.surfaces[1]
        self.assertEqual(eom_v2_surf.mechanism_type, "EOM")
        self.assertAlmostEqual(eom_v2_surf.eom_config.applied_voltage_volts, 100.0)

        # Re-serialize in v2 and load back in optics_app
        v2_dict = stack_v2.to_dict()
        stack_v1_restored = SurfaceStack.from_dict(v2_dict)
        self.assertEqual(len(stack_v1_restored.surfaces), 2)
        self.assertAlmostEqual(stack_v1_restored.surfaces[0].aom_config.rf_power_w, 1.5)
        self.assertAlmostEqual(stack_v1_restored.surfaces[1].eom_config.applied_voltage_volts, 100.0)


if __name__ == "__main__":
    unittest.main()
