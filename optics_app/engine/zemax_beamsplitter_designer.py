"""
Zemax Beamsplitter Designer & Reference Optical Architectures.
==============================================================
Provides standard Zemax OpticStudio sequential modeling and reference designs
for commercial Thorlabs amplitude (non-polarizing) and polarization beam splitters:

1. Michelson Interferometer (`build_zemax_michelson_interferometer`):
   - Source -> Thorlabs BS016 (50:50 non-polarizing cube) -> Reference Arm & Test Arm
   - Arm lengths L1 and L2 (with sub-micron displacement delta_d)
   - Recombination -> Thorlabs AC254-075-A Achromatic Doublet -> Detector
   - Computes: Optical Path Difference (OPD), phase difference, fringe visibility,
     intensity interference curve, and Zemax .zmx prescription export.

2. Mach-Zehnder Interferometer (`build_zemax_mach_zehnder_interferometer`):
   - Dual-arm balanced interferometer with BS1 (BS016) -> Fold Mirrors (PF10-03-P01)
     -> Sample Arm Phase Shift -> BS2 (BS016) recombination
   - Computes complementary dual-port output intensities:
     I1 = I0 * cos^2(Delta_phi / 2), I2 = I0 * sin^2(Delta_phi / 2)
     with strict energy conservation (I1 + I2 = I0).

3. Polarization Beam Splitter Optical Isolator (`build_zemax_pbs_optical_isolator`):
   - Polarization-based optical diode: P-polarized laser -> Thorlabs PBS251 (PBS Cube)
     -> Zero-order QWP at 45 deg (Linear -> Circular) -> Target Specular Reflector
     -> Return pass through QWP (Circular -> S-polarized) -> PBS251 90-degree reflection
     into Beam Dump / Photodiode monitor (BT610)
   - Computes isolation rejection (>35-45 dB), leakage power, and Stokes/Jones vectors.

4. Beam Sampler / Power Pick-Off (`build_zemax_beam_sampler`):
   - Calibrated 90:10 / 70:30 / 10:90 pick-off using Thorlabs BS019 or BSW32
   - Main transmission branch to target + 90-degree reflected monitor branch.

5. Zemax OpticStudio Prescription (.zmx) Export:
   - Formatted sequential coordinate breaks (COORDBRK), standard surfaces (STANDARD),
     mirrors (MIRROR), glass indices (GLAS), and surface comments (COMM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

try:
    from optics_app.engine.surface import Surface, SurfaceType
    from optics_app.engine.lde import SurfaceStack
    from optics_app.engine.components import (
        BeamSplitterConfig,
        OpticalComponent,
        create_beamsplitter_component,
        create_polarizing_beamsplitter_component,
        create_amplitude_beamsplitter_component,
    )
    from optics_app.catalogs.thorlabs import (
        get_catalog_lens,
        get_lens_catalog,
        get_amplitude_beamsplitter_catalog,
        get_polarizing_beamsplitter_catalog,
    )
    from optics_app.io.zmx_parser import export_to_zmx_string
except ImportError:
    from v2.engine.surface import Surface, SurfaceType
    from v2.engine.lde import SurfaceStack
    from v2.engine.components import (
        BeamSplitterConfig,
        OpticalComponent,
        create_beamsplitter_component,
        create_polarizing_beamsplitter_component,
        create_amplitude_beamsplitter_component,
    )
    from v2.catalogs.thorlabs import (
        get_catalog_lens,
        get_lens_catalog,
        get_amplitude_beamsplitter_catalog,
        get_polarizing_beamsplitter_catalog,
    )
    from v2.io.zmx_parser import export_to_zmx_string


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MichelsonDesignResult:
    """Design metrics and optical systems for a Zemax Michelson Interferometer."""
    name: str
    bs_part: str
    arm1_length_mm: float
    arm2_length_mm: float
    displacement_delta_um: float
    opd_um: float
    wavelength_um: float
    phase_difference_rad: float
    phase_difference_deg: float
    intensity_arm1: float
    intensity_arm2: float
    interference_intensity: float
    fringe_visibility: float
    system: SurfaceStack
    reference_arm_stack: SurfaceStack
    test_arm_stack: SurfaceStack
    zmx_text: str
    components_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bs_part": self.bs_part,
            "arm1_length_mm": self.arm1_length_mm,
            "arm2_length_mm": self.arm2_length_mm,
            "displacement_delta_um": self.displacement_delta_um,
            "opd_um": self.opd_um,
            "wavelength_um": self.wavelength_um,
            "phase_difference_rad": self.phase_difference_rad,
            "phase_difference_deg": self.phase_difference_deg,
            "intensity_arm1": self.intensity_arm1,
            "intensity_arm2": self.intensity_arm2,
            "interference_intensity": self.interference_intensity,
            "fringe_visibility": self.fringe_visibility,
            "components_used": self.components_used,
        }


@dataclass
class MachZehnderDesignResult:
    """Design metrics and optical systems for a Zemax Mach-Zehnder Interferometer."""
    name: str
    bs1_part: str
    bs2_part: str
    arm_length_mm: float
    phase_shift_deg: float
    phase_shift_rad: float
    wavelength_um: float
    port1_intensity: float
    port2_intensity: float
    total_transmitted_intensity: float
    extinction_ratio_port2_db: float
    system: SurfaceStack
    arm1_stack: SurfaceStack
    arm2_stack: SurfaceStack
    zmx_text: str
    components_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bs1_part": self.bs1_part,
            "bs2_part": self.bs2_part,
            "arm_length_mm": self.arm_length_mm,
            "phase_shift_deg": self.phase_shift_deg,
            "phase_shift_rad": self.phase_shift_rad,
            "wavelength_um": self.wavelength_um,
            "port1_intensity": self.port1_intensity,
            "port2_intensity": self.port2_intensity,
            "total_transmitted_intensity": self.total_transmitted_intensity,
            "extinction_ratio_port2_db": self.extinction_ratio_port2_db,
            "components_used": self.components_used,
        }


@dataclass
class PBSOpticalIsolatorResult:
    """Design metrics and optical systems for a Polarizing Beam Splitter Optical Isolator."""
    name: str
    pbs_part: str
    wavelength_um: float
    input_polarization: str
    forward_transmitted_power: float
    forward_jones: Tuple[complex, complex]
    target_reflected_power: float
    return_polarization: str
    return_jones: Tuple[complex, complex]
    dump_rejected_power: float
    laser_leakage_power: float
    optical_isolation_db: float
    system: SurfaceStack
    zmx_text: str
    components_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pbs_part": self.pbs_part,
            "wavelength_um": self.wavelength_um,
            "input_polarization": self.input_polarization,
            "forward_transmitted_power": self.forward_transmitted_power,
            "forward_jones": [str(self.forward_jones[0]), str(self.forward_jones[1])],
            "target_reflected_power": self.target_reflected_power,
            "return_polarization": self.return_polarization,
            "return_jones": [str(self.return_jones[0]), str(self.return_jones[1])],
            "dump_rejected_power": self.dump_rejected_power,
            "laser_leakage_power": self.laser_leakage_power,
            "optical_isolation_db": self.optical_isolation_db,
            "components_used": self.components_used,
        }


@dataclass
class BeamSamplerResult:
    """Design metrics and optical system for a calibrated beam sampler pick-off."""
    name: str
    bs_part: str
    wavelength_um: float
    input_power_mw: float
    transmitted_power_mw: float
    sampled_power_mw: float
    pick_off_ratio_percent: float
    insertion_loss_db: float
    system: SurfaceStack
    zmx_text: str
    components_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bs_part": self.bs_part,
            "wavelength_um": self.wavelength_um,
            "input_power_mw": self.input_power_mw,
            "transmitted_power_mw": self.transmitted_power_mw,
            "sampled_power_mw": self.sampled_power_mw,
            "pick_off_ratio_percent": self.pick_off_ratio_percent,
            "insertion_loss_db": self.insertion_loss_db,
            "components_used": self.components_used,
        }


# =============================================================================
# 1. Michelson Interferometer Designer (Zemax Approach)
# =============================================================================

def build_zemax_michelson_interferometer(
    bs_part: str = "BS016",
    arm_length_mm: float = 100.0,
    d_delta_um: float = 0.0,
    wavelength_um: float = 0.6328,
    imaging_lens_part: Optional[str] = "AC254-075-A",
    reference_mirror_part: str = "PF10-03-P01",
    test_mirror_part: str = "PF10-03-P01",
    detector_distance_mm: Optional[float] = None,
) -> MichelsonDesignResult:
    """
    Synthesizes a precision Michelson Interferometer following Zemax OpticStudio sequential conventions:
    1. Source: Collimated beam (e.g. HeNe 632.8 nm).
    2. Input distance: 30 mm to Beamsplitter.
    3. Thorlabs Beamsplitter Cube (BS016 50:50, 25.4mm cube, N-BK7):
       - Entrance Face (AR coated, flat, thickness = cube_size/2)
       - 45-degree hypotenuse splitting interface (R:T = 50:50)
    4. Reference Arm (Arm 1): length L1 = arm_length_mm to Reference Mirror (PF10-03-P01).
       Reflects normal to mirror and returns to BS.
    5. Test Arm (Arm 2): length L2 = arm_length_mm + d_delta_um * 1e-3 to Test Mirror.
       Reflects normal to mirror and returns to BS.
    6. Recombination at BS hypotenuse: recombines Arm 1 & Arm 2 wavefronts.
    7. Imaging Doublet (Thorlabs AC254-075-A, f=75mm) focusing onto Detector.
    """
    bs_info = get_catalog_lens(bs_part)
    cube_size = float(bs_info.get("dimensions_mm", {}).get("x", 25.4)) if isinstance(bs_info.get("dimensions_mm"), dict) else 25.4
    half_size = cube_size / 2.0
    substrate = str(bs_info.get("surfaces", [{}])[0].get("glass", "N-BK7"))

    split_r = float(bs_info.get("split_ratio_R", 0.50))
    split_t = float(bs_info.get("split_ratio_T", 0.50))

    # Calculate OPD and phase difference
    l1_mm = float(arm_length_mm)
    delta_mm = float(d_delta_um) * 1e-3
    l2_mm = l1_mm + delta_mm
    opd_mm = 2.0 * delta_mm
    opd_um = opd_mm * 1000.0

    k_wave = 2.0 * math.pi / float(wavelength_um)
    phase_diff_rad = k_wave * opd_um
    phase_diff_deg = math.degrees(phase_diff_rad) % 360.0

    # Intensities from each arm (double pass through BS: R*T for both arms)
    # Arm 1: BS transmits (T) -> Mirror (R_m=0.99) -> BS reflects (R) -> I1 = I0 * T * 0.99 * R
    # Arm 2: BS reflects (R) -> Mirror (R_m=0.99) -> BS transmits (T) -> I2 = I0 * R * 0.99 * T
    r_mirror = 0.99
    i1 = split_t * r_mirror * split_r
    i2 = split_r * r_mirror * split_t

    # Interference equation: I = I1 + I2 + 2*sqrt(I1*I2)*cos(Delta_phi)
    interf_intensity = i1 + i2 + 2.0 * math.sqrt(i1 * i2) * math.cos(phase_diff_rad)
    fringe_visibility = (2.0 * math.sqrt(i1 * i2)) / (i1 + i2) if (i1 + i2) > 0 else 0.0

    # Construct Sequential Recombined SurfaceStack (Zemax OpticStudio format)
    stack = SurfaceStack()
    stack.name = f"Thorlabs Michelson Interferometer ({bs_part})"
    stack.primary_wavelength = float(wavelength_um)
    stack.wavelengths = [float(wavelength_um)]

    # S0: Object / Laser Source
    s_obj = Surface(name="HeNe Laser Source (Collimated)", surface_type=SurfaceType.OBJECT, thickness=30.0, glass="AIR", semi_diameter=5.0)
    stack.add_surface(s_obj)

    # S1: BS Entrance Face
    s_bs_in = Surface(name=f"{bs_part} Entrance Face (AR)", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=half_size, glass=substrate, semi_diameter=half_size)
    stack.add_surface(s_bs_in)

    # S2: Hypotenuse Coordinate Break (tilted 45 deg)
    s_cb1 = Surface(name=f"{bs_part} Hypotenuse Tilt 45°", surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=45.0, semi_diameter=half_size * math.sqrt(2))
    s_cb1.is_coord_break = True
    stack.add_surface(s_cb1)

    # S3: Beamsplitter Splitting Surface
    cfg = BeamSplitterConfig(
        part_number=bs_part,
        splitter_type="Cube",
        is_polarizing=False,
        split_ratio_R=split_r,
        split_ratio_T=split_t,
        coating_type="NPBS_50_50",
        substrate_glass=substrate,
        cube_side_length_mm=cube_size,
        clear_aperture_mm=cube_size,
    )
    s_split = Surface(name=f"{bs_part} 50:50 Splitting Interface", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0, glass=substrate, semi_diameter=half_size * math.sqrt(2))
    s_split.is_beam_splitter = True
    s_split.beamsplitter_config = cfg
    stack.add_surface(s_split)

    # S4: Coordinate Break Undo Tilt
    s_cb2 = Surface(name=f"{bs_part} Coordinate Break Undo Tilt", surface_type=SurfaceType.COORDINATE_BREAK, tilt_x=-45.0, thickness=half_size, glass=substrate, semi_diameter=half_size)
    s_cb2.is_coord_break = True
    stack.add_surface(s_cb2)

    # S5: BS Exit Face to Test Arm
    s_bs_exit_test = Surface(name=f"{bs_part} Exit Face (Test Arm)", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=l2_mm - half_size, glass="AIR", semi_diameter=half_size)
    stack.add_surface(s_bs_exit_test)

    # S6: Test Arm Mirror (PF10-03-P01)
    s_test_mirror = Surface(name=f"Test Mirror ({test_mirror_part})", surface_type=SurfaceType.MIRROR, radius=0.0, thickness=-(l2_mm - half_size), glass="MIRROR", semi_diameter=12.7)
    stack.add_surface(s_test_mirror)

    # S7: Return to BS Splitting Surface (recombination pass)
    s_recomb = Surface(name=f"{bs_part} Recombination Interface", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=half_size, glass=substrate, semi_diameter=half_size)
    s_recomb.is_beam_splitter = True
    s_recomb.beamsplitter_config = cfg
    stack.add_surface(s_recomb)

    # S8: Output Face towards Imaging Lens
    s_bs_out = Surface(name=f"{bs_part} Output Face", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=40.0, glass="AIR", semi_diameter=half_size)
    stack.add_surface(s_bs_out)

    components_used = [bs_part, reference_mirror_part, test_mirror_part]

    # S9-S11: Optional Imaging Achromat (e.g. AC254-075-A)
    if imaging_lens_part:
        lens_info = get_catalog_lens(imaging_lens_part)
        components_used.append(imaging_lens_part)
        for s_idx, s_raw in enumerate(lens_info.get("surfaces", [])):
            s_doublet = Surface(
                name=f"{imaging_lens_part}_S{s_idx+1}",
                surface_type=SurfaceType.STANDARD,
                radius=float(s_raw.get("radius_mm", 0.0)),
                thickness=float(s_raw.get("thickness_mm", 5.0)),
                glass=str(s_raw.get("glass", "AIR")),
                semi_diameter=float(lens_info.get("semi_diameter", 12.7)),
            )
            stack.add_surface(s_doublet)

        bfl = float(lens_info.get("back_focal_length_mm", 69.8))
        img_dist = detector_distance_mm if detector_distance_mm is not None else bfl
        stack.surfaces[-1].thickness = img_dist
    else:
        img_dist = detector_distance_mm if detector_distance_mm is not None else 50.0
        stack.surfaces[-1].thickness = img_dist

    # Final Image Surface / Detector
    s_det = Surface(name="Interferogram CMOS Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=10.0)
    stack.add_surface(s_det)

    # Sub-stacks for individual arms
    ref_stack = SurfaceStack()
    ref_stack.name = f"Michelson Reference Arm ({l1_mm:.3f} mm)"
    ref_stack.add_surface(Surface(name="BS Reference Port", thickness=l1_mm, glass="AIR", semi_diameter=half_size))
    ref_stack.add_surface(Surface(name=f"Reference Mirror ({reference_mirror_part})", surface_type=SurfaceType.MIRROR, thickness=-l1_mm, glass="MIRROR", semi_diameter=12.7))
    ref_stack.add_surface(Surface(name="BS Return Port", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=half_size))

    test_stack = SurfaceStack()
    test_stack.name = f"Michelson Test Arm ({l2_mm:.4f} mm, OPD={opd_um:+.2f} µm)"
    test_stack.add_surface(Surface(name="BS Test Port", thickness=l2_mm, glass="AIR", semi_diameter=half_size))
    test_stack.add_surface(Surface(name=f"Test Mirror ({test_mirror_part})", surface_type=SurfaceType.MIRROR, thickness=-l2_mm, glass="MIRROR", semi_diameter=12.7))
    test_stack.add_surface(Surface(name="BS Return Port", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=half_size))

    # Generate Zemax OpticStudio prescription
    zmx_text = export_to_zmx_string(stack, title=stack.name, wavelengths=[float(wavelength_um)])

    return MichelsonDesignResult(
        name="Thorlabs Michelson Interferometer",
        bs_part=bs_part,
        arm1_length_mm=l1_mm,
        arm2_length_mm=l2_mm,
        displacement_delta_um=float(d_delta_um),
        opd_um=opd_um,
        wavelength_um=float(wavelength_um),
        phase_difference_rad=phase_diff_rad,
        phase_difference_deg=phase_diff_deg,
        intensity_arm1=i1,
        intensity_arm2=i2,
        interference_intensity=interf_intensity,
        fringe_visibility=fringe_visibility,
        system=stack,
        reference_arm_stack=ref_stack,
        test_arm_stack=test_stack,
        zmx_text=zmx_text,
        components_used=components_used,
    )


# =============================================================================
# 2. Mach-Zehnder Interferometer Designer (Zemax Approach)
# =============================================================================

def build_zemax_mach_zehnder_interferometer(
    bs1_part: str = "BS016",
    bs2_part: str = "BS016",
    mirror1_part: str = "PF10-03-P01",
    mirror2_part: str = "PF10-03-P01",
    arm_length_mm: float = 100.0,
    phase_shift_deg: float = 0.0,
    wavelength_um: float = 0.6328,
) -> MachZehnderDesignResult:
    """
    Synthesizes a dual-arm Mach-Zehnder Interferometer following Zemax OpticStudio conventions:
    1. Input Source (e.g. HeNe 632.8 nm) -> BS1 (BS016 50:50).
    2. Arm 1 (Transmitted branch): BS1 (T=0.5) -> Fold Mirror 1 (45 deg) -> Sample Cell (Delta_phi) -> BS2 (BS016).
    3. Arm 2 (Reflected branch): BS1 (R=0.5) -> Fold Mirror 2 (45 deg) -> BS2 (BS016).
    4. Recombination at BS2 produces complementary dual output ports:
       - Output Port 1: I1 = I0 * cos^2(Delta_phi / 2)
       - Output Port 2: I2 = I0 * sin^2(Delta_phi / 2)
       - Energy conservation: I1 + I2 = 1.0 (strict).
    """
    phase_rad = math.radians(float(phase_shift_deg))

    # Analytical interference for complementary ports (assuming 50:50 splitters)
    port1_i = math.cos(phase_rad / 2.0) ** 2
    port2_i = math.sin(phase_rad / 2.0) ** 2
    total_i = port1_i + port2_i

    # Extinction ratio for Port 2 (dB) relative to dark null
    ext_ratio_db = -10.0 * math.log10(max(port2_i, 1e-12)) if port2_i > 0 else 120.0

    # Build primary SurfaceStack representing Arm 1 recombination through Port 1
    stack = SurfaceStack()
    stack.name = f"Thorlabs Mach-Zehnder Interferometer ({bs1_part}+{bs2_part})"
    stack.primary_wavelength = float(wavelength_um)
    stack.wavelengths = [float(wavelength_um)]

    # S0: Source
    stack.add_surface(Surface(name="Collimated Laser Input", surface_type=SurfaceType.OBJECT, thickness=25.0, glass="AIR", semi_diameter=4.0))

    # S1-S3: BS1 (Splitting Cube)
    cfg1 = BeamSplitterConfig(
        part_number=bs1_part,
        splitter_type="Cube",
        is_polarizing=False,
        split_ratio_R=0.5,
        split_ratio_T=0.5,
        coating_type="NPBS_50_50",
    )
    stack.add_surface(Surface(name=f"BS1 Entrance Face ({bs1_part})", thickness=12.7, glass="N-BK7", semi_diameter=12.7))
    s_bs1_hyp = Surface(name=f"BS1 50:50 Hypotenuse", thickness=12.7, glass="N-BK7", semi_diameter=12.7, tilt_x=45.0)
    s_bs1_hyp.is_beam_splitter = True
    s_bs1_hyp.beamsplitter_config = cfg1
    stack.add_surface(s_bs1_hyp)
    stack.add_surface(Surface(name=f"BS1 Exit Face", thickness=float(arm_length_mm), glass="AIR", semi_diameter=12.7))

    # S4: Fold Mirror 1 (PF10-03-P01 at 45 deg)
    s_m1 = Surface(name=f"Fold Mirror 1 ({mirror1_part})", surface_type=SurfaceType.MIRROR, thickness=float(arm_length_mm), glass="MIRROR", semi_diameter=12.7, tilt_x=45.0)
    stack.add_surface(s_m1)

    # S5-S7: BS2 (Recombination Cube)
    cfg2 = BeamSplitterConfig(
        part_number=bs2_part,
        splitter_type="Cube",
        is_polarizing=False,
        split_ratio_R=0.5,
        split_ratio_T=0.5,
        coating_type="NPBS_50_50",
    )
    stack.add_surface(Surface(name=f"BS2 Entrance Face ({bs2_part})", thickness=12.7, glass="N-BK7", semi_diameter=12.7))
    s_bs2_hyp = Surface(name=f"BS2 Recombination Hypotenuse", thickness=12.7, glass="N-BK7", semi_diameter=12.7, tilt_x=45.0)
    s_bs2_hyp.is_beam_splitter = True
    s_bs2_hyp.beamsplitter_config = cfg2
    stack.add_surface(s_bs2_hyp)
    stack.add_surface(Surface(name=f"BS2 Exit Face (Port 1)", thickness=30.0, glass="AIR", semi_diameter=12.7))

    # S8: Detector Port 1 (Constructive)
    stack.add_surface(Surface(name="Output Port 1 Detector (In-Phase)", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=10.0))

    # Sub-stacks for Arm 1 and Arm 2
    arm1_stack = SurfaceStack()
    arm1_stack.name = f"Mach-Zehnder Arm 1 (Transmitted Path, Δφ={phase_shift_deg:.1f}°)"
    arm1_stack.add_surface(Surface(name="BS1 Transmitted Port", thickness=float(arm_length_mm), glass="AIR", semi_diameter=12.7))
    arm1_stack.add_surface(Surface(name=f"Mirror 1 ({mirror1_part})", surface_type=SurfaceType.MIRROR, thickness=float(arm_length_mm), glass="MIRROR", semi_diameter=12.7))
    arm1_stack.add_surface(Surface(name="BS2 Input Port", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=12.7))

    arm2_stack = SurfaceStack()
    arm2_stack.name = "Mach-Zehnder Arm 2 (Reflected Reference Path)"
    arm2_stack.add_surface(Surface(name="BS1 Reflected Port", thickness=float(arm_length_mm), glass="AIR", semi_diameter=12.7))
    arm2_stack.add_surface(Surface(name=f"Mirror 2 ({mirror2_part})", surface_type=SurfaceType.MIRROR, thickness=float(arm_length_mm), glass="MIRROR", semi_diameter=12.7))
    arm2_stack.add_surface(Surface(name="BS2 Input Port", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=12.7))

    zmx_text = export_to_zmx_string(stack, title=stack.name, wavelengths=[float(wavelength_um)])

    return MachZehnderDesignResult(
        name="Thorlabs Mach-Zehnder Interferometer",
        bs1_part=bs1_part,
        bs2_part=bs2_part,
        arm_length_mm=float(arm_length_mm),
        phase_shift_deg=float(phase_shift_deg),
        phase_shift_rad=phase_rad,
        wavelength_um=float(wavelength_um),
        port1_intensity=port1_i,
        port2_intensity=port2_i,
        total_transmitted_intensity=total_i,
        extinction_ratio_port2_db=ext_ratio_db,
        system=stack,
        arm1_stack=arm1_stack,
        arm2_stack=arm2_stack,
        zmx_text=zmx_text,
        components_used=[bs1_part, bs2_part, mirror1_part, mirror2_part],
    )


# =============================================================================
# 3. PBS Optical Isolator Designer (Zemax Approach)
# =============================================================================

def build_zemax_pbs_optical_isolator(
    pbs_part: str = "PBS251",
    wavelength_um: float = 0.6328,
    qwp_angle_deg: float = 45.0,
    distance_pbs_to_qwp_mm: float = 30.0,
    distance_qwp_to_target_mm: float = 50.0,
    target_reflectance: float = 0.99,
) -> PBSOpticalIsolatorResult:
    """
    Synthesizes a Polarization Beam Splitter Optical Isolator (optical diode)
    following Zemax OpticStudio conventions:
    1. Linearly polarized laser source: P-polarization [0, 1]^T.
    2. Thorlabs PBS Cube (e.g. PBS251, 1" cube, 620-1000 nm):
       - Transmits P-polarization with high efficiency (Tp >= 90%).
       - Rejects S-polarization to 90-degree port (Rs >= 99.5%).
    3. Quarter-Wave Plate (QWP) at 45 degrees:
       - Retardation delta = pi/2.
       - Transforms linear P-pol into Right-Hand Circular (RHC) polarization.
    4. Target Specular Reflector (R_target = 0.99):
       - Reflection flips handedness: RHC -> LHC.
    5. Return pass through QWP at 45 degrees:
       - Transforms LHC into linear S-polarization [1, 0]^T.
    6. Re-entry into PBS251:
       - Encounters PBS hypotenuse as S-polarization.
       - PBS reflects S-pol at 90 degrees into a Beam Dump / Photodiode (BT610).
       - Only residual leakage Ts (<= 0.005) leaks back into the laser.
    7. Computes optical isolation in dB: -10 * log10(P_leak / P_in).
    """
    pbs_info = get_catalog_lens(pbs_part)
    cube_size = float(pbs_info.get("dimensions_mm", {}).get("x", 25.4)) if isinstance(pbs_info.get("dimensions_mm"), dict) else 25.4
    half_size = cube_size / 2.0
    substrate = str(pbs_info.get("surfaces", [{}])[0].get("glass", "N-BK7"))

    # Catalog extinction and transmission specs
    ext_ratio = float(pbs_info.get("extinction_ratio", 1000.0))
    tp_nom = float(pbs_info.get("split_ratio_T", 0.90))
    rs_nom = float(pbs_info.get("split_ratio_R", 0.995))
    ts_leak = max(1.0 - rs_nom, 1.0 / ext_ratio)

    # 1. Forward Pass:
    # Input P-polarization Jones vector: [0.0, 1.0] (along local Y)
    j_in = np.array([0.0, 1.0], dtype=complex)
    p_fwd = tp_nom  # Forward transmitted power through PBS

    # Double pass through QWP at theta with normal mirror reflection:
    # Acts as an effective half-wave plate (HWP) at angle theta with retardation Gamma = pi.
    # W_hwp(theta) = R(-theta) @ diag(1, -1) @ R(theta)
    # At theta = 45 deg, W_hwp converts P-pol [0, 1] exactly to S-pol [1, 0].
    th_rad = math.radians(float(qwp_angle_deg))
    cos_t = math.cos(th_rad)
    sin_t = math.sin(th_rad)
    r_rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=complex)
    r_inv = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=complex)

    # Single pass QWP matrix (retardation pi/2)
    w_qwp = r_inv @ np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex) @ r_rot
    j_circ = w_qwp @ j_in
    j_circ = j_circ / np.linalg.norm(j_circ)

    # Double-pass effective HWP matrix (retardation pi)
    w_hwp = r_inv @ np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex) @ r_rot
    j_ret = w_hwp @ j_in
    j_ret = j_ret / np.linalg.norm(j_ret)

    p_target = p_fwd * float(target_reflectance)

    # Return state fraction in S and P bases:
    frac_s = float(abs(j_ret[0]) ** 2)
    frac_p = float(abs(j_ret[1]) ** 2)

    # Return power to PBS:
    # S-pol component reflects to dump (Rs * frac_s)
    # P-pol component transmits back to laser (Tp * frac_p)
    # S-pol leakage transmits back to laser (Ts * frac_s)
    p_dump = p_target * (rs_nom * frac_s + (1.0 - tp_nom) * frac_p)
    p_leak = p_target * (ts_leak * frac_s + tp_nom * frac_p)

    # Optical isolation in dB relative to input:
    iso_db = -10.0 * math.log10(max(p_leak, 1e-12))

    # Build SurfaceStack for sequential optical train
    stack = SurfaceStack()
    stack.name = f"Thorlabs PBS Optical Isolator ({pbs_part})"
    stack.primary_wavelength = float(wavelength_um)
    stack.wavelengths = [float(wavelength_um)]

    # S0: Source
    stack.add_surface(Surface(name="Laser Source (Linear P-Pol)", surface_type=SurfaceType.OBJECT, thickness=25.0, glass="AIR", semi_diameter=3.0))

    # S1-S3: PBS251 Cube
    cfg_pbs = BeamSplitterConfig(
        part_number=pbs_part,
        splitter_type="Cube",
        is_polarizing=True,
        split_ratio_R=rs_nom,
        split_ratio_T=tp_nom,
        Rs=rs_nom,
        Rp=1.0 - tp_nom,
        Ts=ts_leak,
        Tp=tp_nom,
        extinction_ratio=ext_ratio,
        substrate_glass=substrate,
        cube_side_length_mm=cube_size,
        clear_aperture_mm=cube_size,
    )
    stack.add_surface(Surface(name=f"{pbs_part} Entrance Face", thickness=half_size, glass=substrate, semi_diameter=half_size))
    s_hyp = Surface(name=f"{pbs_part} Polarizing Hypotenuse", thickness=half_size, glass=substrate, semi_diameter=half_size, tilt_x=45.0)
    s_hyp.is_beam_splitter = True
    s_hyp.beamsplitter_config = cfg_pbs
    stack.add_surface(s_hyp)
    stack.add_surface(Surface(name=f"{pbs_part} Exit Face", thickness=float(distance_pbs_to_qwp_mm), glass="AIR", semi_diameter=half_size))

    # S4: Quarter-Wave Plate (WPH10M at 45 deg)
    stack.add_surface(Surface(name=f"Quarter-Wave Plate (θ={qwp_angle_deg:g}°)", thickness=float(distance_qwp_to_target_mm), glass="QUARTZ", semi_diameter=12.7))

    # S5: Target Specular Mirror
    stack.add_surface(Surface(name="Target Specular Reflector", surface_type=SurfaceType.MIRROR, thickness=-float(distance_qwp_to_target_mm), glass="MIRROR", semi_diameter=12.7))

    # S6: Return QWP pass
    stack.add_surface(Surface(name="QWP Return Pass", thickness=-float(distance_pbs_to_qwp_mm), glass="QUARTZ", semi_diameter=12.7))

    # S7: PBS Return Hypotenuse (Reflecting S-pol to dump)
    s_ret_hyp = Surface(name=f"{pbs_part} Return Hypotenuse (S-Pol Fold)", surface_type=SurfaceType.MIRROR, thickness=half_size, glass=substrate, semi_diameter=half_size, tilt_x=45.0)
    stack.add_surface(s_ret_hyp)

    # S8: Dump / Photodiode Detector (BT610)
    stack.add_surface(Surface(name="Beam Dump / Power Monitor (BT610)", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=10.0))

    zmx_text = export_to_zmx_string(stack, title=stack.name, wavelengths=[float(wavelength_um)])

    return PBSOpticalIsolatorResult(
        name="Thorlabs PBS Optical Isolator",
        pbs_part=pbs_part,
        wavelength_um=float(wavelength_um),
        input_polarization="Linear P-Polarization",
        forward_transmitted_power=p_fwd,
        forward_jones=(complex(j_in[0]), complex(j_in[1])),
        target_reflected_power=p_target,
        return_polarization="Converted S-Polarization",
        return_jones=(complex(j_ret[0]), complex(j_ret[1])),
        dump_rejected_power=p_dump,
        laser_leakage_power=p_leak,
        optical_isolation_db=iso_db,
        system=stack,
        zmx_text=zmx_text,
        components_used=[pbs_part, "WPH10M-633", "PF10-03-P01", "BT610"],
    )


# =============================================================================
# 4. Beam Sampler / Power Pick-Off Designer (Zemax Approach)
# =============================================================================

def build_zemax_beam_sampler(
    bs_part: str = "BS019",
    wavelength_um: float = 0.6328,
    input_power_mw: float = 100.0,
) -> BeamSamplerResult:
    """
    Synthesizes a calibrated laser beam sampler pick-off following Zemax OpticStudio conventions:
    1. Laser Source with nominal power P0 (mW).
    2. Thorlabs Beam Sampler (e.g. BS019 90:10 cube or BSW32 90:10 plate).
    3. Transmitted main line: 90% power delivers to experiment target.
    4. Reflected monitor pick-off: 10% power deflected at 90 degrees to calibrated photodetector.
    5. Computes pick-off ratio and insertion loss in dB.
    """
    bs_info = get_catalog_lens(bs_part)
    cube_size = float(bs_info.get("dimensions_mm", {}).get("x", 25.4)) if isinstance(bs_info.get("dimensions_mm"), dict) else 25.4
    half_size = cube_size / 2.0
    substrate = str(bs_info.get("surfaces", [{}])[0].get("glass", "N-BK7"))

    r_ratio = float(bs_info.get("split_ratio_R", 0.10))
    t_ratio = float(bs_info.get("split_ratio_T", 0.90))

    p_in = float(input_power_mw)
    p_trans = p_in * t_ratio
    p_sample = p_in * r_ratio

    pick_pct = (r_ratio / (r_ratio + t_ratio)) * 100.0 if (r_ratio + t_ratio) > 0 else 0.0
    loss_db = -10.0 * math.log10(max(t_ratio, 1e-12))

    stack = SurfaceStack()
    stack.name = f"Thorlabs Beam Sampler Pick-Off ({bs_part})"
    stack.primary_wavelength = float(wavelength_um)
    stack.wavelengths = [float(wavelength_um)]

    # S0: Laser Source
    stack.add_surface(Surface(name="High-Power Laser Input", surface_type=SurfaceType.OBJECT, thickness=30.0, glass="AIR", semi_diameter=3.0))

    # S1-S3: Beam Sampler Cube
    cfg = BeamSplitterConfig(
        part_number=bs_part,
        splitter_type="Cube",
        is_polarizing=False,
        split_ratio_R=r_ratio,
        split_ratio_T=t_ratio,
        substrate_glass=substrate,
        cube_side_length_mm=cube_size,
        clear_aperture_mm=cube_size,
    )
    stack.add_surface(Surface(name=f"{bs_part} Entrance Face", thickness=half_size, glass=substrate, semi_diameter=half_size))
    s_hyp = Surface(name=f"{bs_part} 90:10 Hypotenuse", thickness=half_size, glass=substrate, semi_diameter=half_size, tilt_x=45.0)
    s_hyp.is_beam_splitter = True
    s_hyp.beamsplitter_config = cfg
    stack.add_surface(s_hyp)
    stack.add_surface(Surface(name=f"{bs_part} Main Exit Face (90% Power)", thickness=50.0, glass="AIR", semi_diameter=half_size))

    # S4: Target Delivery Plane
    stack.add_surface(Surface(name="Experimental Target Plane (90% Transmission)", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=12.7))

    zmx_text = export_to_zmx_string(stack, title=stack.name, wavelengths=[float(wavelength_um)])

    return BeamSamplerResult(
        name="Thorlabs Beam Sampler Pick-Off",
        bs_part=bs_part,
        wavelength_um=float(wavelength_um),
        input_power_mw=p_in,
        transmitted_power_mw=p_trans,
        sampled_power_mw=p_sample,
        pick_off_ratio_percent=pick_pct,
        insertion_loss_db=loss_db,
        system=stack,
        zmx_text=zmx_text,
        components_used=[bs_part],
    )


# =============================================================================
# 5. Generic Zemax Beamsplitter Export Helper
# =============================================================================

def export_beamsplitter_design_to_zmx(
    design_result: Union[MichelsonDesignResult, MachZehnderDesignResult, PBSOpticalIsolatorResult, BeamSamplerResult, SurfaceStack],
    output_filepath: Optional[str] = None,
) -> str:
    """
    Exports any beamsplitter design result or SurfaceStack to a Zemax OpticStudio (.zmx) prescription.
    Optionally writes to disk if output_filepath is provided.
    """
    if hasattr(design_result, "zmx_text"):
        zmx_content = design_result.zmx_text
    elif isinstance(design_result, SurfaceStack):
        zmx_content = export_to_zmx_string(design_result, title=design_result.name)
    else:
        raise TypeError(f"Unsupported design result type: {type(design_result)}")

    if output_filepath:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(zmx_content)

    return zmx_content
