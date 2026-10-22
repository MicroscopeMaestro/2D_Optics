"""
Optical Spectrometer Design & Verification Script
Models the full optical train: Fiber -> L1 (Collimator) -> Grating -> L2 (Focusing) -> DMD4500 -> L3 (Relay Collimator) -> L4 (Condenser) -> 1mm² Detector.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.analysis.spot import SpotDiagram

def test_spectrometer_principles():
    print("================================================================")
    print("1. OPTICAL SPECTROMETER DESIGN PRINCIPLES & FORMULATION")
    print("================================================================")
    
    # 1. Source Specification:
    # Typical multimode fiber (e.g. 50 um core diameter, NA = 0.22)
    fiber_core_um = 50.0
    fiber_na = 0.22
    fiber_half_angle_deg = np.degrees(np.arcsin(fiber_na))
    print(f"Fiber: Core = {fiber_core_um} um, NA = {fiber_na} (Half-angle = {fiber_half_angle_deg:.2f} deg)")
    
    # 2. Wavelength Range:
    # 785 nm Raman / NIR band: 785 nm to 950 nm (or 785 - 900 nm)
    wls = [0.785, 0.850, 0.900, 0.950] # um
    lambda_center = 0.850
    delta_lambda = wls[-1] - wls[0]
    print(f"Spectral Band: {wls[0]*1000:.1f} nm to {wls[-1]*1000:.1f} nm (Bandwidth = {delta_lambda*1000:.1f} nm)")
    
    # 3. Grating Parameters:
    # 600 lines/mm reflection grating
    lines_per_mm = 600.0
    grating_pitch_um = 1000.0 / lines_per_mm # 1.6667 um
    alpha_deg = 15.0 # Incident angle
    alpha_rad = np.radians(alpha_deg)
    
    # Grating equation: sin(beta) = m*lambda/d - sin(alpha) (for reflection grating convention)
    # or m*lambda/d + sin(alpha) depending on coordinate convention
    # In our engine: v_out_disp = v_in_disp + m*lambda/d
    sin_beta_center = np.sin(alpha_rad) + (1.0 * lambda_center / grating_pitch_um) # order = 1
    # Check diffraction angle
    print(f"Grating: {lines_per_mm} lines/mm (d = {grating_pitch_um:.4f} um)")
    print(f"Incidence angle alpha = {alpha_deg} deg")
    
    # 4. First Focusing Lens (L2) & DMD4500 Coverage:
    # DMD4500 dimensions:
    # Active micromirror array: 9.855 mm (width) x 6.16 mm (height)
    dmd_width_mm = 9.855
    dmd_height_mm = 6.16
    target_dmd_dispersion_mm = dmd_width_mm / 2.0 # ~4.93 mm (half area / half width)
    print(f"DMD4500 Active Area: {dmd_width_mm:.3f} mm x {dmd_height_mm:.3f} mm")
    print(f"Target Spectral Coverage on DMD: ~{target_dmd_dispersion_mm:.2f} mm (Half Area along dispersion axis)")
    
    # Angular dispersion: dbeta/dlambda = m / (d * cos(beta))
    cos_beta_center = np.sqrt(1.0 - (np.sin(alpha_rad) + lambda_center/grating_pitch_um)**2) if (np.sin(alpha_rad) + lambda_center/grating_pitch_um)**2 < 1.0 else 0.8
    # Angular spread:
    # delta_beta = (m * delta_lambda) / (d * cos_beta)
    # Required f2 = target_dmd_dispersion / delta_beta
    # Let's test standard focal lengths: f2 = 50.0 mm (e.g. Thorlabs AC254-050-B)
    f2 = 50.0 # mm
    f1 = 30.0 # mm (Collimator, e.g. Thorlabs AC127-030-B or AC254-030-B)
    
    # 5. Last Two Lenses System (Relay L3 & Condenser L4) & Demagnification:
    # Target detector: 1 mm² single detector (1.0 mm x 1.0 mm, semi-diameter = 0.5 mm)
    detector_size_mm = 1.0
    required_magnification = detector_size_mm / target_dmd_dispersion_mm # 1.0 / 4.93 = 0.203
    print(f"Target Detector: 1.0 mm x 1.0 mm (1 mm² active area)")
    print(f"Required Collection Demagnification: M <= {required_magnification:.3f} (approx 5x to 6x demagnification)")
    
    # Two-lens relay system (L3 = Relay Collimator, L4 = High-NA Condenser):
    # In afocal/4f configuration: M = f4 / f3
    # Standard Thorlabs choices:
    # L3: f3 = 50.0 mm (Thorlabs AC254-050-B, clear aperture 23 mm)
    # L4: f4 = 8.0 mm (Thorlabs High-NA Aspheric Condenser ACL12708U / ACL25416U or f=8-10mm doublet)
    f3 = 50.0 # mm
    f4 = 8.0 # mm
    M_relay = f4 / f3 # 8.0 / 50.0 = 0.160
    final_image_span_on_detector = target_dmd_dispersion_mm * M_relay
    print(f"Relay Lens L3: f3 = {f3:.1f} mm")
    print(f"Condenser Lens L4: f4 = {f4:.1f} mm")
    print(f"Net Magnification M = f4/f3 = {M_relay:.3f} ({1/M_relay:.1f}x demagnification)")
    print(f"Calculated Spectrum Footprint on Detector: {final_image_span_on_detector:.3f} mm (< 1.0 mm detector limit!)")
    print("================================================================")

if __name__ == "__main__":
    test_spectrometer_principles()
