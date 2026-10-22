# Thorlabs 785nm Raman Fiber Spectrometer

## Complete 3D Folded Architecture with Thorlabs Lenses

We have completed the optimization of the complete, physically folded 3D fiber spectrometer using realistic Thorlabs `AC254-040-A` achromatic doublets, specifically tuned for a **785 nm Raman laser** source (simulating wavelengths of 785 nm, 850 nm, and 900 nm).

The optical engine properly incorporates the divergence of the multimode fiber (NA=0.22) and folds the 3D coordinate system around the diffraction grating and the DMD mirror correctly.

### Optimized Mechanical Spacings (Thicknesses)

The Nelder-Mead optimization algorithm successfully converged, significantly improving the focused spot RMS radius on the detector from **1.14 mm** down to **0.19 mm** by adjusting the free-space propagation distances. 

Here are the final, optimized air-gap distances between your components:

- **Fiber to Collimating Lens (L1):** 30.17 mm
- **L1 to Grating:** 20.63 mm
- **Grating to Focusing Lens (L2):** 19.77 mm
- **L2 to DMD Mirror:** 27.24 mm
- **DMD Mirror to Relay Lens (L3):** 31.36 mm
- **L3 to Condenser Lens:** 25.71 mm
- **Condenser to CCD Detector:** 16.49 mm

*(Note: These distances are the air gaps separating the back vertex of the previous element to the front vertex of the next element).*

### Optical Configuration Files

The full prescription, including all radii of curvature for the Thorlabs doublets, glass materials, folds (tilts), and the optimized mechanical spacings, has been saved to:

[`fiber_spectrometer_thorlabs_optimized.json`](file:///C:/Users/Juan/Documents/GitHub/2D_Optics/fiber_spectrometer_thorlabs_optimized.json)

### Next Steps & Verification

You can now open this JSON file directly in your `2D_Optics` GUI application to visually inspect the 3D optical layout and the resulting spot diagrams on the detector tab. The system now realistically models:
1. **Diverging rays** emerging from the fiber.
2. **True 3D Spatial Diffraction** taking place at the Grating (13° incidence).
3. **True 3D Specular Reflection** taking place at the DMD (15° incidence).
4. **Chromatic Focusing** across the Thorlabs glass components for the infrared spectrum.

Let me know if you would like me to adjust the entrance angles or constrain any of these physical distances further!
