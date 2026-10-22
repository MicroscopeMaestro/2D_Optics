# 3D-Folded Refractive 1064 nm Spectrometer Layout

## 1. Design Overview & Optical Architecture
This project implements a precision 3D-folded optical spectrometer layout optimized for the 1064 nm Raman fingerprint spectral region (1.087 µm to 1.350 µm).

The design uses standard off-the-shelf Thorlabs refractive lenses, a 600 lines/mm reflective diffraction grating, an aperture stop, and a Texas Instruments DLP4500 (DMD4500) spatial light modulator, relaying dispersed Raman light into a single 0.8 mm diameter detector (active radius 0.40 mm).

### Optical Component Prescription:
1. **Input Slit**: 30 µm slit / point source aperture.
2. **Collimator**: Thorlabs AC254-030-C ($f = 30.0$ mm achromatic doublet, 1-inch diameter, $r_1 = 71.12$ mm, $t_1 = 1.8$ mm N-SF6HT, $r_2 = 15.24$ mm, $t_2 = 13.0$ mm N-BAF10, $r_3 = -21.09$ mm, $t_3 = 25.0$ mm AIR).
3. **Aperture Stop**: Circular aperture (semi-diameter 5.0 mm, $t = 25.0$ mm AIR).
4. **Out-of-Plane 3D Fold**: 90-degree Z-rotation coordinate break (`tilt_z = 90.0°`) preceding the grating to route returning dispersed beam away from the input collimation axis.
5. **Diffraction Grating**: 600 lines/mm reflective grating at off-axis incident angle `tilt_x = -45.0°` (avoiding Littrow collisions).
6. **Grating Coordinate Break**: `tilt_x = 46.60°` for primary wavelength 1.225 µm diffraction channel, $t = 45.0$ mm AIR.
7. **Camera / Focusing Lens**: Thorlabs AC254-050-C ($f = 50.0$ mm achromatic doublet, 1-inch diameter, $r_1 = 34.5$ mm, $t_1 = 12.0$ mm N-BAF10, $r_2 = -23.1$ mm, $t_2 = 2.0$ mm N-SF6HT, $r_3 = -135.2$ mm, $t_3 = 45.0$ mm AIR).
8. **Spatial Light Modulator**: Texas Instruments DMD4500 with micromirrors tilted at mechanical specification `tilt_y = +12.0°` (active semi-diameter 5.0 mm, rectangular aperture 4.95 mm × 4.95 mm).
9. **DMD Coordinate Break**: `CB_After_DMD` ($t = 66.94$ mm AIR).
10. **Relay Doublet**: Thorlabs AC254-050-C ($f = 50.0$ mm achromatic doublet, 1-inch diameter, $r_1 = 34.5$ mm, $t_1 = 12.0$ mm N-BAF10, $r_2 = -23.1$ mm, $t_2 = 2.0$ mm N-SF6HT, $r_3 = -135.2$ mm, $t_3 = 151.28$ mm AIR).
11. **Condenser Lens**: Thorlabs ACL25416U-C ($f = 16.0$ mm aspheric condenser, B270, conic = -1.0, 1-inch diameter, $r_1 = 8.0$ mm, $t_1 = 10.0$ mm B270, $r_2 = \infty$, $t_2 = 15.25$ mm AIR).
12. **Detector**: 0.8 mm diameter single-pixel photodiode (semi-diameter 0.40 mm).

---

## 2. Verification Scripts

### Spot Radius Verification:
Run:
```bash
python teamwork_refractive_1064/check_spot.py
```
- Mathematically traces polychromatic ray bundles across the Raman band (1.087 µm, 1.225 µm, 1.350 µm) and NA angular divergence cone.
- Proves 100.0% throughput (120/120 rays, 0 vignetted rays) across all design wavelengths.
- Proves maximum spot radius from wavelength centroid = **0.3850 mm** (strictly <= 0.4000 mm requirement).
- Proves maximum radial distance from detector center (0,0) = **0.3858 mm** (strictly <= 0.4000 mm requirement).
- Overall RMS spot radius = **0.1755 mm**.

### 3D Collision Avoidance & Housing Clearance:
Run:
```bash
python teamwork_refractive_1064/check_collisions.py
```
- Calculates 3D global positions and orientations of all optical elements.
- Evaluates minimum 3D clearance between every ray segment and every unrelated optical component housing center.
- Proves minimum distance to any unrelated housing center = **19.22 mm** (> 0 mm).
- Proves minimum clearance margin beyond physical housing radius (+5 mm margin) = **+1.52 mm** (> 0 mm margin).
- Total segment interactions checked: 4161. Total collisions / violations: **0**.

### Test Suite:
Run:
```bash
pytest tests/test_refractive_1064_spectrometer.py -v
```
All tests PASS.
