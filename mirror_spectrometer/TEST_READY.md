# TEST_READY: All-Mirror Spectrometer Layout E2E Test Suite

## Overview
A comprehensive, 4-tier End-to-End (E2E) verification test suite has been established for the **All-Mirror DMD4500 Spectrometer Layout** project. The test suite strictly validates all physical, optical, mechanical, and catalog constraints mandated by `ORIGINAL_REQUEST.md`, `PROJECT.md`, and `TEST_INFRA.md`.

---

## Test Inventory & Tier Architecture

The test harness consists of **20 discrete test methods** organized into 4 hierarchical verification tiers in `mirror_spectrometer/tests/test_all_mirror_spectrometer.py`:

| Tier | Focus Area | Test Count | Key Invariants Verified |
|---|---|---|---|
| **Tier 1** | Feature Coverage & COTS Compliance | 7 Tests | Schema validation; all-mirror architecture (strictly 0 refractive lenses after entrance slit); Thorlabs COTS catalog spherical concave mirror radii compliance ($R \in \{-19, -24, -30, -38, -50, -100, -150, -200, -300, -400, -500, -600\}$ mm, conic $= 0.0$); 600 l/mm reflective grating in order $+1$; aperture stop before M1 with $r \le 6.0$ mm; DMD active dispersion span $\le 5.0$ mm; single detector element with active diameter $\le 0.80$ mm ($r \le 0.40$ mm). |
| **Tier 2** | Boundary & Corner Cases | 5 Tests | Band edge extreme NA ray tracing at $\lambda = 785$ nm and $\lambda = 880$ nm; nominal primary wavelength ($\lambda = 830$ nm) chief ray alignment ($r_{\text{hit}} < 0.20$ mm); slit spatial corners ($x = \pm 0.015$ mm, $y = \pm 0.25$ mm); zero intermediate vignetting across all boundary rays. |
| **Tier 3** | Cross-Feature & Geometric Clearances | 4 Tests | Slit plane ($Z=0$) beam clearance $> 12.7$ mm; collimator plane ($Z=50$) M1-to-DMD clearance $> 12.7$ mm; M3/M4 relay path to detector separation $> 12.7$ mm; 3D solid mirror substrate non-interference. |
| **Tier 4** | Real-World Application Workload | 4 Tests | Dense polychromatic ray tracing bundle ($765$ rays across all 5 operating wavelengths, 9 slit spatial positions, and multi-ring polar pupil sampling); **mathematical proof of 100% ray containment within $r \le 0.4000$ mm**; RMS spot radius $R_{\text{RMS}} < 0.250$ mm; $100.0\%$ encircled energy within $r = 0.4000$ mm. |

---

## Execution Commands

### 1. Standalone Python Test Runner (Recommended)
```bash
# Test against the default or discovered layout
python mirror_spectrometer/tests/run_tests.py

# Test specifically against the reference fixture
python mirror_spectrometer/tests/run_tests.py --fixture

# Test a custom layout file
python mirror_spectrometer/tests/run_tests.py path/to/spectrometer_layout.json
```

### 2. Pytest Execution
```bash
# Standard pytest execution
pytest -v mirror_spectrometer/tests/test_all_mirror_spectrometer.py

# Pytest with explicit layout environment override
python -c "import os, subprocess, sys; env=os.environ.copy(); env['ALL_MIRROR_LAYOUT_PATH']=r'mirror_spectrometer/tests/fixtures/reference_all_mirror_spectrometer.json'; subprocess.run(['pytest', '-v', 'mirror_spectrometer/tests/test_all_mirror_spectrometer.py'], env=env)"
```

---

## Pass / Fail Semantics

- **Strict Acceptance Criteria**:
  1. `test_t1_2`: Fails if ANY refractive glass element (e.g. N-BK7, F_SILICA, SF11) is present after surface 0 (entrance slit).
  2. `test_t1_3`: Fails if any mirror radius of curvature does not match Thorlabs off-the-shelf catalog spherical concave mirrors or if conic $\neq 0$.
  3. `test_t1_6`: Fails if spectral dispersion width on DMD exceeds $5.05$ mm across the $785 - 880$ nm band.
  4. `test_t3_1` / `test_t3_2`: Fails if physical clearance between optical paths and housing planes is $\le 12.7$ mm (standard $0.5''$ clearance).
  5. `test_t4_2`: Fails if **even a single ray** at the detector plane has $\sqrt{x^2 + y^2} > 0.4000$ mm (100% containment requirement).
  6. `test_t4_3`: Fails if detector RMS spot radius $\ge 0.250$ mm.

- **Current Reference Fixture Baseline**:
  - **20 / 20 Tests Passed (100%)** in 5.14s.
  - Maximum spot radius: $0.3984$ mm $\le 0.4000$ mm.
  - RMS spot radius: $0.2152$ mm $< 0.2500$ mm.
  - Encircled energy at $r = 0.400$ mm: $100.0\%$.
