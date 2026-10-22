# Project: Zemax-Standard Optical Optimization Engine

## Architecture
Production-grade optical optimization engine following Zemax industry standards across `optics_app/` and `v2/`:
- **Merit Function & Sampling Subsystem**: Multi-field ray tracing ($H_y \in [0.0, 0.707, 1.0]$) with area/quadrature weighting $[1.0, 2.0, 1.0]$, full-pupil hexapolar and Gauss-Legendre quadrature sampling, field-segregated merit evaluation, centroid and chief-ray referenced spot diagrams, and paraxial Rayleigh-Hamilton RMS wavefront OPD scaling ($W_{\text{RMS}} = R_{\text{RMS}} / (8 \cdot F/\# \cdot \lambda)$).
- **Boundary Operands & Parameter Controls**: Numerical barrier penalties and DLS residuals for `MNCT`, `MXCT` (1.5 <= CT <= 18.0 mm), `MNET`, `MXET` (ET >= 1.0 mm at clear semi-diameter $y_{sd}$ via exact surface sag), `MNEA` ($ET_{\text{air}} \ge 0.5$ mm), continuous `EFFL` ($\Phi = -C$ paraxial matrix power), and `TOTR` (measuring Surface 1 to image, strictly excluding object distance S0). Curvature space parameterization ($c = 1/R$) with dynamic bounds clamping based on clear semi-diameter ($|c| \le 0.95 / y_{sd}$).
- **Optimization Algorithms**: Native multi-surface Levenberg-Marquardt DLS engine with adaptive damping ($\lambda_{k+1} = \lambda_k \times \beta$), SVD/Cholesky ill-conditioning protection, and step backtracking; Cauchy/Gaussian simulated annealing Hammer optimizer and discrete sign-flip Global Search multi-start.
- **Application Delivery & Tree Synchronization**: `MainWindow.on_optimize` results delivery to `LensDataTableModel` and `SurfaceStack`, automatic refresh of LDE table, 2D/3D visualizers, multi-field ray traces, optimizer dock status, and `SystemManufacturingReport` audit tab. 100% byte-for-byte synchronization between `optics_app/` and `v2/` trees.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Multi-Field Sampling & Area Weighting | Standard $H_y \in [0.0, 0.707, 1.0]$ with area/quadrature weighting $[1.0, 2.0, 1.0]$; multi-field ray generation in optimizer; field-aware ray aiming. | M1 | ORIGINAL_REQUEST §R1 |
| F2 | Full-Pupil Hexapolar & Gaussian Quadrature Grid Sampling | Full pupil hexapolar sampling up to $\rho = 1.0$ (eliminating hardcoded 0.85 truncation); Gauss-Legendre quadrature pupil grid with radial roots and weights. | M1 | ORIGINAL_REQUEST §R1 |
| F3 | Optimization Targets (Centroid & Chief-Ray Spot, Paraxial RMS Wavefront OPD) | Chief-ray referenced RMS spot radius in `SpotDiagram`; field-segregated spot evaluation in `MeritFunction`; paraxial Rayleigh-Hamilton RMS wavefront OPD scaling ($W_{\text{RMS}} = R_{\text{RMS}} / (8 \cdot F/\# \cdot \lambda)$) in DLS residuals and merit function. | M1 | ORIGINAL_REQUEST §R1 |
| F4 | Zemax Boundary Operands | `MNCT`, `MXCT` (1.5 <= CT <= 18.0 mm); `MNET`, `MXET` (ET >= 1.0 mm at clear semi-diameter $y_{sd}$ via exact sag); `MNEA` ($ET_{\text{air}} \ge 0.5$ mm); `EFFL` ($\Phi = -C$ paraxial power); `TOTR` (Surface 1 to image, excluding S0). | M2 | ORIGINAL_REQUEST §R2 |
| F5 | Curvature Space Parameterization & Bounds Clamping | $c = 1/R$ optimization with bounds dynamically scaled to clear semi-diameter ($|c| \le 0.95 / y_{sd}$). | M2 | ORIGINAL_REQUEST §R2 |
| F6 | Damped Least Squares (DLS) Engine with Levenberg-Marquardt Adaptation | Native multi-surface Levenberg-Marquardt DLS engine with adaptive damping ($\lambda_{k+1} = \lambda_k \times \beta$, SVD/Cholesky conditioning, step backtracking). | M2 | ORIGINAL_REQUEST §R3 |
| F7 | Global Search & Hammer Optimization | Cauchy/Gaussian perturbations within manufacturing bounds, simulated annealing, bounded local refinement. | M2 | ORIGINAL_REQUEST §R3 |
| F8 | UI Result Delivery & Model Refresh | Update `LensDataTableModel` and `SurfaceStack`, refresh LDE table, 2D/3D visualizers, multi-field ray traces, and optimizer dock status; diagnostics vignetting fix. | M3 | ORIGINAL_REQUEST §R4 |
| F9 | 100% Tree Synchronization between optics_app/ and v2/ | Replace all stubs in `v2/`, synchronize all differing files, ensure identical behavior across trees. | M3 | ORIGINAL_REQUEST §R4 |
| F10 | Comprehensive E2E Verification & Manufacturability Validation | Verification of doublet and Cooke triplet convergence across 3 fields, 0 critical manufacturing violations, and 100% test pass rate across both trees. | M4 | ORIGINAL_REQUEST §Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | E2E Testing Track | Design & implement opaque-box E2E test harness and test cases (Tiers 1–4) covering F1–F10; publish TEST_READY.md. | none | DONE |
| M1 | Merit Function & Multi-Field/Pupil Sampling | Implement F1, F2, F3: multi-field ray tracing in optimizer, area weighting, full pupil hexapolar + Gaussian quadrature, chief-ray referenced spot radius, and Rayleigh-Hamilton OPD. | none | DONE |
| M2 | Boundary Operands, Parameter Controls & DLS Engine | Implement F4, F5, F6, F7: MNCT, MXCT, MNET, MXET, MNEA, EFFL, TOTR (S0 fix), dynamic curvature clamping, native multi-surface LM DLS engine with $\lambda_{k+1}=\lambda_k \beta$, and Hammer/Global integration. | M1 interface contracts | DONE |
| M3 | UI Delivery, Diagnostics & 100% Tree Synchronization | Implement F8, F9: LensDataTableModel/SurfaceStack live updates, diagnostics vignetting fix, 100% byte-for-byte synchronization of optics_app/ and v2/. | M1, M2 | DONE |
| M4 | Final Milestone: 100% E2E Test Pass & Tier 5 Adversarial Hardening | Phase 1: Pass 100% of E2E tests (Tiers 1-4) published in TEST_READY.md. Phase 2: Tier 5 adversarial stress testing, manufacturing audit verification. | M0, M1, M2, M3 | IN_PROGRESS |

## Interface Contracts
### Merit Function & Sampling (M1) ↔ Optimizer Engine (M2)
- `OpticalOptimizer._generate_rays(eval_stack)`:
  - Supports `self.fields: List[FieldPoint]`. If fields are provided, generates ray bundles for each field point ($H_y \in [0.0, 0.707, 1.0]$) with proper chief ray aiming and normalized entrance pupil coordinates.
  - Returns `traced_rays` with field tags or structured ray dictionary: `{field_idx: List[Ray]}`.
- `SpotDiagram.calculate_rms_radius(reference="centroid" | "chief_ray", chief_ray_coord=None) -> float`:
  - Returns RMS spot radius relative to centroid or chief-ray intercept.
- `OpticalOptimizer._compute_dls_residuals(x, var_dict, use_curv, bounds_param)`:
  - Formulates composite residuals partitioned by field, scaled by field weights $W_f \in [1.0, 2.0, 1.0]$ and pupil quadrature weights.
  - For Wavefront Error target, scales transverse residuals by paraxial factor $\frac{1}{8 (F/\#) \lambda}$.
- `MeritFunction.evaluate(traced_rays, surfaces, system_config=None) -> float`:
  - Partitions hits by `(surface_index, field_index, wavelength)` to compute true per-field RMS spots.

### Optimizer Engine (M2) ↔ UI & Models (M3)
- `OpticalOptimizer.optimize(...) -> OptimizationResult`:
  - Returns `OptimizationResult` containing `optimized_stack: SurfaceStack`, `initial_merit: float`, `final_merit: float`, `converged: bool`, `history: List[float]`.
- `LensDataTableModel.set_surface_stack(stack: SurfaceStack) -> None`:
  - Atomically resets table model and emits `system_updated`.
- `SystemManufacturingReport.audit_optical_system_manufacturing(surfaces, config=None) -> SystemManufacturingReport`:
  - Evaluates MNCG, MXCG, MNEG, MXEG, MNCA, MNEA, TOTR (excluding S0), aspect ratio, test plates, and RAID. Returns 0 critical violations for compliant systems.

## Code Layout
- `optics_app/engine/optimizer.py` & `v2/engine/optimizer.py`: Core optimization engine, DLS solver, Hammer optimizer, boundary residuals.
- `optics_app/engine/merit_function.py` & `v2/engine/merit_function.py`: Merit operands, multi-field evaluation.
- `optics_app/engine/system_config.py` & `v2/engine/system_config.py`: Field points, presets, area weights, pupil sampling patterns.
- `optics_app/engine/pupil_solve.py` & `v2/engine/pupil_solve.py`: Hexapolar and Gauss-Legendre quadrature pupil generation.
- `optics_app/analysis/spot.py` & `v2/analysis/spot.py`: Spot diagram calculations, centroid vs chief ray reference.
- `optics_app/engine/manufacturing_standards.py` & `v2/engine/manufacturing_standards.py`: ISO 10110 manufacturing limits and audit calculations.
- `optics_app/gui/main_window.py` & `v2/gui/main_window.py`: Optimization request handling, model update and ray tracing dispatch.
- `optics_app/gui/optimizer_dock.py` & `v2/gui/optimizer_dock.py`: Optimizer dock UI, method selection, variable checkboxes.
- `tests/test_e2e_zemax_optimizer.py`: E2E test suite (Tiers 1–4).
