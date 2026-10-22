# E2E Test Infra: Zemax-Standard Optical Optimization Engine

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on implementation design.
- Methodology: Category-Partition + BVA + Pairwise + Workload Testing.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| 1 | Multi-Field Sampling & Area Weighting | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| 2 | Full-Pupil Hexapolar & Gauss Quadrature | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| 3 | Optimization Targets (Spot & Wavefront OPD) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| 4 | Boundary Operands (MNCT, MXCT, MNET, MXET, MNEA, EFFL, TOTR) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| 5 | Curvature Space Parameterization & Clamping | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| 6 | DLS with Levenberg-Marquardt Adaptation | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ |
| 7 | Global Search & Hammer Optimization | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ |
| 8 | UI Result Delivery & Model Refresh | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ |
| 9 | 100% Tree Synchronization | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ |
| 10 | Real-World Convergence & Manufacturability | Acceptance Criteria | 5 | 5 | ✓ |

## Test Architecture
- Test runner: `pytest tests/test_e2e_zemax_optimizer.py`
- Location: `tests/test_e2e_zemax_optimizer.py`
- Pass/Fail semantics: All tests pass with exit code 0

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Achromatic Doublet (F/4, 100mm EFL) Multi-Field Optimization | F1, F2, F3, F4, F6, F10 | High |
| 2 | Cooke Triplet (F/4.5, 50mm EFL) 3-Field DLS Optimization | F1, F2, F3, F4, F5, F6, F10 | High |
| 3 | Deep Meniscus Air Gap Collision Avoidance (MNEA) | F4, F6, F10 | Medium |
| 4 | Hammer Global Optimization Local Minima Escape | F4, F5, F7, F10 | High |
| 5 | Full Headless UI Optimization & Model Sync Trace | F1, F8, F9, F10 | High |

## Coverage Thresholds
- Tier 1: >= 5 per feature
- Tier 2: >= 5 per feature (boundary and corner cases)
- Tier 3: Pairwise coverage of major feature interactions
- Tier 4: >= 5 realistic application scenarios
- Total target: >= 120 tests
