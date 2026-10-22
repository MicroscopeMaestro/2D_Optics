# 2D Optics Application — AI Continuation & Developer Architecture Guide

This document provides a comprehensive handover manual for any future AI assistant or human optical engineer continuing development on the **2D Optics** application.

---

## 1. Repository Layout & Architecture

The project has an evolved codebase structure with parallel branches designed to support modular development and standalone desktop compilation:

```
2D_Optics/
├── optics_app/                 # Primary development package
│   ├── engine/                 # Core physics, ray tracing, matrix optics, tolerancing
│   │   ├── surface.py          # Surface data model (standard, aspheric, mirror, grating, CB)
│   │   ├── lde.py              # SurfaceStack & Lens Data Editor management
│   │   ├── tracer.py           # 3D vector sequential ray tracer & polarization engine
│   │   ├── matrix_optics.py    # 2x2 ABCD transfer matrices & cardinal points
│   │   ├── matrix_tracer.py    # Multi-surface matrix propagation & Q-factor engine
│   │   ├── tolerancing.py      # Zemax-Style Tolerancing & Monte Carlo Simulation Engine
│   │   ├── extended_merit.py   # TRAR, RSCE, SPOT, MNET operands & Glatzel adaptive DLS
│   │   ├── ai_sequences.py     # 6-stage physics-informed automated lens design pipeline
│   │   └── thick_lens.py       # ThickLens & ThinLens paraxial optical elements
│   ├── gui/                    # Qt graphical user interface
│   │   ├── main_window.py      # Master application window & dock coordinator
│   │   ├── lde_table.py        # Lens Data Editor table model (2-way live binding)
│   │   ├── tolerancing_dock.py # Zemax Tolerance Analysis Dock Widget
│   │   ├── optimizer_dock.py   # Numerical optimization dock widget
│   │   └── qt_compat.py        # Qt5/Qt6/PySide6 compatibility shim
│   └── ai/                     # AI assistant & tool calling system
│       ├── tools.py            # Central ToolRegistry & tool implementations
│       ├── nemotron_client.py  # LLM client & natural language heuristic parser
│       └── optical_design_guide.md # Classical & modern lens design handbook
├── v2/                         # Desktop release branch (targeted by PyInstaller)
│   ├── engine/                 # Mirror of optics_app/engine/
│   ├── gui/                    # Mirror of optics_app/gui/
│   └── ai/                     # Mirror of optics_app/ai/
├── tests/                      # Automated test suites (pytest)
│   ├── test_zemax_tolerancing.py     # Zemax tolerancing unit & integration tests
│   ├── test_physical_lens_design.py  # Matrix optics, thick lens, optimization tests
│   ├── test_optical_ai_sequences.py  # AI sequence pipeline tests
│   └── test_e2e_ai_sequences.py      # End-to-end multi-stage design tests
├── launch_v2.py                # Python entrypoint for launching V2 GUI application
├── 2DOptics_V2.spec            # PyInstaller build specification
└── research_report.md          # Classical & modern optical design literature synthesis
```

> **IMPORTANT RULE FOR CODE EDITS:**  
> When modifying files in `optics_app/` (e.g. `optics_app/engine/tolerancing.py`, `optics_app/gui/tolerancing_dock.py`, `optics_app/ai/tools.py`), **always synchronize the changes into `v2/` and `v2/optics_app/`**! PyInstaller compiles directly from `launch_v2.py` which imports `v2`.

---

## 2. Modeling Packages & Infrastructure

The application leverages specialized Python scientific and CAD libraries:

| Package | Version | Primary Responsibility in 2D Optics |
| :--- | :--- | :--- |
| **`numpy`** | `2.4.4` | Vector ray tracing, Snell's law refraction, coordinate rotation matrices, sag arrays, and ABCD matrix math. |
| **`scipy`** | `1.15+` | Levenberg-Marquardt (DLS) optimization, 1D bounded compensator refocusing (`scipy.optimize.minimize_scalar`), root-finding for aspheres. |
| **`build123d`** | `0.8+` | Parametric Boundary Representation (B-Rep) solid CAD modeling of lenses, mirrors, and optomechanical mounts. |
| **`OCP`** | Installed | OpenCASCADE Python bindings providing the STEP solid geometry export kernel. |
| **`trimesh`** | `5.1+` | 3D mesh slicing, collision/clearance checking, STL/OBJ generation. |
| **`ezdxf`** | `1.4+` | 2D mechanical CAD drafting, cross-section layout drawings, and dual-plane DXF export. |
| **`openai`** | `1.63+` | AI client communication with OpenAI-compatible LLM endpoints (Nemotron, Gemma, etc.). |
| **`matplotlib`** | `3.10+` | Spot diagrams, transverse ray aberration fans, wavefront error, and 2D/3D ray propagation plots. |
| **`PySide6`** | `6.8+` | Modern Qt6 GUI interface, custom QGraphicsScene ray visualizer, and dock widgets. |

---

## 3. Optical Engine & Ray Tracing

### Sequential Ray Tracing (`optics_app/engine/tracer.py`)
- **Coordinate Systems:** Each surface maintains its own local coordinate system (`origin`, `R_mat`). Global accumulated positions are computed via forward translation and Euler rotation matrices ($R_z(\theta_z) R_y(\theta_y) R_x(\theta_x)$).
- **Exact Snell Refraction:** Computed in 3D vector form:
  $$\mathbf{s}' = \frac{n}{n'} \mathbf{s} + \left( \frac{n}{n'} \cos I - \sqrt{1 - \left(\frac{n}{n'}\right)^2 (1 - \cos^2 I)} \right) \mathbf{N}$$
- **Aspheric Surfaces (`optics_app/engine/aspheric.py`):** Standard ISO 10110-12 representation with conic constant $k$ and even polynomial deformation coefficients $\alpha_4, \alpha_6, \alpha_8$:
  $$z(r) = \frac{c r^2}{1 + \sqrt{1 - (1+k) c^2 r^2}} + \sum_{i=2}^N \alpha_{2i} r^{2i}$$
- **Coordinate Breaks:** Virtual surfaces that tilt and decenter the downstream optical axis without refracting light.

### Paraxial Matrix Optics (`matrix_optics.py` & `matrix_tracer.py`)
- Propagates $2 \times 2$ transfer matrices $\mathbf{R}$ (refraction) and $\mathbf{T}$ (translation).
- Cardinal points are extracted from the system matrix $\begin{bmatrix} A & B \\ C & D \end{bmatrix}$:
  $$EFL = -\frac{1}{C}, \quad BFL = -\frac{A}{C}, \quad FFL = \frac{D}{C}$$

---

## 4. Zemax-Style Tolerancing Simulation Suite

The tolerancing engine is implemented in `optics_app/engine/tolerancing.py` (and mirrored in `v2/engine/tolerancing.py`). It supports both sequential `SurfaceStack` systems (with 3D ray tracing) and paraxial `OpticalElement` systems.

### A. Supported Zemax Tolerance Operands (TDE)

| Operand | Description | Parameter Affected |
| :--- | :--- | :--- |
| **`TRAD`** | Tolerance on Radius of Curvature | Surface $R \pm \Delta R$ (mm or %) |
| **`TCUR`** | Tolerance on Curvature | Surface curvature $c \pm \Delta c$ ($1/\text{mm}$) |
| **`TTHI`** | Tolerance on Thickness / Airspace | Surface thickness / spacing $t \pm \Delta t$ (mm) |
| **`TSDX` / `TSDY`** | Tolerance on Surface Decenter X / Y | Individual surface lateral shift $\Delta x, \Delta y$ (mm) |
| **`TSTX` / `TSTY`** | Tolerance on Surface Tilt X / Y | Individual surface angular tilt $\Delta \theta_x, \Delta \theta_y$ (deg) |
| **`TEDX` / `TEDY`** | Tolerance on Element Decenter X / Y | Rigid body lateral shift of all element surfaces |
| **`TETX` / `TETY`** | Tolerance on Element Tilt X / Y | Rigid body angular tilt of all element surfaces |
| **`TIND`** | Tolerance on Refractive Index | Glass refractive index $n \pm \Delta n$ |
| **`TABB`** | Tolerance on Abbe Number | Dispersion $V_d \pm \Delta V_d$ (%) |
| **`TCON`** | Tolerance on Conic Constant | Conic constant $k \pm \Delta k$ |
| **`TIRR`** | Tolerance on Surface Irregularity | Fringes/waves of irregular sag distortion |
| **`COMP`** | Tolerance Compensator | Designates surface (e.g. BFL) for auto-refocusing |

### B. Probability Distributions
- **`uniform`**: Equal probability density over $[-\Delta, +\Delta]$.
- **`gaussian` / `normal`**: Zero-mean Gaussian where $\Delta$ represents the standard $3\sigma$ fabrication limit ($\sigma = \Delta / 3.0$), truncated at $[-\Delta, +\Delta]$.
- **`parabolic`**: Standard Zemax OpticStudio parabolic density $P(u) = \frac{3}{4\Delta}(1 - u^2)$ sampled via rejection sampling.

### C. Performance Criteria
- **`RMS_SPOT_RADIUS` (TRAC / RSCE):** Centroid RMS spot radius computed by tracing a 2D polar pupil ray grid (rings and spokes) through the system to the detector.
- **`GEO_SPOT_RADIUS` (GEOR):** 100% geometric encircled spot circle radius.
- **`EFFL`:** Effective focal length deviation from nominal.
- **`BFL`:** Back focal length deviation.
- **`MERIT_FUNCTION`:** User-defined `MeritFunction` or `PhysicalMeritFunction`.

### D. Compensator Optimization (Auto-Refocusing)
In real optical manufacturing, the technician adjusts the focus ring (detector position) during assembly. The function `solve_system_compensator()` runs a 1D bounded optimization (`minimize_scalar`) on the designated compensator surface (usually the last airspace before the detector) to find the minimum spot size for every perturbation step and Monte Carlo trial.

### E. Sensitivity Analysis (Worst-Case & RSS)
For each tolerance operand, the engine evaluates $+\Delta$ and $-\Delta$ with compensator refocusing:
- **Estimated Worst-Case Change:** $\Delta C_{WC} = \sum_k \max(|\Delta C_{k+}|, |\Delta C_{k-}|)$
- **Estimated Root-Sum-Square (RSS) Change:** $\Delta C_{RSS} = \sqrt{\sum_k \max(\Delta C_{k+}, \Delta C_{k-})^2}$
- Ranks worst offenders in an ASCII table matching Zemax OpticStudio.

### F. Monte Carlo Simulation & Manufacturing Yield
- Runs $N$ trials (10 to 50,000) sampling all operands simultaneously from their respective distributions.
- Adjusts compensators per trial.
- Computes sample statistics: Mean, Standard Deviation, Min, Max.
- Cumulative percentiles: **50% (median), 80%, 90% (confidence), 95%, 98%, 99% (3-sigma)**.
- Method `yield_percentage(threshold)` calculates the percentage of manufactured systems that meet the performance specification (e.g. RMS spot radius $\le 0.020\,\text{mm}$).

### G. Standard Quality Presets
- `create_commercial_tolerances(system)`: Thorlabs / Edmund standard optics ($\Delta R = 0.2\%$, $\Delta t = 0.05\,\text{mm}$, Decenter $= 0.025\,\text{mm}$, Tilt $= 1\,\text{mrad}$, Index $= 0.001$).
- `create_precision_tolerances(system)`: Research grade ($\Delta R = 0.1\%$, $\Delta t = 0.02\,\text{mm}$, Decenter $= 0.010\,\text{mm}$, Tilt $= 0.5\,\text{mrad}$, Index $= 0.0005$).
- `create_high_precision_tolerances(system)`: Laser / lithography grade ($\Delta R = 0.05\%$, $\Delta t = 0.01\,\text{mm}$, Decenter $= 0.005\,\text{mm}$, Tilt $= 0.2\,\text{mrad}$, Index $= 0.0002$).

---

## 5. GUI Integration

### Tolerancing Dock (`optics_app/gui/tolerancing_dock.py` & `v2/gui/tolerancing_dock.py`)
- **Dock Widget:** `TolerancingDockWidget` tabified next to the Optimizer Dock on the left dock area.
- **Controls:**
  1. Quality Preset selector with "Auto-Populate System" button.
  2. Interactive Tolerance Data Editor (TDE) table supporting adding/editing operands.
  3. Performance Criterion dropdown (RMS Spot, Geo Spot, EFFL, BFL, Merit).
  4. Auto-Refocus (Compensator) checkbox and adjustment range spinbox.
  5. Monte Carlo Trials and Random Seed inputs.
  6. "Run Sensitivity Analysis (RSS)" and "Run Monte Carlo Simulation" buttons executing on a background `TolerancingWorker` (`QThread`) to keep the UI responsive.
  7. Interactive Yield Rate calculator: enter a threshold (e.g. `0.250 mm`) and click "Calculate Yield %".
  8. "Export Report (.txt)" (Zemax OpticStudio ASCII report) and "Export to .ZMX" buttons.

### Menu & Toolbar
- In `MainWindow`, under the `Window` menu: `"📐 Zemax Tolerance Analysis"` toggles the dock.

---

## 6. AI Assistant & Tool Registry Integration

The application includes an autonomous AI assistant powered by OpenAI-compatible LLMs (Nemotron / Gemma).

### Registered AI Tools (`tools.py`)
1. **`run_zemax_tolerance_analysis`**:
   - Executes Sensitivity Analysis and/or Monte Carlo Simulation directly on the active system.
   - Accepts parameters: `preset`, `criterion`, `compensator_refocus`, `num_trials`, `run_monte_carlo`, `run_sensitivity`.
   - Returns a structured dictionary with worst offenders, RSS/Worst-case predictions, percentiles, yield, and formatted text.
2. **`export_zemax_tolerance_report`**:
   - Generates and writes a standard Zemax OpticStudio ASCII report to a file.
3. **`run_optical_design_sequence`**:
   - Executes the 6-stage automated design pipeline for synthesizing lenses from scratch.
4. **`run_lens_bending_optimization`**:
   - Analyzes Coddington shape factor bending to eliminate spherical aberration.
5. **`run_aspheric_optimization`**:
   - Progressively aspherizes surfaces using even polynomials.

### Heuristic NLP Routing (`nemotron_client.py`)
Natural language queries in the AI chat dock are parsed by `parse_optical_command_heuristics()`:
- *"run zemax tolerance analysis on the current lens"*
- *"run tolerance analysis with 1000 trials precision"*
- *"check worst offenders for manufacturing tolerances"*
- *"simulate manufacturing yield"*
- *"export zemax tolerance report"*

These automatically map to `run_zemax_tolerance_analysis` or `export_zemax_tolerance_report`, execute against the active `SurfaceStack`, and return results to the user.

---

## 7. Testing & Verification Guide

Always run tests before shipping changes:

```bash
# 1. Run all tolerancing unit and integration tests
pytest tests/test_zemax_tolerancing.py -v

# 2. Run thick lens and matrix optics benchmarks
pytest tests/test_physical_lens_design.py -v

# 3. Run AI optical design sequence tests
pytest tests/test_optical_ai_sequences.py tests/test_e2e_ai_sequences.py -v

# 4. Run entire project test suite
pytest -k "toleranc or sequence or lens"
```

### Rebuilding Desktop Executable
When changes are ready for release, recompile `2DOptics_V2.exe`:

```powershell
# Kill any running process
taskkill /F /IM 2DOptics_V2.exe /T

# Rebuild with PyInstaller
pyinstaller -y 2DOptics_V2.spec
```
The compiled output is located in `dist/2DOptics_V2/2DOptics_V2.exe`.

---

## 8. Summary of Completed Milestone Capabilities

1. **Full Zemax OpticStudio Tolerancing Engine** (`optics_app/engine/tolerancing.py`, `v2/engine/tolerancing.py`).
2. **Real 3D Ray Tracing on `SurfaceStack`** with centroid RMS spot size (TRAC) and geometric spot radius (GEOR).
3. **Auto-Refocus Compensator Optimization** simulating real lens barrel focusing.
4. **Univariate Sensitivity Analysis** computing Root-Sum-Square (RSS) and Worst-Case predicted degradations.
5. **Multivariate Monte Carlo Simulation** supporting Uniform, Gaussian (3-sigma), and Parabolic distributions.
6. **Cumulative Percentiles (P50 to P99)** and interactive manufacturing yield calculations.
7. **Industry Presets** for Commercial, Precision, and High-Precision fabrication.
8. **Interactive GUI Dock** (`tolerancing_dock.py`) in `v2` and `optics_app` main windows.
9. **AI Tool Integration** (`run_zemax_tolerance_analysis`) and chat heuristic routing.
10. **Zemax File Integration** exporting formatted ASCII reports and `.zmx` files with `TOL` records.
