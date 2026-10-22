# 2D Optics AI Assistant — Optical Design Guide & AI Sequences
============================================================

## 1. Introduction & Triad Architecture

The 2D Optics AI platform incorporates automated, physics-informed optical design sequences derived from seminal optical engineering literature (Coddington, Kingslake, Kidger, Fischer, Smith, Glatzel) and modern AI agent paradigms (OptiAgent, Yow et al. 2026).

The design workflows are structured across three interconnected layers:
1. **Engine Layer (`optics_app.engine.ai_sequences`, `extended_merit`)**:
   - Discrete, modular design stages implementing exact physical models.
   - `OpticalDesignSequenceRunner`: Orchestrator supporting end-to-end execution (`.run_full_sequence()`) and single-step manual debugging (`.step()`).
   - Extended optical operands (`TRAR`, `RSCE`, `SPOT`, `MNET`) and Glatzel adaptive DLS optimization.
2. **AI Tool Registry Layer (`optics_app.ai.tools`, `v2.ai.tools`)**:
   - First-class agent tools registered with complete JSON schemas via `@registry.register`.
   - Accessible to AI agents via OpenAI-compatible function calling and programmatic execution via `registry.execute(...)`.
3. **AI Assistant Guidance & NLP Layer (`optics_app.ai.agent`, `nemotron_client`)**:
   - `NemotronAgent` system prompt equipped with full optical sequence methodologies and formulas.
   - Regex-based NLP heuristic parser (`parse_optical_command_heuristics`) translating natural language queries into sequence tool calls.

---

## 2. The 5/6-Stage Optical Design Methodology

```
+--------------------------+     +--------------------------+     +-------------------------------+
| Stage 1:                 |     | Stage 2:                 |     | Stage 3:                      |
| Paraxial Synthesis       | --> | Coddington Bending       | --> | Thick Lens Realization        |
| - Power: Phi = 1/f'      |     | - q_min = 2(n^2-1)/(n+2) |     | - Budget: tc >= 0.10*D, >=1.5 |
| - Thin equiconvex seed   |     | - Minimize S_I at zero   |     | - Mechanical te >= 1.5 mm     |
|                          |     |   power change           |     | - Gullstrand power conservation|
+--------------------------+     +--------------------------+     +-------------------------------+
                                                                                  |
                                                                                  v
+--------------------------+     +--------------------------+     +-------------------------------+
| Stage 6:                 |     | Stage 5:                 |     | Stage 4:                      |
| Tolerancing & Yield      | <-- | Aspherization            | <-- | Glatzel Adaptive DLS          |
| - Commercial tolerances  |     | - Conic null: k = -n^2   |     | - Parameter-specific damping  |
| - 500 Monte Carlo trials |     | - Polynomial alpha_4     |     |   D_jj = p^2*(J^T W^2 J)*gamma|
| - Yield >= 90%           |     | - Spot radius < 10 um    |     | - Dynamic steering by rho_j   |
+--------------------------+     +--------------------------+     +-------------------------------+
```

### Stage 1: Paraxial Synthesis (`ParaxialSynthesisStage`)
- Solves net required optical power $\Phi = 1 / f'$ (in $\text{mm}^{-1}$) for target Effective Focal Length (EFL).
- Computes entrance pupil diameter $D = f' / (F/\#)$ and conjugate factors.
- Resolves substrate glass refractive index $n$ at Fraunhofer d-line ($\lambda = 587.6\text{ nm}$).
- Seeds initial symmetric thin-lens curvatures:
  $$c_1 = -c_2 = \frac{\Phi}{2(n - 1)}$$

### Stage 2: Analytical Coddington Bending (`CoddingtonBendingStage`)
- Based on Coddington (1829) and Kingslake (1978): for an object at conjugate factor $p = (s' - s)/(s' + s)$ (where $p = -1$ for infinite conjugate distance $s = -\infty$), the optimal shape factor $q_{\min}$ that minimizes primary spherical aberration $S_I$ is:
  $$q_{\min} = \frac{2(n^2 - 1)}{n + 2} |p|$$
- For standard crown glass (N-BK7, $n \approx 1.5168$), $q_{\min} \approx 0.7397$, yielding a strongly asymmetric biconvex "best-form" singlet.
- Derives front and rear surface curvatures preserving net optical power:
  $$c_1 = \frac{\Phi}{2(n - 1)} (1 + q_{\min})$$
  $$c_2 = \frac{\Phi}{2(n - 1)} (q_{\min} - 1)$$

### Stage 3: Thick Lens Realization (`ThickLensRealizationStage`)
- Converts the thin-lens approximation into a manufacturable physical `ThickLens`.
- Center thickness $t_c$ budgeted according to industrial optical standards:
  $$t_c = \max(1.5\text{ mm}, 0.10 \times D)$$
- Edge thickness $t_e$ evaluated at margin height $r = D/2$:
  $$t_e = t_c + \text{sag}_2(D/2) - \text{sag}_1(D/2) \ge 1.5\text{ mm}$$
- Optical power shift accounted for via the Gullstrand equation:
  $$\Phi_{\text{thick}} = \Phi_1 + \Phi_2 - \frac{t_c}{n} \Phi_1 \Phi_2$$
  with automatic curvature scaling to restore nominal focal length.

### Stage 4: Glatzel Adaptive DLS Optimization (`DLSAberrationMinimizationStage`)
- Implements Levenberg-Marquardt Damped Least Squares (DLS) with Erhard Glatzel's (1968) parameter-specific diagonal damping:
  $$(\mathbf{J}^T \mathbf{W}^2 \mathbf{J} + \mathbf{D}) \Delta \mathbf{x} = -\mathbf{J}^T \mathbf{W}^2 \mathbf{r}$$
  $$D_{jj} = p^2 (\mathbf{J}^T \mathbf{W}^2 \mathbf{J})_{jj} \cdot \gamma_j$$
- Dynamic steering factor $\gamma_j$ updates based on non-linear discrepancy ratio $\rho_j = |r_j(\mathbf{x} + \Delta\mathbf{x}) - (r_j + \mathbf{J}_j \Delta\mathbf{x})| / |r_j|$:
  - If $\rho_j > 0.5$ (runaway non-linearity): $\gamma_j \leftarrow \min(10^5, \gamma_j \times 2.0)$
  - If $\rho_j < 0.1$ (well-behaved linear regime): $\gamma_j \leftarrow \max(0.01, \gamma_j \times 0.7)$

### Stage 5: Aspherization Stage (`AspherizationStage`)
- Upgrades spherical element to `AsphericLens`.
- Evaluates ISO 10110-12 even polynomial sag:
  $$z(r) = \frac{c r^2}{1 + \sqrt{1 - (1 + k) c^2 r^2}} + \alpha_4 r^4 + \alpha_6 r^6$$
- Initializes front surface conic constant to theoretical aplanatic null (Fischer 2008):
  $$k = -n^2$$
  (for N-BK7, $k = -(1.5168^2) \approx -2.3007$).
- Refines $k_1$ and 4th-order coefficient $\alpha_4$ targeting RMS spot size ($RSCE < 10\ \mu\text{m}$) and exact EFL.

### Stage 6: Tolerancing & Manufacturing Yield (`TolerancingYieldStage`)
- Configures standard commercial fabrication tolerances:
  - Surface radius delta: $\Delta R / R = \pm 0.2\%$ ($\ge 0.05\text{ mm}$)
  - Center thickness delta: $\Delta t = \pm 0.05\text{ mm}$
  - Refractive index delta: $\Delta n = \pm 0.001$
  - Radial decenter: $\Delta y = \pm 0.025\text{ mm}$
  - Surface tilt: $\Delta \theta = \pm 0.05^\circ$
- Runs 50 to 500 Monte Carlo perturbation trials via `TolerancingEngine`.
- Verifies manufacturing yield $\ge 90\%$ against merit threshold $\text{MF} \le 3 \times \text{MF}_{\text{nominal}} + 1.0$.

---

## 3. Extended Merit Operands

Available in `optics_app.engine.extended_merit`:

| Operand | Name | Mathematical Definition | Purpose |
|---------|------|-------------------------|---------|
| `TRAR` | Transverse Ray Aberration | $y_{\text{intercept}} - y_{\text{chief}}$ | Evaluates ray transverse intercept errors at paraxial image plane. |
| `RSCE` | RMS Spot Radius | $\sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \bar{y})^2}$ | Primary optimization operand for focusing performance. |
| `SPOT` | 100% Geometric Spot Radius | $\max_{i} |y_i - \bar{y}|$ | Enforces hard upper bound on outermost flare rays. |
| `MNET` | Minimum Edge Thickness | $\max(0, t_{\text{min}} - t_e)$ | One-sided quadratic penalty enforcing manufacturable lens edge substance ($t_e \ge 1.5\text{ mm}$). |

---

## 4. Tool Registry Reference

Registered in both `optics_app.ai.tools` and `v2.ai.tools`:

### 1. `run_optical_design_sequence`
Executes complete 5/6-stage design sequence from paraxial synthesis to tolerancing.
```python
from optics_app.ai.tools import registry

result = registry.execute("run_optical_design_sequence", {
    "target_efl": 25.0,
    "target_diameter": 25.4,
    "target_f_number": 2.0,
    "glass": "N-BK7"
})
```
**Returns**:
- `status`: `"SUCCESS"`
- `target_efl`, `target_diameter`, `target_f_number`, `glass`
- `stages_executed`: 6
- `initial_metrics` vs `final_metrics` (`efl`, `merit`, `spot_radius`, `yield`)
- `total_improvement_factor`: $> 10\times$
- `final_lens`: Optimized `AsphericLens` object

### 2. `run_lens_bending_optimization`
Executes Stage 1 (Synthesis) and Stage 2 (Coddington Bending).
```python
result = registry.execute("run_lens_bending_optimization", {
    "efl": 25.0,
    "diameter": 25.4,
    "glass": "N-BK7"
})
```

### 3. `run_aspheric_optimization`
Executes Stages 1 through 5, delivering an aspherized singlet with conic constant and polynomial deformation.
```python
result = registry.execute("run_aspheric_optimization", {
    "efl": 25.0,
    "diameter": 25.4,
    "glass": "N-BK7"
})
```

### 4. `run_tolerancing_sequence`
Runs full design and conducts Monte Carlo simulation with specified trial count.
```python
result = registry.execute("run_tolerancing_sequence", {
    "efl": 25.0,
    "diameter": 25.4,
    "trials": 500
})
```

---

## 5. Single-Step Manual Debugging & Inspection

The `OpticalDesignSequenceRunner` architecture is built specifically to allow interactive stepping and debugging:

```python
from optics_app.engine.ai_sequences import OpticalDesignSequenceRunner

runner = OpticalDesignSequenceRunner(target_efl=25.0, target_diameter=25.4)

while not runner.is_finished:
    stage_name = runner.current_stage_name
    step_result = runner.step()
    print(f"Stage: {stage_name}")
    print(f"  Status: {step_result['status']}")
    print(f"  Merit before -> after: {step_result['metrics_before']['merit']:.4f} -> {step_result['metrics_after']['merit']:.4f}")
    print(f"  Spot radius: {step_result['metrics_after']['spot_radius']*1000.0:.2f} um")
```

---

## 6. Literature & Theoretical Citations

1. **Coddington, H. (1829)**. *A Treatise on the Reflexion and Refraction of Light*. Cambridge University Press.
2. **Kingslake, R. (1978)**. *Lens Design Fundamentals*. Academic Press, New York. (Chapter 3: The Aberrations of a Thin Lens; Coddington equations for minimum spherical aberration).
3. **Kidger, M. J. (2002)**. *Fundamental Optical Design*. SPIE Press Monograph PM92. (Chapter 7: Damped Least Squares Optimization; Chapter 9: Singlet & Doublet Design).
4. **Fischer, R. E., Tadic-Galeb, B., & Yoder, P. R. (2008)**. *Optical System Design* (2nd ed.). McGraw-Hill. (Chapter 8: Aspheric Surfaces; Conic constants $k = -n^2$ for infinite conjugate aplanats).
5. **Smith, W. J. (2007)**. *Modern Optical Engineering* (4th ed.). McGraw-Hill. (Chapter 11: The Design of Optical Systems; Tolerancing budgets and Monte Carlo yield).
6. **Glatzel, E. (1968)**. "New Principles of Lens Design: Adaptive Damping in Non-Linear Optimization." *Optica Acta*, 15(4), pp. 317–328.
7. **Bociort, F., & van Turnhout, M. (2008)**. "Generating alternative optical designs using saddle points." *Proc. SPIE 7060, Current Developments in Lens Design and Optical Engineering IX*.
8. **ISO 10110-12 (2019)**. *Optics and photonics — Preparation of drawings for optical elements and systems — Part 12: Aspheric surfaces*. International Organization for Standardization.

---

## 7. Illumination, Mask & Projection Optical Systems (Herbert Gross & Coursera)

The design of high-energy-throughput projection systems (photolithography steppers, laser micromachining projectors, display engines, and telecentric metrology trains) requires coupling an illumination subsystem, an intermediate mask/reticle plane, and a projection lens objective into a single coherent optical train.

```
+---------------------------------------------------------------------------------------------------------+
|                                    COMPLETE OPTICAL TRAIN ARCHITECTURE                                  |
+---------------------------------------------------------------------------------------------------------+
|  [SOURCE]  -->  [COLLECTOR]  -->  [HOMOGENIZER]  -->  [CONDENSER]  -->  [MASK]  -->  [PROJECTION]  --> [IMAGE]  |
|  Arc Lamp /      High-NA           Light Pipe /        Telecentric      Reticle /    Double-           Wafer /  |
|  Laser / LED     Doublet /         Fly's Eye           Relay Group      DMD / PLM    Telecentric       Screen   |
|  (P_0, NA_src)   Aspheric          Integrator          (f_cond, NA_ill) (2*y_mask)   Objective         (2*y_img)|
+---------------------------------------------------------------------------------------------------------+
        |                                   |                    |                        |
        v                                   v                    v                        v
  [PUPIL CONJUGATE]                   [FIELD CONJUGATE]    [FIELD CONJUGATE]        [PUPIL CONJUGATE]
  Source Emitter                      Homogenizer Exit     Mask / Reticle           Projection Stop / Exit
```

### 7.1 Köhler Illumination vs. Critical (Abbe) Illumination
- **Köhler Illumination (Herbert Gross Vol 2 & 6)**:
  - **Pupil Conjugate**: Source $\leftrightarrow$ Condenser Aperture Diaphragm $\leftrightarrow$ Projection Lens Entrance Pupil.
  - **Field Conjugate**: Field Stop (or Homogenizer exit) $\leftrightarrow$ Mask Plane $\leftrightarrow$ Wafer / Image Plane.
  - **Decoupling Principle**: In Köhler illumination, each point on the source emitter illuminates the entire mask field evenly, converting spatial arc or filament hot-spots into an ultra-uniform flat-top irradiance profile.
  - **Telecentric Illumination**: Placing the field stop at the front focal plane of the condenser renders chief rays parallel to the optical axis ($\theta_{\text{chief}} \approx 0$). This ensures angle-independent mask transmission and eliminates magnification error if the reticle shifts axially.
- **Critical (Abbe) Illumination**:
  - The condenser directly images the source emitter onto the mask plane.
  - Provides maximum central flux density, but any spatial filament/arc non-uniformity prints directly onto the mask and image.

### 7.2 Spatial Homogenization: Light Pipes & Fly's Eye Lenslet Arrays
- **Light Pipe (Integrator Rod / Kaleidoscope)**:
  - Solid quartz / N-BK7 or hollow reflective tunnel.
  - Bounded internal reflections fold the beam repeatedly:
    $$N_{\text{bounces}} \approx \frac{L \cdot \tan\theta_{\text{int}}}{W}$$
  - Enforcing $N_{\text{bounces}} \ge 4 - 6$ yields $>95\%$ spatial flat-top uniformity.
- **Fly's Eye Tandem Integrator Array**:
  - Pair of identical 2D micro-lens arrays (MLA1, MLA2) coupled to a Fourier condenser lens.
  - Subdivides the incoming beam into a grid of beamlets and superimposes their magnified images onto the mask plane.

### 7.3 Pupil Matching & Zero-Vignetting Solve
- For 100% geometric energy throughput from mask to image, the **illumination exit pupil must match the projection entrance pupil**:
  $$Z_{\text{exit\_pupil, ill}} = Z_{\text{entrance\_pupil, proj}}, \quad D_{\text{exit\_pupil, ill}} \approx \sigma \cdot D_{\text{entrance\_pupil, proj}}$$
- Pupil mismatch causes severe peripheral ray clipping (vignetting loss $> 50\%$) and field-dependent apodization.
- The partial coherence parameter $\sigma$ balances throughput, resolution, and edge contrast:
  $$\sigma = \frac{NA_{\text{ill, mask}}}{NA_{\text{proj, obj}}} = \frac{NA_{\text{condenser}}}{NA_{\text{obj}}}$$
  *(Typical lithography values: $\sigma \in [0.5, 0.85]$)*.

---

## 8. Radiometric Throughput & Étendue Conservation Principles

### 8.1 Conservation of Étendue (Optical Invariant)
*(Coursera Optical Engineering & Herbert Gross Vol 6)*
- Geometrical Étendue (Throughput $G$):
  $$G = A \cdot \Omega = \pi A \cdot NA^2 = \pi^2 H^2 \quad [\text{mm}^2 \cdot \text{sr}]$$
- The Second Law of Thermodynamics dictates that **optical étendue can never decrease without energy loss**:
  $$G_{\text{image}} \le G_{\text{projection}} \le G_{\text{mask}} \le G_{\text{condenser}} \le G_{\text{source}}$$
- The system limiting throughput is determined by the minimum étendue aperture:
  $$G_{\text{limit}} = \min(G_{\text{source}}, G_{\text{condenser}}, G_{\text{mask}}, G_{\text{proj}}, G_{\text{img}})$$

### 8.2 End-to-End Power Transfer Budget (Watts)
$$\Phi_{\text{image}} = \Phi_{\text{source}} \cdot \eta_{\text{coll}} \cdot \tau_{\text{homog}} \cdot \tau_{\text{cond}} \cdot \tau_{\text{mask}} \cdot \eta_{\text{pupil}} \cdot \tau_{\text{proj}}$$
- **Collection Efficiency** ($\eta_{\text{coll}}$): $\min(1.0, (NA_{\text{coll}} / NA_{\text{src}})^2) \cdot \tau_{\text{AR}}$.
- **Pupil Coupling Efficiency** ($\eta_{\text{pupil}}$): $(1 / \max(1.0, D_{\text{ill}} / D_{\text{proj}}))^2 \cdot \exp(-(\Delta Z / 60)^2)$.
- **Total Throughput Percentage**: $\eta_{\text{total}} = (\Phi_{\text{image}} / \Phi_{\text{source}}) \times 100\%$.

### 8.3 Lithography Resolution & Depth of Focus (DOF)
- Rayleigh resolution:
  $$R_{\text{Rayleigh}} = 0.61 \cdot \frac{\lambda}{NA_{\text{img}}}$$
- Lithographic minimum half-pitch:
  $$HP = k_1 \cdot \frac{\lambda}{NA_{\text{img}}} \quad (k_1 \approx 0.40)$$
- Rayleigh Depth of Focus ($\pm DOF$):
  $$DOF = \pm \frac{\lambda}{2 \cdot NA_{\text{img}}^2}$$

---

## 9. Authoritative Illumination & Projection Literature Citations

1. **Gross, Herbert (Ed.) (2005)**. *Handbook of Optical Systems, Volume 2: Physical Image Formation*. Wiley-VCH, Weinheim. ISBN: 978-3-527-40378-3. (Chapter 17: Coherence; Chapter 18: Abbe Theory and Köhler Illumination).
2. **Gross, Herbert, Blechinger, Fritz, & Achtner, Bertram (2008)**. *Handbook of Optical Systems, Volume 4: Survey of Optical Systems*. Wiley-VCH, Weinheim. ISBN: 978-3-527-40380-6. (Chapter 34: Lithographic Projection Objectives & Double-Telecentric Systems).
3. **Gross, Herbert, Kuhn, Dirk, & Siehr, Harald (2012)**. *Handbook of Optical Systems, Volume 6: Illumination Systems*. Wiley-VCH, Weinheim. (Collector Optics, Integrator Rods, Fly's Eye Arrays, Radiometry & Étendue Matching).
4. **Coursera / University of Colorado Boulder (2020)**. *Optical Engineering Specialization: First-Order Optical System Design & Optical Aberrations*. Prof. Robert McLeod.
5. **Mack, Chris A. (2007)**. *Fundamental Principles of Optical Lithography: The Science of Microfabrication*. John Wiley & Sons. ISBN: 978-0-470-01893-4.
6. **Levenson, Marc D. (1982)**. "Improving resolution in photolithography with a phase-shifting mask." *IEEE Transactions on Electron Devices*, 29(12), pp. 1828–1836.
7. **Welford, W. T., & Winston, R. (1989)**. *High Collection Nonimaging Optics*. Academic Press, San Diego. ISBN: 978-0-12-742885-7.
