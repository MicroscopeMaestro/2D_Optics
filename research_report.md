# Optical Literature Research Report: AI Sequences, Optimization Workflows & Merit Functions

**Author**: Project Orchestrator (`orchestrator_ai_seq_1`)  
**Target Repository**: `c:\Users\Juan\Documents\GitHub\2D_Optics`  
**Date**: 2026-09-17  
**Status**: Completed  
**Referenced Surveys**: 
- `c:\Users\Juan\Documents\GitHub\2D_Optics\.agents\explorer_lit_survey_1\analysis.md`
- `c:\Users\Juan\Documents\GitHub\2D_Optics\.agents\explorer_codebase_survey_1\analysis.md`
- `c:\Users\Juan\Documents\GitHub\2D_Optics\.agents\explorer_integration_survey_1\analysis.md`

---

## 1. Executive Summary

This report establishes the theoretical and mathematical foundation for integrating automated optical design sequences and advanced optimization workflows into the **2D Optics** application. 

Classical optical engineering and contemporary AI literature demonstrate that automated lens design is most effectively accomplished not by unconstrained black-box parameter searches, but through **physics-informed sequential workflows**. By decomposing the optical design process into discrete stages—from paraxial power synthesis, analytical Coddington shape bending, and thick-lens realization, to adaptive DLS optimization, aspheric deformation, and tolerancing—the optical assistant achieves rapid, deterministic convergence to manufacturable, high-performance designs.

---

## 2. Key Mathematical Formulations Extracted from Literature

### 2.1 First-Order Paraxial Optics & Cardinal Points
*(Source: Kingslake 1978, Kidger 2001, Smith 2007)*
- **Paraxial Refraction & Transfer Matrices**:
  $$\mathbf{R} = \begin{bmatrix} 1 & 0 \\ -\frac{n' - n}{R} & 1 \end{bmatrix}, \quad \mathbf{T} = \begin{bmatrix} 1 & \frac{d}{n} \\ 0 & 1 \end{bmatrix}$$
- **System ABCD Matrix**:
  $$\mathbf{M}_{\text{sys}} = \begin{bmatrix} A & B \\ C & D \end{bmatrix} = \mathbf{R}_k \mathbf{T}_{k-1} \dots \mathbf{T}_1 \mathbf{R}_1$$
- **Cardinal Points**:
  $$EFL = -\frac{1}{C}, \quad BFL = -\frac{A}{C}, \quad FFL = \frac{D}{C}$$
  $$P_1 = \frac{D - 1}{C}, \quad P_2 = \frac{1 - A}{C}$$

### 2.2 Analytical Coddington Shape Factor Bending
*(Source: Coddington 1829, Smith 2005, Shannon 1997)*
For a singlet lens of refractive index $n$ and conjugate position factor $p = \frac{s' - s}{s' + s}$ ($p = -1$ for an object at infinity), the 3rd-order spherical aberration $S_I$ is a quadratic function of the shape factor $q = \frac{r_2 + r_1}{r_2 - r_1}$:
- **Optimal Shape Factor for Minimum Spherical Aberration**:
  $$q_{\min} = \frac{2(n^2 - 1)}{n + 2} p$$
  *(For $n = 1.5168$ [N-BK7] at infinity conjugate $p = -1$: $q_{\min} \approx -0.7388$)*.
- **Surface Curvatures Derived from $q_{\min}$ and Total Power $\Phi$**:
  $$c_1 = \frac{\Phi}{2(n - 1)} (q + 1), \quad c_2 = \frac{\Phi}{2(n - 1)} (q - 1)$$

### 2.3 Seidel Stop-Shift Equations & Coma Nullification
*(Source: Kingslake 2010, Fischer 2008)*
When the aperture stop is shifted axially by distance $\delta z$, the chief ray height $\bar{y}$ shifts by $\bar{y}^* = \bar{y} + \bar{S} y$, where $\bar{S} = \frac{\delta z}{y}$.
- **Transformed Seidel Aberrations**:
  $$S_I^* = S_I$$
  $$S_{II}^* = S_{II} + \bar{S} S_I$$
  $$S_{III}^* = S_{III} + 2\bar{S} S_{II} + \bar{S}^2 S_I$$
- **Coma Nullification**:
  Setting $S_{II}^* = 0$ yields the optimal stop shift parameter:
  $$\bar{S}_{\text{opt}} = -\frac{S_{II}}{S_I}$$

### 2.4 Achromatic Doublet Power Distribution
*(Source: Kingslake 1978, Smith 2007)*
To nullify primary axial chromatic aberration ($C_L = 0$) for total power $\Phi$ using crown ($V_1$) and flint ($V_2$) glasses:
$$\Phi_1 = \Phi \left( \frac{V_1}{V_1 - V_2} \right), \quad \Phi_2 = -\Phi \left( \frac{V_2}{V_1 - V_2} \right)$$
$$\frac{\Phi_1}{V_1} + \frac{\Phi_2}{V_2} = 0$$

### 2.5 Damped Least Squares (DLS) & Glatzel's Adaptive Damping
*(Source: Levenberg 1944, Marquardt 1963, Glatzel 1968, Kidger 1993)*
Standard Levenberg-Marquardt minimizes the merit function $\Psi(\mathbf{x}) = \frac{1}{2} \|\mathbf{W} \mathbf{r}(\mathbf{x})\|_2^2$:
$$(\mathbf{J}^T \mathbf{W}^2 \mathbf{J} + \lambda \mathbf{I}) \Delta \mathbf{x} = -\mathbf{J}^T \mathbf{W}^2 \mathbf{r}$$
- **Erhard Glatzel's Adaptive Parameter Steering (1968)**:
  Uniform damping fails in optical systems due to extreme eigenvalue spreads ($\kappa > 10^8$). Glatzel replaces $\lambda \mathbf{I}$ with a parameter-specific diagonal matrix $\mathbf{D}$:
  $$(\mathbf{J}^T \mathbf{W}^2 \mathbf{J} + \mathbf{D}) \Delta \mathbf{x} = -\mathbf{J}^T \mathbf{W}^2 \mathbf{r}$$
  $$D_{jj} = p^2 (\mathbf{J}^T \mathbf{W}^2 \mathbf{J})_{jj} \cdot \gamma_j$$
  where $\gamma_j$ is a dynamic steering factor adjusted based on the discrepancy between the linear model prediction and actual non-linear ray evaluation:
  $$\rho_j = \frac{|r_{\text{actual}, j} - r_{\text{predicted}, j}|}{|r_{\text{predicted}, j}|}$$
  If $\rho_j > 0.5$, $\gamma_j \leftarrow \gamma_j \times 2.0$ (damp runaway parameters); if $\rho_j < 0.1$, $\gamma_j \leftarrow \max(0.1, \gamma_j \times 0.7)$ (accelerate well-behaved parameters).

### 2.6 Saddle-Point Construction (Bociort's Global Optimization)
*(Source: Bociort 2001–2008, TU Delft)*
To escape a stagnant local minimum $M_0$ without random perturbations:
1. Insert a zero-power meniscus "null element" ($c_1 = c_2, d = 0, n > 1$). Because power and thickness are zero, the system's merit value is identical to $M_0$.
2. The null insertion transforms the local minimum into a Morse index-1 saddle point $S$.
3. Compute the Hessian $\mathbf{H} = \mathbf{J}^T \mathbf{W}^2 \mathbf{J}$ at $S$ and find the eigenvector $\mathbf{v}_1$ corresponding to the unique negative eigenvalue $\mu_1 < 0$.
4. Perturb along $\pm \epsilon \mathbf{v}_1$ and run local DLS optimization to discover two distinct local minima.

### 2.7 Aspheric Surface Modeling (ISO 10110-12)
*(Source: ISO 10110-12, Forbes 2007, Fischer 2008)*
Surface sag $z(r)$ with conic constant $k$ and even polynomial deformation:
$$z(r) = \frac{c r^2}{1 + \sqrt{1 - (1+k) c^2 r^2}} + \alpha_4 r^4 + \alpha_6 r^6 + \alpha_8 r^8$$
- Conic constant for perfect spherical aberration nullification at infinite conjugate:
  $$k = -n^2$$
  *(For $n = 1.5168$, $k \approx -2.30$)*.

---

## 3. Optical Optimization Targets & Standard Merit Operands

To provide robust feedback for both human optical engineers and automated AI sequences, the optical merit function is formulated as:
$$\Psi(\mathbf{x}) = \sum_{k=1}^M w_k^2 \cdot \left[ \frac{O_k(\mathbf{x}) - T_k}{\sigma_k} \right]^2 + \sum_{j=1}^P P_j(\mathbf{x})$$

| Operand Code | Name | Mathematical Definition | Target / Purpose |
|--------------|------|-------------------------|------------------|
| **EFFL** | Effective Focal Length | $EFL = -1/C$ from ABCD matrix | Match nominal system focal length ($|EFL - T| \le 0.1\%$) |
| **BFL** | Back Focal Length | Distance from last surface to paraxial image plane | Enforce mechanical sensor clearance |
| **TRAR** | Transverse Ray Aberration | $y_{\text{intercept}} - y_{\text{chief}}$ at image plane | Minimize geometric ray deviation |
| **RSCE** | RMS Spot Radius (Centroid) | $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \bar{y})^2}$ | Primary resolution metric (diffraction-limited target) |
| **SPOT** | 100% Geometric Spot Size | $\max_i |y_i - \bar{y}|$ | Enforce total blur circle encirclement |
| **OPDX** | Optical Path Difference | $OPD(\rho, \theta) = [\text{Ray Path}] - [\text{Ref Sphere}]$ | Wavefront error minimization ($\le \lambda/4$) |
| **ZERN** | Fringe Zernike Coefficient | Projection onto orthonormal Zernike modes $Z_4$ (Defocus), $Z_9$ (Spherical) | Modal aberration diagnosis & balancing |
| **LONA** | Longitudinal Chromatic Aberration | $s'_{F} - s'_{C}$ | Nullify axial focal shift across Fraunhofer lines |
| **MNCT** | Minimum Center Thickness | $\min_i t_{c, i}$ | $t_c \ge 0.10 \times \varnothing$ (prevent structural failure) |
| **MNET** | Minimum Edge Thickness | $\min_i t_{e, i} = t_{c, i} - (z_{\text{front}}(r_e) - z_{\text{back}}(r_e))$ | $t_e \ge 1.5\text{ mm}$ (prevent edge chipping) |
| **RAIM** | Max Ray Incidence Angle | $\max_{k, i} |I_{k, i}|$ | $I \le 45^\circ$ (prevent high-order aberration blow-up) |

---

## 4. The 7-Step Canonical AI Lens Design Sequence

Based on classical literature (Kidger, Kingslake, Fischer, Smith, Shannon) and modern AI frameworks (OptiAgent, Yow et al. 2026), automated design is structured as a progressive multi-stage pipeline:

```
Step 1: Paraxial Thin-Lens Synthesis
  Input:  EFL, F-number (F/#), Half Field-of-View (HFOV), Conjugates
  Output: Ideal thin-lens power distribution (Phi_1, Phi_2...) and paraxial airspaces

Step 2: Analytical Coddington Shape Bending
  Input:  Thin lens powers, glass index n, conjugate factor p
  Action: Calculate q_min = (2(n^2 - 1)/(n + 2)) * p
  Output: Optimal starting curvatures c_1, c_2 minimizing 3rd-order spherical aberration S_I

Step 3: Physical Thick-Lens Realization & Boundary Enforcement
  Input:  Paraxial curvatures, clear aperture diameter D
  Action: Convert to thick lens with center thickness t_c >= 0.10*D and edge thickness t_e >= 1.5 mm
  Output: Physical thick lens with realistic geometric volume

Step 4: Seidel Stop-Position Optimization
  Input:  Thick lens system, marginal and chief ray heights
  Action: Solve stop-shift parameter S_bar = -S_II / S_I to nullify coma
  Output: Optimal aperture stop placement

Step 5: Chromatic Doublet Achromatization
  Input:  Net optical power Phi, Crown Abbe V1, Flint Abbe V2
  Action: Solve Phi_1 = Phi*(V1/(V1-V2)), Phi_2 = -Phi*(V2/(V1-V2))
  Output: Achromatic doublet nullifying longitudinal chromatic aberration (LONA = 0)

Step 6: Incidence-Angle Bottleneck Splitting (Fischer Heuristic)
  Trigger: Max surface incidence angle |I_k| > 45° or element power |Phi_i| > 1.5*Phi
  Action:  Split high-strain element into two singlets with half power (Phi_a ≈ Phi_b ≈ 0.5*Phi)
  Output:  Relaxed system with dramatically reduced higher-order aberrations

Step 7: Aspheric Deformation & High-Order Wavefront Nulling
  Input:  Optimized spherical system
  Action: 1. Introduce conic constant k = -n^2 to eliminate primary spherical aberration.
          2. Release even polynomial coefficients (alpha_4, alpha_6) with Glatzel adaptive DLS.
  Output: Diffraction-limited aspheric optical element
```

---

## 5. Architectural Blueprint for 2D Optics Application

To integrate these findings cleanly into the existing repository without violating modularity or regression constraints:

1. **Engine Layer (`optics_app/engine/ai_sequences.py`)**:
   - `OpticalDesignSequenceRunner`: Orchestrator class managing sequential stages (`ParaxialSynthesisStage`, `CoddingtonBendingStage`, `DLSAberrationMinimizationStage`, `AspherizationStage`, `TolerancingYieldStage`).
   - Supports `.step()` for single-step manual debugging (satisfying user constraint: *"make space for future new implementations, models and also debugging manually"*).
2. **Extended Operands (`optics_app/engine/extended_merit.py`)**:
   - Implements `TransverseRayAberrationOperand`, `SpotSizeRMSOperand`, `EdgeThicknessConstraintOperand`, and `GlatzelAdaptiveDLSOptimizer`.
3. **AI Tool Registry (`optics_app/ai/tools.py`)**:
   - Register `run_optical_design_sequence` with comprehensive JSON schemas for LLM tool invocation.
   - Fix `ToolRegistry.funcs` bug (`@property def funcs(self): return self.tools`) in `optics_app/ai/tools.py` and `v2/ai/tools.py`.
4. **AI Assistant Guidance (`optics_app/ai/agent.py` & `optical_design_guide.md`)**:
   - Update `SYSTEM_PROMPT` in `NemotronAgent` to equip the AI assistant with domain heuristics, Coddington formulas, and the 5-to-7 stage design philosophy.
5. **Programmatic Test / Demo Script (`verify_optical_ai_sequences.py`)**:
   - Executes the complete workflow on a sample optical element (e.g. 25 mm aspheric lens).
   - Programmatically asserts measurable improvements:
     - Focal length accuracy: $|EFL - 25.0| / 25.0 < 0.001$ ($<0.1\%$).
     - Merit reduction: $>10\times$ improvement from unoptimized starting state.
     - Spherical aberration / spot size: reduced to $<10\ \mu\text{m}$.
     - Monte Carlo manufacturing yield: $>90\%$ at standard commercial tolerances.

---

## 6. Authoritative Citations & References

1. **Kidger, Michael J.** (2001). *Fundamental Optical Design*. SPIE Press, Bellingham, WA. ISBN: 9780819439154. DOI: [10.1117/3.418015](https://doi.org/10.1117/3.418015).
2. **Kidger, Michael J.** (2004). *Intermediate Optical Design*. SPIE Press, Bellingham, WA. ISBN: 9780819455321. DOI: [10.1117/3.553210](https://doi.org/10.1117/3.553210).
3. **Kingslake, Rudolf, & Johnson, R. Barry** (2010). *Lens Design Fundamentals* (2nd ed.). Academic Press / Elsevier. ISBN: 9780123743015.
4. **Fischer, Robert E., Tadic-Galeb, Biljana, & Yoder, Paul R.** (2008). *Optical System Design* (2nd ed.). McGraw-Hill Professional. ISBN: 9780071472487.
5. **Smith, Warren J.** (2007). *Modern Optical Engineering: The Design of Optical Systems* (4th ed.). McGraw-Hill Professional. ISBN: 9780071476874.
6. **Smith, Warren J.** (2005). *Modern Lens Design* (2nd ed.). McGraw-Hill Professional. ISBN: 9780071448307.
7. **Shannon, Robert R.** (1997). *The Art and Science of Optical Design*. Cambridge University Press. ISBN: 9780521588683. DOI: [10.1017/CBO9780511608933](https://doi.org/10.1017/CBO9780511608933).
8. **Glatzel, Erhard, & Wilson, Ray** (1968). "The Adaptive Automatic Correction of Aberrations". *Applied Optics*, 7(2), 265–276. DOI: [10.1364/AO.7.000265](https://doi.org/10.1364/AO.7.000265).
9. **Bociort, Florian, & van Turnhout, Maarten** (2001). "Generating new lens designs by saddle-point construction". *Proceedings of SPIE*, 4442, 68–79. DOI: [10.1117/12.448405](https://doi.org/10.1117/12.448405).
10. **Bociort, Florian, & van Turnhout, Maarten** (2008). "Saddle points in the optical landscape". *Optics & Photonics News*, 19(4), 22–27. DOI: [10.1364/OPN.19.4.000022](https://doi.org/10.1364/OPN.19.4.000022).
11. **Grey, David S.** (1963). "Aberration Polynomials and Merit Function for Solving Optical Design Problems". *Journal of the Optical Society of America*, 53(6), 672–676. DOI: [10.1364/JOSA.53.000672](https://doi.org/10.1364/JOSA.53.000672).
12. **Levenberg, Kenneth** (1944). "A method for the solution of certain non-linear problems in least squares". *Quarterly of Applied Mathematics*, 2(2), 164–168.
13. **Marquardt, Donald W.** (1963). "An algorithm for least-squares estimation of nonlinear parameters". *SIAM Journal on Applied Mathematics*, 11(2), 431–441. DOI: [10.1137/0111030](https://doi.org/10.1137/0111030).
14. **OptiAgent Research Group** (2026). "OptiAgent: Physics-Informed Agentic Framework for Automated Lens Design". *arXiv preprint arXiv:2602.23761*.
15. **Yow, K. et al.** (2026). "Automated optical system synthesis using deep reinforcement learning on discrete-continuous action spaces". *Optica*, 13(3), 312–324.
16. **ISO 10110-12:2019** (2019). *Optics and photonics — Preparation of drawings for optical elements and systems — Part 12: Aspheric surfaces*. International Organization for Standardization, Geneva.
