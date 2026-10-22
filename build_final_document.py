# -*- coding: utf-8 -*-
import os

p_out = r"c:/Users/Juan/Documents/GitHub/2D_Optics/research_notes.md"

header = """# Comprehensive Literature Survey & Optical Architecture Reference Report
# High-Precision Folded Hybrid 1064 nm Raman Spectrometer

**Document Version**: 3.0 (Publication Grade)  
**Author**: Worker M1 (Research, Optical Physics & Architecture Specialist)  
**Date**: 2026-08-21  
**Project**: 1064 nm Hybrid Raman Spectrometer with DMD Spatial Light Modulation  
**Working Directory**: c:/Users/Juan/Documents/GitHub/2D_Optics/.agents/worker_m1/  
**Workspace Root**: c:/Users/Juan/Documents/GitHub/2D_Optics/  
**Target Optical Engine**: Folded In-Line Czerny-Turner Spectrograph with Strictly 1-Inch Thorlabs COTS Optics (AC254 Series Doublets, -C AR Coating \\text{--}1700\\text{ nm}$), Ruled Reflection Grating (GR25-0316, \\text{ lines/mm}$, Blaze \\text{ nm}$, Gold), Texas Instruments DLP4500 DMD (.664\\text{ mm} \\times 4.873\\text{ mm}$ active area), and Single-Element InGaAs PIN Photodiode ($\\varnothing 1.0\\text{ mm}$)

---

## Table of Contents
1. [Executive Summary & System Architecture Overview](#1-executive-summary--system-architecture-overview)
2. [State-of-the-Art Hybrid Raman & DMD Spectrometer Designs](#2-state-of-the-art-hybrid-raman--dmd-spectrometer-designs)
   - 2.1 The 1064 nm Raman Paradigm: Complete Autofluorescence Suppression
   - 2.2 Detector Architecture Evolution: Multi-Channel Linear Arrays vs. Spatial Light Modulators
   - 2.3 Operating Modes: Programmable Slewing Slit vs. Hadamard Transform Spectroscopy (HTS)
   - 2.4 Mathematical Formulations of Multiplex (Fellgett) & Throughput (Jacquinot) Advantages
3. [Texas Instruments DLP NIRscan Nano Reference Architecture Analysis](#3-texas-instruments-dlp-nirscan-nano-reference-architecture-analysis)
   - 3.1 Optical Engine Topology & Ray-Path Sequence (DLPA061 & DLPU030B)
   - 3.2 DMD Micromirror Physics & Tilt Dynamics ($\\pm 12.0^\\circ$)
   - 3.3 Vector Ray Reflection & Specular Glint Isolation at Normal Incidence ($\\theta_{\\text{in}} = 0.0^\\circ$)
   - 3.4 Étendue Invariant Conservation & Condenser Collection Theory
4. [Spectrometer Optical Architectures & Folded Layout Best Practices](#4-spectrometer-optical-architectures--folded-layout-best-practices)
   - 4.1 Classical Czerny-Turner vs. Crossed Czerny-Turner vs. Folded In-Line Refractive Doublet
   - 4.2 Aberration Physics: On-Axis Astigmatism Elimination & Coma Suppression
   - 4.3 Longitudinal Chromatic Aberration & Depth of Focus
   - 4.4 Mechanical Clearance, Coaxial SM1 Lens Tubes & Stray Light Baffling
5. [1064 nm Raman Spectroscopy Physics (\\text{ to }3400\\text{ cm}^{-1}$)](#5-1064-nm-raman-spectroscopy-physics-400-to-3400-text-cm-1)
   - 5.1 Inelastic Scattering Mechanism & Selection Rules
   - 5.2 Exact Mathematical Mapping: Wavenumber Shift $\\leftrightarrow$ Wavelength
   - 5.3 Landmark Raman Spectral Bands & Molecular Assignments Table
   - 5.4 Full-Band (.3\\text{--}1667.1\\text{ nm}$) vs. Intermediate (.3\\text{--}1613.3\\text{ nm}$) Targets
6. [Diffraction Grating Physics, Evanescent Cutoff Proofs & Selection](#6-diffraction-grating-physics-evanescent-cutoff-proofs--selection)
   - 6.1 Grating Equation & Angular Dispersion
   - 6.2 Rigorous Mathematical Proof of Evanescent Wave Cutoff for $\\ge 600\\text{ lines/mm}$ Gratings
   - 6.3 Grating Optimization: Thorlabs GR25-0316 (\\text{ lines/mm}$, Blaze \\text{ nm}$, Gold)
   - 6.4 Quasi-Littrow Angle of Incidence ($\\alpha = 12.00^\\circ$) Parametric Analysis
7. [Thorlabs 1-Inch NIR Achromatic Doublets & Optical Glass Prescriptions](#7-thorlabs-1-inch-nir-achromatic-doublets--optical-glass-prescriptions)
   - 7.1 Overview & Antireflection Coating (-C Coating: 1050–1700 nm)
   - 7.2 Complete Physical Prescriptions Catalog Table (, R_2, R_3, t_{c1}, t_{c2}, t_e, \\text{BFL}$)
   - 7.3 Doublet Orientations & Geometric Aberration Minimization
   - 7.4 3-Term Sellmeier Formulation & Optical Glass Dispersion
   - 7.5 Sellmeier Dispersion Coefficients Matrix
   - 7.6 Discrete Refractive Indices Table Across Landmark Raman Lines
8. [DMD4500 Spatial Footprint Matching & Single 1 mm Detector Focusing](#8-dmd4500-spatial-footprint-matching--single-1-mm-detector-focusing)
   - 8.1 Spectrograph Camera Doublet Sizing & Linear Dispersion Modeling
   - 8.2 Chief Ray Dispersion Spread Verification (.00\\text{ mm} \\le \\Delta x \\le 8.664\\text{ mm}$)
   - 8.3 Instrumental Bandpass & Spectral Resolution
   - 8.4 Post-DMD Two-Stage Telecentric Condenser Relay ( = 0.1067\\times$)
   - 8.5 Ray-Trace Spot Size Derivation ($\\le 0.850\\text{ mm}$ on $\\varnothing 1.0\\text{ mm}$ Detector)
   - 8.6 Zero Vignetting (.0\\%$ Ray Loss) Verification
9. [Complete Sequential Optical Surface Stack & Bill of Materials](#9-complete-sequential-optical-surface-stack--bill-of-materials)
   - 9.1 Sequential Optical Surface Stack (2D_Optics Schema v2)
   - 9.2 Bill of Materials (BOM)
10. [Conclusions & Downstream Implementation Roadmap](#10-conclusions--downstream-implementation-roadmap)
11. [Scientific & Technical References](#11-scientific--technical-references)
"""

with open(p_out, "w", encoding="utf-8") as f:
    f.write(header + "\n\n")

print("Header successfully written.")
