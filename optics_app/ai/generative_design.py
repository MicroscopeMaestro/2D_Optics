"""
Generative Optical System Design Engine.

Translates high-level natural language prompts (e.g. "build a spectrometer with
785nm laser, 600 lpm grating, and 2x magnification") into fully instantiated,
physically valid 2D optical system designs with complete surface layouts.
"""

import os
import re
import math
import json
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

try:
    from optics_app.engine.surface import Surface, SurfaceType
    from optics_app.engine.lde import SurfaceStack
    from optics_app.engine.models import SpectrometerSystem, SourceConfig
except ImportError:
    from v2.engine.surface import Surface, SurfaceType
    from v2.engine.lde import SurfaceStack
    from v2.engine.models import SpectrometerSystem, SourceConfig


class GenerativeOpticalDesigner:
    """
    Synthesizes complete 2D optical systems from natural language descriptions.
    """

    @classmethod
    def parse_specifications(cls, prompt: str) -> Dict[str, Any]:
        """
        Extract optical design specifications from natural language prompt.
        """
        text = prompt.lower()

        # 1. Wavelength / Laser
        wavelength_um = 0.785  # Default 785nm
        m_wl_nm = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:nm|nanometers?)", text)
        m_wl_um = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:um|µm|microns?)", text)

        if m_wl_nm:
            val = float(m_wl_nm.group(1))
            if val > 10.0:  # e.g. 785, 1064, 532
                wavelength_um = val / 1000.0
        elif m_wl_um:
            wavelength_um = float(m_wl_um.group(1))
        elif "1064" in text or "raman 1064" in text:
            wavelength_um = 1.064
        elif "785" in text or "raman 785" in text:
            wavelength_um = 0.785
        elif "532" in text:
            wavelength_um = 0.532
        elif "nir" in text or "infrared" in text:
            wavelength_um = 1.250

        # 2. Grating Grooves / Frequency
        grating_lpm = 600.0
        m_grat = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:lpm|l/mm|lines/mm|grooves/mm|lines\s*per\s*mm|grooves\s*per\s*mm)", text)
        if m_grat:
            grating_lpm = float(m_grat.group(1))
        elif "1200" in text:
            grating_lpm = 1200.0
        elif "300" in text:
            grating_lpm = 300.0
        elif "150" in text:
            grating_lpm = 150.0

        # 3. Magnification
        magnification = 1.0
        m_mag = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*x\s*(?:magnification|mag)?", text)
        m_mag2 = re.search(r"magnification\s*(?:of|is|:)?\s*([0-9]+(?:\.[0-9]+)?)", text)
        if m_mag:
            magnification = float(m_mag.group(1))
        elif m_mag2:
            magnification = float(m_mag2.group(1))

        # 4. Optics Style: All-Mirror vs Refractive/Lens
        optics_type = "mirror" if ("mirror" in text or "reflective" in text or "czerny" in text) else "mirror"
        if "lens" in text or "achromat" in text or "doublet" in text or "refractive" in text:
            optics_type = "lens"

        return {
            "wavelength_um": float(wavelength_um),
            "grating_lines_per_mm": float(grating_lpm),
            "magnification": float(magnification),
            "optics_type": optics_type,
            "raw_prompt": prompt
        }

    @classmethod
    def design_spectrometer(
        cls,
        wavelength_um: float = 0.785,
        grating_lines_per_mm: float = 600.0,
        magnification: float = 1.0,
        optics_type: str = "mirror",
        name: str = "AI_Generated_Spectrometer"
    ) -> SpectrometerSystem:
        """
        Build a complete, physically valid 2D spectrometer system.
        """
        stack = SurfaceStack()

        # Multi-wavelength band (+/- 5% of center wavelength)
        wl_center = float(wavelength_um)
        wls = [round(wl_center * 0.95, 4), round(wl_center, 4), round(wl_center * 1.05, 4)]
        stack.wavelengths = wls
        stack.primary_wavelength = wl_center

        # Collimator and Focuser focal lengths based on requested magnification
        f_coll = 50.0  # mm
        f_foc = f_coll * float(magnification)  # mm (e.g. 2x -> 100mm)

        # 1. Entrance Slit (Object Plane)
        s_obj = Surface(
            name="Entrance_Slit",
            surface_type=SurfaceType.OBJECT,
            radius=0.0,
            thickness=f_coll,
            glass="AIR",
            semi_diameter=1.0
        )
        stack.add_surface(s_obj)

        if optics_type == "mirror":
            # 2. Collimating Mirror M1
            r_coll = -2.0 * f_coll  # concave mirror
            s_m1 = Surface(
                name="Collimator_M1",
                surface_type=SurfaceType.MIRROR,
                radius=r_coll,
                thickness=40.0,
                glass="MIRROR",
                semi_diameter=12.7,
                tilt_x=-6.0
            )
            stack.add_surface(s_m1)

            # 3. Diffraction Grating
            # Grating equation Littrow angle: sin(alpha) = m * G * lambda / 2
            alpha_rad = math.asin(min(0.85, (1 * grating_lines_per_mm * wl_center * 1e-3) / 2.0))
            alpha_deg = math.degrees(alpha_rad)

            s_grat = Surface(
                name=f"Grating_{int(grating_lines_per_mm)}lpm",
                surface_type=SurfaceType.GRATING,
                radius=0.0,
                thickness=45.0,
                glass="MIRROR",
                semi_diameter=12.7,
                grating_lines_per_mm=grating_lines_per_mm,
                grating_order=1,
                tilt_x=float(alpha_deg)
            )
            stack.add_surface(s_grat)

            # 4. Focusing Mirror M2
            r_foc = -2.0 * f_foc
            s_m2 = Surface(
                name="Focusing_M2",
                surface_type=SurfaceType.MIRROR,
                radius=r_foc,
                thickness=f_foc,
                glass="MIRROR",
                semi_diameter=12.7 * max(1.0, magnification),
                tilt_x=6.0
            )
            stack.add_surface(s_m2)

        else:
            # Refractive / Lens Spectrometer
            # 2. Achromat Collimator
            s_coll_f = Surface(
                name="Collimator_Front",
                surface_type=SurfaceType.STANDARD,
                radius=28.3,
                thickness=6.0,
                glass="N-LAK22",
                semi_diameter=12.7
            )
            s_coll_m = Surface(
                name="Collimator_Mid",
                surface_type=SurfaceType.STANDARD,
                radius=-21.6,
                thickness=2.0,
                glass="N-SF6HT",
                semi_diameter=12.7
            )
            s_coll_b = Surface(
                name="Collimator_Back",
                surface_type=SurfaceType.STANDARD,
                radius=-105.0,
                thickness=40.0,
                glass="AIR",
                semi_diameter=12.7
            )
            stack.add_surface(s_coll_f)
            stack.add_surface(s_coll_m)
            stack.add_surface(s_coll_b)

            # 3. Grating
            alpha_rad = math.asin(min(0.85, (1 * grating_lines_per_mm * wl_center * 1e-3) / 2.0))
            s_grat = Surface(
                name=f"Grating_{int(grating_lines_per_mm)}lpm",
                surface_type=SurfaceType.GRATING,
                radius=0.0,
                thickness=45.0,
                glass="MIRROR",
                semi_diameter=12.7,
                grating_lines_per_mm=grating_lines_per_mm,
                grating_order=1,
                tilt_x=float(math.degrees(alpha_rad))
            )
            stack.add_surface(s_grat)

            # 4. Focusing Lens
            s_foc_f = Surface(
                name="Focuser_Front",
                surface_type=SurfaceType.STANDARD,
                radius=28.3 * magnification,
                thickness=6.0,
                glass="N-LAK22",
                semi_diameter=12.7 * max(1.0, magnification)
            )
            s_foc_m = Surface(
                name="Focuser_Mid",
                surface_type=SurfaceType.STANDARD,
                radius=-21.6 * magnification,
                thickness=2.0,
                glass="N-SF6HT",
                semi_diameter=12.7 * max(1.0, magnification)
            )
            s_foc_b = Surface(
                name="Focuser_Back",
                surface_type=SurfaceType.STANDARD,
                radius=-105.0 * magnification,
                thickness=f_foc,
                glass="AIR",
                semi_diameter=12.7 * max(1.0, magnification)
            )
            stack.add_surface(s_foc_f)
            stack.add_surface(s_foc_m)
            stack.add_surface(s_foc_b)

        # 5. Detector / Image Plane
        s_det = Surface(
            name="Detector_0.8mm",
            surface_type=SurfaceType.DETECTOR,
            radius=0.0,
            thickness=0.0,
            glass="AIR",
            semi_diameter=0.4
        )
        stack.add_surface(s_det)

        stack._update_accumulated_positions()

        # Build SpectrometerSystem container
        source_cfg = SourceConfig(
            source_type="Collimated",
            beam_radius=4.0,
            divergence_angle=5.0,
            num_rays=9,
            wavelengths=f"{wls[0]}, {wls[1]}, {wls[2]}",
            input_aperture_width=0.03,
            numerical_aperture=0.12
        )

        system_cfg = {
            "aperture_value": 8.0,
            "slit_width_um": 30.0,
            "magnification": magnification,
            "center_wavelength_um": wl_center,
            "grating_lines_per_mm": grating_lines_per_mm
        }

        system = SpectrometerSystem(
            surface_stack=stack,
            wavelengths=wls,
            primary_wavelength=wl_center,
            source_config=source_cfg,
            system_config=system_cfg,
            name=name
        )
        system.auto_detect_indices()
        return system

    @classmethod
    def design_from_prompt(
        cls,
        prompt: str,
        output_json_path: Optional[str] = None
    ) -> SpectrometerSystem:
        """
        Generate complete optical system from a natural language prompt.
        Optionally saves to JSON file.
        """
        specs = cls.parse_specifications(prompt)
        sys_name = f"Spectrometer_{int(specs['wavelength_um']*1000)}nm_{int(specs['grating_lines_per_mm'])}lpm_{specs['magnification']}x"

        system = cls.design_spectrometer(
            wavelength_um=specs["wavelength_um"],
            grating_lines_per_mm=specs["grating_lines_per_mm"],
            magnification=specs["magnification"],
            optics_type=specs["optics_type"],
            name=sys_name
        )

        if output_json_path:
            system.save_to_json(output_json_path)

        return system
