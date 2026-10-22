"""
NVIDIA Nemotron & Gemma Cloud API Client for 2D Optics Engine.

Connects to NVIDIA NIM API (https://integrate.api.nvidia.com/v1) and Gemma Cloud APIs
with dual-tier model configuration (high/medium), automated API key resolution
from environment/.env, demo integrity mode, and structured optical tool-calling.
"""

import os
import re
import sys
import json
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Generator

logger = logging.getLogger("NemotronClient")


class ModelFamily(str, Enum):
    """Supported Cloud AI Model Families."""
    NEMOTRON = "nemotron"
    GEMMA = "gemma"


class ModelTier(str, Enum):
    """Supported AI model tiers."""
    HIGH = "high"
    MEDIUM = "medium"


# Model Presets for NVIDIA NIM API (Nemotron & LLaMA)
TIER_MODEL_MAP = {
    ModelTier.HIGH: "nvidia/llama-3.1-nemotron-70b-instruct",
    ModelTier.MEDIUM: "meta/llama-3.1-8b-instruct",
}

# Alternate fallback high/medium models
TIER_FALLBACK_MAP = {
    ModelTier.HIGH: "nvidia/nemotron-4-340b-instruct",
    ModelTier.MEDIUM: "nvidia/nemotron-mini-4b-instruct",
}

# Model Presets for Gemma Models (NVIDIA NIM or Google Cloud)
GEMMA_TIER_MODEL_MAP = {
    ModelTier.HIGH: "google/gemma-2-27b-it",
    ModelTier.MEDIUM: "google/gemma-2-9b-it",
}

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
GEMMA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def find_and_load_dotenv() -> Dict[str, str]:
    """Search current and parent directories for .env and extract key-value pairs."""
    env_vars = {}
    search_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
        os.path.expanduser("~"),
    ]
    for d in search_dirs:
        env_file = os.path.join(d, ".env")
        if os.path.isfile(env_file):
            try:
                with open(env_file, "r", encoding="utf-8", errors="ignore") as fp:
                    for line in fp:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            env_vars[k] = v
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception as e:
                logger.debug(f"Could not read .env at {env_file}: {e}")
            break
    return env_vars


def _extract_axis(text: str, default: str = "x") -> str:
    """Extract coordinate axis (x, y, z) from natural language text with diverse phrasings."""
    low = text.lower()
    # Y-axis
    if (
        re.search(r"\b(?:around|along|about|in)\s+(?:the\s*)?y(?:\s*-\s*|\s+)axis\b", low)
        or re.search(r"\b(?:around|along|about|in)\s+y\b", low)
        or re.search(r"\by(?:\s*-\s*|\s+)axis\b", low)
        or re.search(r"\bin\s+(?:the\s+)?y(?:\s+direction)?\b", low)
        or "along y" in low or "around y" in low or "about y" in low
    ):
        return "y"
    # Z-axis
    if (
        re.search(r"\b(?:around|along|about|in)\s+(?:the\s*)?z(?:\s*-\s*|\s+)axis\b", low)
        or re.search(r"\b(?:around|along|about|in)\s+z\b", low)
        or re.search(r"\bz(?:\s*-\s*|\s+)axis\b", low)
        or re.search(r"\bin\s+(?:the\s+)?z(?:\s+direction)?\b", low)
        or "along z" in low or "around z" in low or "about z" in low
    ):
        return "z"
    # X-axis
    if (
        re.search(r"\b(?:around|along|about|in)\s+(?:the\s*)?x(?:\s*-\s*|\s+)axis\b", low)
        or re.search(r"\b(?:around|along|about|in)\s+x\b", low)
        or re.search(r"\bx(?:\s*-\s*|\s+)axis\b", low)
        or re.search(r"\bin\s+(?:the\s+)?x(?:\s+direction)?\b", low)
        or "along x" in low or "around x" in low or "about x" in low
    ):
        return "x"
    return default


def _extract_pivot(text: str, default: Any = "front_vertex") -> Any:
    """Extract pivot point from natural language rotation commands."""
    low = text.lower()
    m_vec = re.search(r"(?:about|around|at)?\s*pivot\s*[\[\(]([+\-0-9.,\s]+)[\]\)]", low)
    if m_vec:
        try:
            parts = [float(x.strip()) for x in m_vec.group(1).split(",") if x.strip()]
            if parts:
                return parts
        except ValueError:
            pass

    m_vec2 = re.search(r"pivot\s*=\s*[\[\(]([+\-0-9.,\s]+)[\]\)]", low)
    if m_vec2:
        try:
            parts = [float(x.strip()) for x in m_vec2.group(1).split(",") if x.strip()]
            if parts:
                return parts
        except ValueError:
            pass

    if any(k in low for k in ["center of mass", "center", "com", "mid", "middle"]):
        return "mid"
    if any(k in low for k in ["rear_vertex", "back_vertex", "rear", "back"]):
        return "rear_vertex"
    if any(k in low for k in ["front_vertex", "front", "vertex"]):
        return "front_vertex"
    return default


def _extract_zmx_path(raw: str, default: str) -> str:
    """Extract a Zemax (.zmx) file path from raw text, supporting quotes and spaces."""
    # 1. Quoted path: "..." or '...'
    m_quoted = re.search(r'["\']([^"\']+\.zmx)["\']', raw, re.IGNORECASE)
    if m_quoted:
        return m_quoted.group(1).strip()

    # 2. Path preceded by keywords (file, to, from, import, export, load, open)
    m_kw = re.search(r'(?:file|to|from|import|export|load|open)\s+([a-zA-Z]:[^\r\n,;]+?\.zmx|\/[^\r\n,;]+?\.zmx|[^\s"\'=]+?\.zmx)', raw, re.IGNORECASE)
    if m_kw:
        return m_kw.group(1).strip()

    # 3. Standard Windows drive path: C:\...zmx (allowing spaces until .zmx)
    m_win = re.search(r'([a-zA-Z]:[^\r\n,;"\']+\.zmx)', raw, re.IGNORECASE)
    if m_win:
        return m_win.group(1).strip()

    # 4. Standard POSIX or unquoted word path
    m_std = re.search(r'([^\s"\'<>]+\.zmx)', raw, re.IGNORECASE)
    if m_std:
        return m_std.group(1).strip()

    return default


def _resolve_mirror_or_lens_idx(text_low: str) -> Optional[Union[int, str]]:
    """Helper to resolve element index from ordinal phrases ('first mirror', '2nd mirror', etc.),
    generic phrases ('the mirror', 'mirror 1', 'mirror'), or explicit element references."""
    # 1. Mirror resolution
    if bool(re.search(r"\b(?:mirror|mirrors|m1|m2)\b", text_low)):
        target_ordinal = 1
        if re.search(r"\b(?:second|2nd)\s+mirror\b|\bmirror\s*#?\s*2\b|\bm2\b", text_low):
            target_ordinal = 2
        elif re.search(r"\b(?:third|3rd)\s+mirror\b|\bmirror\s*#?\s*3\b|\bm3\b", text_low):
            target_ordinal = 3
        elif re.search(r"\b(?:first|1st)\s+mirror\b|\bmirror\s*#?\s*1\b|\bm1\b", text_low):
            target_ordinal = 1

        try:
            try:
                from v2.ai.tools import get_active_optical_system
            except ImportError:
                from optics_app.ai.tools import get_active_optical_system
            act_sys = get_active_optical_system()
            if act_sys and hasattr(act_sys, "surfaces"):
                try:
                    from optics_app.engine.surface import SurfaceType
                except ImportError:
                    from optics_app.engine.surface import SurfaceType
                mirrs = []
                for i, s in enumerate(act_sys.surfaces):
                    st = getattr(s, "surface_type", None)
                    gl = str(getattr(s, "glass", "")).upper()
                    nm = str(getattr(s, "name", "")).upper()
                    if (st == SurfaceType.MIRROR or gl == "MIRROR" or "MIRROR" in nm or re.match(r"^M[0-9]", nm)) and st != SurfaceType.GRATING and "GRATING" not in nm and "DMD" not in nm:
                        mirrs.append(i)
                effective = mirrs
                if effective:
                    idx = target_ordinal - 1
                    if 0 <= idx < len(effective):
                        return effective[idx]
                    return effective[-1]
        except Exception:
            pass

        return 8 if target_ordinal == 2 else 2

    # 2. Explicit numbered element, lens, or surface (e.g. "element 2", "surface 1", "lens #3")
    m_num = re.search(r"\b(?:element|lens|surface)\s*#?\s*(\d+)\b", text_low)
    if m_num:
        return int(m_num.group(1))

    # 3. Ordinal lens and element resolution
    if re.search(r"\b(?:second|2nd)\s+(?:lens|element|surface)\b", text_low):
        return 2
    if re.search(r"\b(?:third|3rd)\s+(?:lens|element|surface)\b", text_low):
        return 3
    if re.search(r"\b(?:first|1st)\s+(?:lens|element|surface)\b", text_low):
        return 1

    # 4. Generic lens or element without number (e.g. "the lens", "the element", "lens", "element")
    if re.search(r"\b(?:the\s+)?(?:lens|element)\b", text_low):
        return 1

    return None


def parse_optical_command_heuristics(text: str) -> Optional[Dict[str, Any]]:
    """
    Interprets natural language optical commands and maps them into structured tool calls.
    Supports: optical design AI sequences, rotate, flip, displace, tilt, replace, offset bias,
    optimize, build spectrometer, q-factor.
    """
    raw = text.strip()
    low = raw.lower()

    # Mechanical optical routing: DMD/shutter/filter-wheel states.
    state_match = re.search(r"\b(?:turn|set|switch|put)?\s*(?:the\s+)?(dmd(?:4500)?|shutter|filter(?:\s+wheel)?)\s+(?:to\s+)?(on|off)\b", low)
    if state_match:
        component = "DMD" if state_match.group(1).startswith("dmd") else state_match.group(1)
        return {
            "name": "set_mechanical_state",
            "arguments": {"component": component, "state": state_match.group(2).upper()},
        }

    # Beamsplitter & Interferometer Reference Architectures (Zemax Approach)
    if any(k in low for k in ("interferometer", "optical isolator", "pbs isolator", "beam sampler", "power pick-off", "power pickoff")):
        wl_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*nm", low)
        wl_um = float(wl_m.group(1)) * 1e-3 if wl_m else 0.6328

        if "michelson" in low:
            d_m = re.search(r"([+\-]?[0-9]+(?:\.[0-9]+)?)\s*(?:um|µm)", low) or re.search(r"(?:delta|displacement|opd|shift)\s*(?:=|of|:)?\s*([+\-]?[0-9]+(?:\.[0-9]+)?)", low)
            d_val = float(d_m.group(1)) if d_m else 0.0
            bs_m = re.search(r"\b(bs\d{3}|bsw\d{2})\b", low)
            bs_p = bs_m.group(1).upper() if bs_m else "BS016"
            return {
                "name": "create_zemax_interferometer_design",
                "arguments": {
                    "design_type": "michelson",
                    "bs_part": bs_p,
                    "d_delta_um": d_val,
                    "wavelength_um": wl_um,
                }
            }
        elif "mach" in low or "zehnder" in low:
            phi_m = re.search(r"([+\-]?[0-9]+(?:\.[0-9]+)?)\s*(?:deg|°)", low) or re.search(r"(?:phase|shift|retardation)\s*(?:=|of|:)?\s*([+\-]?[0-9]+(?:\.[0-9]+)?)", low)
            phi_val = float(phi_m.group(1)) if phi_m else 0.0
            return {
                "name": "create_zemax_interferometer_design",
                "arguments": {
                    "design_type": "mach_zehnder",
                    "phase_shift_deg": phi_val,
                    "wavelength_um": wl_um,
                }
            }
        elif "isolat" in low:
            pbs_m = re.search(r"\b(pbs\d{3}|wpbs\d{2}-[a-z]+)\b", low)
            pbs_p = pbs_m.group(1).upper() if pbs_m else "PBS251"
            return {
                "name": "create_zemax_interferometer_design",
                "arguments": {
                    "design_type": "pbs_isolator",
                    "bs_part": pbs_p,
                    "wavelength_um": wl_um,
                }
            }
        elif "sampler" in low or "pick" in low:
            bs_m = re.search(r"\b(bs\d{3}|bsw\d{2})\b", low)
            bs_p = bs_m.group(1).upper() if bs_m else "BS019"
            return {
                "name": "create_zemax_interferometer_design",
                "arguments": {
                    "design_type": "beam_sampler",
                    "bs_part": bs_p,
                    "wavelength_um": wl_um,
                }
            }

    # Beamsplitter Polarization Analysis
    if any(k in low for k in ("polarization split", "polarization analysis", "extinction ratio", "jones vector", "stokes vector")) and \
       any(k in low for k in ("beamsplitter", "beam splitter", "pbs", "npbs", "cube", "plate", "bs0", "pbs1", "pbs2", "wpbs")):
        pn_m = re.search(r"\b(pbs\d{3}|bs\d{3}|bsw\d{2}|wpbs\d{2}(?:-[a-z]+)?|bp\d{3}[a-z0-9]*)\b", low)
        pn = pn_m.group(1).upper() if pn_m else "PBS251"
        pol = "S" if "s-pol" in low or " s pol" in low or "vertical" in low else ("circular" if "circular" in low else "P")
        wl_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*nm", low)
        wl_um = float(wl_m.group(1)) * 1e-3 if wl_m else 0.6328
        return {
            "name": "analyze_beamsplitter_polarization",
            "arguments": {
                "part_number": pn,
                "input_polarization": pol,
                "wavelength_um": wl_um,
            }
        }

    # Insert Thorlabs Beamsplitter
    if any(k in low for k in ("insert", "add", "place", "mount")) and \
       any(k in low for k in ("beamsplitter", "beam splitter", "pbs", "npbs", "splitter")):
        pn_m = re.search(r"\b(pbs\d{3}|bs\d{3}|bsw\d{2}|wpbs\d{2}(?:-[a-z]+)?|bp\d{3}[a-z0-9]*)\b", low)
        if pn_m:
            return {
                "name": "insert_thorlabs_beamsplitter",
                "arguments": {
                    "part_number": pn_m.group(1).upper(),
                }
            }

    # Illumination, Mask & Projection Systems (Herbert Gross & Coursera)
    if any(k in low for k in ("illumination", "mask projection", "stepper", "köhler", "kohler", "condenser", "energy throughput", "pupil matching", "projector", "dmd", "reticle", "lithography")):
        if any(k in low for k in ("throughput", "etendue", "irradiance", "uniformity")) and not any(k in low for k in ("design", "create", "build")):
            arch = "stepper_4x"
            if "stepper" in low or "litho" in low:
                arch = "stepper_4x"
            elif "laser" in low:
                arch = "laser_proj"
            elif "lamp" in low or "dmd" in low or "projector" in low or "display" in low:
                arch = "lamp_dmd"
            elif "metrology" in low or "inspection" in low:
                arch = "metrology_1x"
            return {
                "name": "calculate_energy_throughput_and_uniformity",
                "arguments": {"archetype": arch}
            }
        elif any(k in low for k in ("match pupil", "pupil matching", "solve pupil", "conjugate pupil")):
            arch = "stepper_4x"
            if "stepper" in low or "litho" in low:
                arch = "stepper_4x"
            elif "laser" in low:
                arch = "laser_proj"
            elif "lamp" in low or "dmd" in low or "projector" in low:
                arch = "lamp_dmd"
            elif "metrology" in low or "inspection" in low:
                arch = "metrology_1x"
            return {
                "name": "optimize_pupil_matching_and_throughput",
                "arguments": {"archetype": arch}
            }
        elif any(k in low for k in ("design", "create", "build", "synthesize", "setup", "illumination", "projector", "stepper")):
            arch = "stepper_4x"
            if "stepper" in low or "litho" in low:
                arch = "stepper_4x"
            elif "laser" in low:
                arch = "laser_proj"
            elif "lamp" in low or "dmd" in low or "display" in low or "projector" in low:
                arch = "lamp_dmd"
            elif "metrology" in low or "inspection" in low:
                arch = "metrology_1x"
            power_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*w(?:att)?\b", low)
            power_val = float(power_m.group(1)) if power_m else None
            wl_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*nm\b", low)
            wl_val = float(wl_m.group(1)) * 1e-3 if wl_m else None
            args: Dict[str, Any] = {"archetype": arch, "push_to_active_system": True}
            if power_val is not None:
                args["optical_power_w"] = power_val
            if wl_val is not None:
                args["wavelength_um"] = wl_val
            return {
                "name": "design_illumination_mask_projection_system",
                "arguments": args
            }

    # Ultrafast pulse-width / dispersion probe analysis.
    if any(term in low for term in ("pulse width", "pulse duration", "femtosecond", "fs pulse")) and \
       any(term in low for term in ("analy", "dispersion", "broaden", "compress", "probe", "measure")):
        wl_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*nm", low)
        duration_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*fs(?:\s+pulse)?", low)
        chirp_match = re.search(r"(?:gdd|chirp)\s*(?:=|of|:)?\s*([+\-]?[0-9]+(?:\.[0-9]+)?)\s*fs", low)
        arguments = {
            "center_wavelength_nm": float(wl_match.group(1)) if wl_match else 800.0,
            "transform_limit_fwhm_fs": float(duration_match.group(1)) if duration_match else 100.0,
            "initial_gdd_fs2": float(chirp_match.group(1)) if chirp_match else 0.0,
        }
        return {"name": "analyze_femtosecond_pulse", "arguments": arguments}

    # Laser scanning & Galvo actuator simulation commands:
    if ("laser scan" in low or "scan simulation" in low or "field aberration" in low or "galvo" in low or "scanner" in low or "f-theta" in low or "ftheta" in low or "scan" in low):
        axis = "XY" if ("2d" in low or "raster" in low or "xy" in low) else ("Y" if "y axis" in low or "y-axis" in low else "X")
        min_ang = -10.0
        max_ang = 10.0
        step = 1.0

        range_match = re.search(r"(?:\+/-|\+-|\u00b1)\s*(\d+(?:\.\d+)?)\s*(?:deg|°)?", low)
        if range_match:
            val = float(range_match.group(1))
            min_ang = -val
            max_ang = val
        else:
            span_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(\+?\d+(?:\.\d+)?)\s*(?:deg|°)?", low)
            if span_match:
                min_ang = float(span_match.group(1))
                max_ang = float(span_match.group(2))

        step_match = re.search(r"step\s*(?:size)?\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*(?:deg|°)?", low)
        if step_match:
            step = float(step_match.group(1))

        if "configure" in low and "galvo" in low:
            surf_match = re.search(r"surface\s*(\d+)", low)
            s_idx = int(surf_match.group(1)) if surf_match else 1
            return {
                "name": "configure_galvo_scanner",
                "arguments": {
                    "surface_index": s_idx,
                    "scan_axis": axis,
                    "min_angle_deg": min_ang,
                    "max_angle_deg": max_ang,
                    "step_size_deg": step,
                }
            }
        else:
            return {
                "name": "run_laser_scan_simulation",
                "arguments": {
                    "scan_axis": axis,
                    "min_angle_deg": min_ang,
                    "max_angle_deg": max_ang,
                    "step_size_deg": step,
                }
            }

    # Spectrometer Quality & Spectral Resolution Analysis (Zemax Approach)
    if any(k in low for k in ("spectrometer quality", "spectral analysis", "spectrometer resolution", "spectrometer throughput", "light throughput", "check the quality of a designed spectrometer", "check the quality of a spectrometer", "spectrometer quality following zemax")):
        slit_w = 25.0
        m_sw = re.search(r"slit\s*(?:width)?\s*(?:=|:|\s)?\s*([0-9.]+)\s*(?:um|µm)", low)
        if m_sw:
            try:
                slit_w = float(m_sw.group(1))
            except ValueError:
                pass

        na_val = 0.12
        m_na = re.search(r"\bna\s*(?:=|:|\s)?\s*([0-9.]+)\b", low) or re.search(r"numerical\s+aperture\s*(?:=|:|\s)?\s*([0-9.]+)", low)
        if m_na:
            try:
                na_val = float(m_na.group(1))
            except ValueError:
                pass

        laser_wl = None
        m_laser = re.search(r"(?:laser|raman|excitation)\s*(?:wavelength|wl)?\s*(?:=|:|\s)?\s*([0-9.]+)\s*(?:nm|um|µm)", low)
        if m_laser:
            try:
                val = float(m_laser.group(1))
                laser_wl = val * 1e-3 if val > 10.0 else val
            except ValueError:
                pass

        if any(e in low for e in ("export", "save report", "write report")):
            out_p = "analysis_outputs/spectrometer_quality_report.txt"
            m_out = re.search(r"(?:to|into|file)\s+([a-zA-Z0-9_\-\.\/\\\:]+\.txt)", raw)
            if m_out:
                out_p = m_out.group(1)
            return {
                "name": "export_spectrometer_quality_report",
                "arguments": {
                    "output_path": out_p,
                    "slit_width_um": slit_w,
                    "input_na": na_val,
                    "raman_laser_wavelength_um": laser_wl
                }
            }

        return {
            "name": "run_spectrometer_quality_analysis",
            "arguments": {
                "slit_width_um": slit_w,
                "input_na": na_val,
                "raman_laser_wavelength_um": laser_wl
            }
        }

    # 0a. Zemax Tolerancing Simulation Commands:
    # e.g. "run zemax tolerance analysis", "run tolerance analysis", "run monte carlo tolerancing with 1000 trials",
    # "check worst offenders", "simulate manufacturing yield"
    if ("tolerancing sequence" not in low) and (
        ("zemax" in low and ("toleran" in low or "simulation" in low)) or \
        ("tolerance analysis" in low) or ("run toleran" in low) or ("monte carlo" in low and "toleran" in low) or \
        ("worst offender" in low) or ("manufacturing yield" in low)
    ):

        if "export" in low and "report" in low:
            return {
                "name": "export_zemax_tolerance_report",
                "arguments": {
                    "output_filepath": "zemax_tolerance_report.txt",
                    "preset": "precision" if "precision" in low else "commercial"
                }
            }

        preset = "commercial"
        if "high" in low or "laser" in low:
            preset = "high_precision"
        elif "precision" in low:
            preset = "precision"

        criterion = "rms_spot_radius"
        if "geo" in low:
            criterion = "geo_spot_radius"
        elif "effl" in low or "focal" in low:
            criterion = "effl"
        elif "merit" in low:
            criterion = "merit_function"

        trials = 500
        m_tr = re.search(r"([0-9]+)\s*trials", low)
        if m_tr:
            try:
                trials = int(m_tr.group(1))
            except ValueError:
                pass

        return {
            "name": "run_zemax_tolerance_analysis",
            "arguments": {
                "preset": preset,
                "criterion": criterion,
                "compensator_refocus": True,
                "num_trials": trials,
                "run_monte_carlo": True,
                "run_sensitivity": True
            }
        }

    # 0. Optical Design AI Sequences & Workflows:
    # e.g. "run optical design sequence for 25mm lens", "optimize 25mm lens with ai sequence",
    # "design optical element efl 25 diameter 25.4", "run lens bending optimization", etc.
    if any(k in low for k in ["optical design sequence", "ai sequence", "lens bending", "aspheric optimization", "tolerancing sequence"]) or \
       (re.search(r"\bdesign\s+(?:an?\s+)?(?:optical\s+element|lens|singlet|asphere)\b", low)):

        # Target EFL extraction
        target_efl = 25.0
        m_efl = re.search(r"\befl\s*(?:=|:|\s)?\s*([0-9.]+)", low)
        if not m_efl:
            m_efl = re.search(r"focal\s+length\s*(?:=|:|\s)?\s*([0-9.]+)", low)
        if not m_efl:
            m_efl = re.search(r"for\s+(?:an?\s+)?([0-9.]+)\s*(?:mm)?\s*(?:lens|singlet|asphere|optics)?", low)
        if not m_efl:
            m_efl = re.search(r"([0-9.]+)\s*mm\s+(?:lens|singlet|asphere|element)", low)
        if not m_efl:
            m_efl = re.search(r"\b([0-9.]+)\s*mm\b", low)
        if m_efl:
            try:
                target_efl = float(m_efl.group(1))
            except ValueError:
                pass

        # Target diameter extraction
        target_diam = 25.4
        m_diam = re.search(r"(?:diam(?:eter)?|aperture)\s*(?:=|:|\s)?\s*([0-9.]+)", low)
        if not m_diam:
            m_diam = re.search(r"\bd\s*(?:=|:)\s*([0-9.]+)", low)
        if not m_diam:
            m_diam = re.search(r"([0-9.]+)\s*mm\s+diam", low)
        if m_diam:
            try:
                target_diam = float(m_diam.group(1))
            except ValueError:
                pass

        # Target F-number extraction
        target_f_num = None
        m_fnum = re.search(r"\bf\s*/\s*([0-9.]+)", low)
        if not m_fnum:
            m_fnum = re.search(r"f-number\s*(?:=|:|\s)?\s*([0-9.]+)", low)
        if m_fnum:
            try:
                target_f_num = float(m_fnum.group(1))
            except ValueError:
                pass

        # Glass extraction
        glass = "N-BK7"
        for g in ["N-BK7", "F_SILICA", "N-SF11", "N-SF6HT", "N-LAK22", "N-BAF10"]:
            if g.lower() in low or g.lower().replace("-", "") in low:
                glass = g
                break

        # Sub-sequence tools
        if "lens bending" in low or "bending optimization" in low:
            return {
                "name": "run_lens_bending_optimization",
                "arguments": {
                    "efl": target_efl,
                    "diameter": target_diam,
                    "glass": glass
                }
            }
        elif "aspheric optimization" in low:
            return {
                "name": "run_aspheric_optimization",
                "arguments": {
                    "efl": target_efl,
                    "diameter": target_diam,
                    "glass": glass
                }
            }
        elif "tolerancing sequence" in low:
            trials = 500
            m_tr = re.search(r"([0-9]+)\s*trials", low)
            if m_tr:
                try:
                    trials = int(m_tr.group(1))
                except ValueError:
                    pass
            return {
                "name": "run_tolerancing_sequence",
                "arguments": {
                    "efl": target_efl,
                    "diameter": target_diam,
                    "trials": trials
                }
            }

        args = {
            "target_efl": target_efl,
            "target_diameter": target_diam,
            "glass": glass
        }
        if target_f_num is not None:
            args["target_f_number"] = target_f_num

        return {
            "name": "run_optical_design_sequence",
            "arguments": args
        }

    # 1. Rotate optical element: e.g. "rotate element 1 by 5 degrees around x" or "rotate mirror by 5 deg"
    if "rotate" in low:
        elem_val = _resolve_mirror_or_lens_idx(low)
        if elem_val is None:
            m_elem = re.search(r"rotate\s+(?:the\s+)?(?:lens|element|surface|mirror)?\s*#?([0-9a-zA-Z_\-]+)", low)
            if m_elem:
                elem = m_elem.group(1)
                if elem.lower() in ("by", "along", "in", "the", "a", "an", "around", "about"):
                    elem_val = 1
                else:
                    elem_val = int(elem) if elem.isdigit() else elem
            else:
                elem_val = 1
        m_ang = re.search(r"(?:by\s+)?([+\-]?[0-9.]+)\s*(?:°|deg|degrees?)\b", low)
        if not m_ang:
            m_ang = re.search(r"\bby\s+([+\-]?[0-9.]+)", low)
        if m_ang:
            try:
                ang = float(m_ang.group(1))
                ax = _extract_axis(low, default="x")
                piv = _extract_pivot(low, default="front_vertex")
                tx = ang if ax == "x" else 0.0
                ty = ang if ax == "y" else 0.0
                tz = ang if ax == "z" else 0.0
                return {
                    "name": "rotate_optical_element",
                    "arguments": {
                        "element_idx": elem_val,
                        "tilt_x": tx,
                        "tilt_y": ty,
                        "tilt_z": tz,
                        "pivot": piv
                    }
                }
            except ValueError:
                pass

    # 2. Tilt optical element: e.g. "tilt lens 2 by -3.5 degrees" or "tilt the mirror by 5 deg"
    if "tilt" in low:
        elem_val = _resolve_mirror_or_lens_idx(low)
        if elem_val is None:
            m_elem = re.search(r"tilt\s+(?:the\s+)?(?:lens|element|surface|mirror)?\s*#?([0-9a-zA-Z_\-]+)", low)
            if m_elem:
                elem = m_elem.group(1)
                if elem.lower() in ("by", "along", "in", "the", "a", "an", "around", "about"):
                    elem_val = 1
                else:
                    elem_val = int(elem) if elem.isdigit() else elem
            else:
                elem_val = 1
        m_ang = re.search(r"(?:by\s+)?([+\-]?[0-9.]+)\s*(?:°|deg|degrees?)\b", low)
        if not m_ang:
            m_ang = re.search(r"\bby\s+([+\-]?[0-9.]+)", low)
        if m_ang:
            try:
                ang = float(m_ang.group(1))
                ax = _extract_axis(low, default="x")
                return {
                    "name": "tilt_optical_element",
                    "arguments": {
                        "element_idx": elem_val,
                        "angle_deg": ang,
                        "axis": ax
                    }
                }
            except ValueError:
                pass

    # 3. Flip optical element: e.g. "flip element 1" or "flip lens 2" or "flip mirror M1" or "flip the mirror"
    if "flip" in low:
        elem_val = _resolve_mirror_or_lens_idx(low)
        if elem_val is None:
            m_flip = re.search(r"flip\s+(?:the\s+)?(?:lens|element|surface|mirror)?\s*#?([0-9a-zA-Z_\-]+)", low)
            if m_flip:
                elem = m_flip.group(1)
                if elem.lower() in ("the", "a", "an", "front", "back"):
                    elem_val = 1
                else:
                    elem_val = int(elem) if elem.isdigit() else elem
            else:
                elem_val = 1
        return {
            "name": "flip_optical_element",
            "arguments": {"element_idx": elem_val}
        }

    # 4. Displace optical element: e.g. "displace element 1 by 2 mm in y" or "move mirror by 10mm" or "move the first mirror by 10mm"
    is_offset_command = "offset" in low or (
        "move" in low and any(k in low for k in ["to surface", "from surface", "relative to", "offset bias", "with an offset", "with offset"])
    )
    if not is_offset_command and any(k in low for k in ["displace", "translate", "shift", "move"]):
        elem_val = _resolve_mirror_or_lens_idx(low)
        if elem_val is None:
            m_elem = re.search(r"(?:displace|translate|shift|move)\s+(?:the\s+)?(?:lens|element|surface|mirror)?\s*#?([0-9a-zA-Z_\-]+)", low)
            if m_elem:
                elem = m_elem.group(1)
                if elem.lower() in ("by", "along", "in", "the", "a", "an", "to", "with"):
                    m_sec = re.search(r"(?:lens|element|surface|mirror)\s*#?([0-9a-zA-Z_\-]+)", low)
                    elem_val = int(m_sec.group(1)) if m_sec and m_sec.group(1).isdigit() else 1
                else:
                    elem_val = int(elem) if elem.isdigit() else elem
            else:
                elem_val = 1

        after_text = low
        m_dist = re.search(r"(?:by\s+)?([+\-]?[0-9.]+)\s*(?:mm|millimeters?)\b", after_text)
        if not m_dist:
            m_dist = re.search(r"by\s+([+\-]?[0-9.]+)", after_text)
        if m_dist:
            try:
                dist = float(m_dist.group(1))
                ax = _extract_axis(low, default="y")
                dx = dist if ax == "x" else 0.0
                dy = dist if ax == "y" else 0.0
                dz = dist if ax == "z" else 0.0
                return {
                    "name": "displace_optical_element",
                    "arguments": {
                        "element_idx": elem_val,
                        "decenter_x": dx,
                        "decenter_y": dy,
                        "decenter_z": dz
                    }
                }
            except ValueError:
                pass

    # 5. Replace optical element: e.g. "replace element 1 with AC254-050-C" or "replace mirror with CM254-050-P01"
    if "replace" in low:
        # Requirement-based catalog replacement. Quoted names are deliberately
        # resolved before the generic "the lens" fallback, which otherwise
        # collapses every named-lens command to surface 1.
        is_achromat_query = bool(re.search(r"\b(?:achromat(?:ic)?|achormatic|achromtic)\b", low))
        has_catalog_part = bool(re.search(r"\bAC\d{3}-\d{3}-[A-Z0-9]+\b", raw, re.IGNORECASE))
        if is_achromat_query and not has_catalog_part and "thorlabs" in low:
            target: Union[int, str, None] = None
            quoted_target = re.search(
                r"replace\s+(?:the\s+)?(?:lens|element)\s+[\"']([^\"']+)[\"']",
                raw,
                re.IGNORECASE,
            )
            if quoted_target:
                target = quoted_target.group(1).strip()
            else:
                named_target = re.search(
                    r"replace\s+(?:the\s+)?(?:lens|element)\s+(.+?)\s+(?:with|for)\s+",
                    raw,
                    re.IGNORECASE,
                )
                if named_target:
                    candidate = named_target.group(1).strip().strip('"\'')
                    if candidate.lower() not in ("a", "an", "the"):
                        target = candidate
            if target is None:
                target = _resolve_mirror_or_lens_idx(low)
            if target is None:
                target = 1

            explicit_diameter = re.search(
                r"(?:diameter|dia\.?|ø)\s*(?:of\s*)?([0-9]+(?:\.[0-9]+)?)\s*mm|"
                r"([0-9]+(?:\.[0-9]+)?)\s*mm\s*(?:diameter|dia\.?)",
                low,
            )
            focal_match = re.search(
                r"(?:focal\s+length|efl|f)\s*(?:of|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*mm",
                low,
            )
            generic_size = re.search(
                r"([0-9]+(?:\.[0-9]+)?)\s*mm\s+(?:focal\s+length\s+)?(?:achromat|achormatic|achromtic)",
                low,
            )
            focal_length = float(focal_match.group(1)) if focal_match else (
                float(generic_size.group(1)) if generic_size else None
            )
            diameter = None
            if explicit_diameter:
                diameter = float(explicit_diameter.group(1) or explicit_diameter.group(2))
            if focal_length is None:
                # In ordinary catalog language "a 30 mm achromat" means EFL;
                # diameter is only inferred when explicitly labelled above.
                unlabelled = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm", low)
                if unlabelled and diameter is None:
                    focal_length = float(unlabelled.group(1))
            if focal_length is not None:
                coating = None
                coating_match = re.search(r"\b(?:coating|coated)\s*[-:]?\s*([abc])\b", low)
                if coating_match:
                    coating = coating_match.group(1).upper()
                arguments: Dict[str, Any] = {
                    "element": target,
                    "focal_length_mm": focal_length,
                }
                if diameter is not None:
                    arguments["diameter_mm"] = diameter
                if coating is not None:
                    arguments["coating_code"] = coating
                return {"name": "replace_lens_from_thorlabs_query", "arguments": arguments}

        elem_val = _resolve_mirror_or_lens_idx(low)
        remainder = ""
        m_with = re.search(r"\bwith\s+(.+)$", low)
        if m_with:
            remainder = m_with.group(1).strip()

        if elem_val is None:
            m_rep = re.search(r"replace\s+(?:the\s+)?(?:lens|element|surface|mirror)?\s*#?([0-9a-zA-Z_\-]+)\s+with\s+(.+)$", low)
            if m_rep:
                elem = m_rep.group(1)
                elem_val = int(elem) if elem.isdigit() else elem
                if not remainder:
                    remainder = m_rep.group(2).strip()
            else:
                elem_val = 1

        if remainder:
            m_zmx_extracted = _extract_zmx_path(raw, "")
            if m_zmx_extracted and m_zmx_extracted.lower().endswith(".zmx"):
                part = m_zmx_extracted
            else:
                cleaned = re.sub(
                    r"^(?:a|an|the|new|custom|thorlabs|lens|mirror|doublet|achromat|grating|element)\s+",
                    "",
                    remainder,
                    flags=re.IGNORECASE
                ).strip()
                cleaned = re.sub(
                    r"^(?:a|an|the|new|custom|thorlabs|lens|mirror|doublet|achromat|grating|element)\s+",
                    "",
                    cleaned,
                    flags=re.IGNORECASE
                ).strip()

                m_token = re.search(r"([A-Za-z0-9]+[A-Za-z0-9_\-\.]*)", cleaned)
                part = m_token.group(1).upper() if m_token else cleaned.upper()
                part = re.sub(r"\s+(?:mirror|lens|doublet|grating|element)$", "", part, flags=re.IGNORECASE).strip()

                for cat_part in [
                    "AC254-050-C", "AC254-030-C", "AC254-075-C", "AC254-100-C", "AC254-150-C", "AC254-200-C",
                    "CM254-050-P01", "CM254-025-P01", "CM254-075-P01", "CM254-100-P01", "CM254-150-P01", "CM254-200-P01",
                    "GR25-0616", "GR25-0610", "GR25-0213", "A230-C", "DMD4500"
                ]:
                    if cat_part.lower() in low:
                        part = cat_part
                        break

            return {
                "name": "replace_optical_element",
                "arguments": {
                    "element_idx": elem_val,
                    "part_number": part
                }
            }

    # 6. Offset element / move to another surface offset bias:
    # e.g. "move element 2 to surface 1 with 15mm offset bias" or "offset mirror by 10 mm from surface 1"
    if "offset" in low or ("move" in low and ("surface" in low or "bias" in low or "offset" in low)):
        ax = _extract_axis(low, default="z")
        elem_val = _resolve_mirror_or_lens_idx(low)
        m_off_a = re.search(
            r"(?:move|offset)\s+(?:the\s+)?(?:first\s+|1st\s+|second\s+|2nd\s+)?(?:lens|element|surface|mirror)?\s*#?([0-9a-zA-Z_\-]+)?\s*"
            r"(?:to|relative to|from)\s+(?:(?:surface|element|lens|mirror)\s+)?#?([0-9a-zA-Z_\-]+)\s+"
            r"(?:(?:with|by)?\s*(?:(?:an?|the)?\s*offset(?:\s+bias)?(?:\s+of)?\s*)?|with\s+|by\s+)?"
            r"([+\-]?[0-9.]+)\s*(?:mm)?",
            low
        )
        m_off_b = re.search(
            r"(?:move|offset)\s+(?:the\s+)?(?:first\s+|1st\s+|second\s+|2nd\s+)?(?:lens|element|surface|mirror)?\s*#?([0-9a-zA-Z_\-]+)?\s*"
            r"(?:(?:by|with)\s+)?([+\-]?[0-9.]+)\s*(?:mm)?\s*"
            r"(?:(?:along|in|around|about)?\s*(?:the\s*)?[xyz](?:-|\s*)axis\s*)?"
            r"(?:(?:along|in|around|about)?\s*[xyz]\s*)?"
            r"(?:(?:an?|the)?\s*offset(?:\s+bias)?(?:\s+of)?\s*)?"
            r"(?:to|relative to|from)\s+(?:(?:surface|element|lens|mirror)\s+)?#?([0-9a-zA-Z_\-]+)",
            low
        )
        if m_off_a:
            raw_elem = m_off_a.group(1)
            if raw_elem and raw_elem.isdigit():
                elem_val = int(raw_elem)
            elif elem_val is None:
                elem_val = int(raw_elem) if raw_elem and raw_elem.isdigit() else (raw_elem if raw_elem else 1)
            ref = m_off_a.group(2)
            ref_val = int(ref) if ref and ref.isdigit() else ref
            bias = float(m_off_a.group(3))
            return {
                "name": "offset_optical_element",
                "arguments": {
                    "element_idx": elem_val,
                    "reference_surface_idx": ref_val,
                    "offset_bias_mm": bias,
                    "axis": ax
                }
            }
        elif m_off_b:
            raw_elem = m_off_b.group(1)
            if raw_elem and raw_elem.isdigit():
                elem_val = int(raw_elem)
            elif elem_val is None:
                elem_val = int(raw_elem) if raw_elem and raw_elem.isdigit() else (raw_elem if raw_elem else 1)
            bias = float(m_off_b.group(2))
            ref = m_off_b.group(3)
            ref_val = int(ref) if ref and ref.isdigit() else ref
            return {
                "name": "offset_optical_element",
                "arguments": {
                    "element_idx": elem_val,
                    "reference_surface_idx": ref_val,
                    "offset_bias_mm": bias,
                    "axis": ax
                }
            }

    # 7. Surface-specific autofocus. Keep this before generic optimization so
    # "optimize focus at Detector" does not silently become aperture tuning.
    if "optimiz" in low:
        focus_request = bool(re.search(r"\b(?:focus|focal|spot)\b", low)) and bool(
            re.search(r"\b(?:at|on|to)\s+(?:the\s+)?(?:surface\s+)?", low)
        )
        if focus_request:
            target: Union[int, str] = "Detector"
            quoted = re.search(r"(?:at|on|to)\s+(?:the\s+)?(?:surface\s+)?[\"']([^\"']+)[\"']", raw, re.IGNORECASE)
            indexed = re.search(r"(?:at|on|to)\s+(?:the\s+)?surface\s*#?\s*(\d+)", low)
            named = re.search(
                r"(?:at|on|to)\s+(?:the\s+)?(?:surface\s+)?([A-Za-z][A-Za-z0-9 _()\-]*?)(?:\s+by\s+|\s+using\s+|\s+within\s+|$)",
                raw,
                re.IGNORECASE,
            )
            if quoted:
                target = quoted.group(1).strip()
            elif indexed:
                target = int(indexed.group(1))
            elif named:
                target = named.group(1).strip().rstrip(" .")
                if target.casefold().endswith(" surface"):
                    target = target[:-8].rstrip()

            args: Dict[str, Any] = {"target_surface": target, "max_iterations": 80}
            adjust = re.search(
                r"(?:by\s+(?:adjusting|varying|moving)|using)\s+(?:the\s+)?(?:gap\s+after\s+)?(?:surface\s+)?(?:[\"']([^\"']+)[\"']|#?(\d+))",
                raw,
                re.IGNORECASE,
            )
            if adjust:
                args["adjust_surface"] = adjust.group(1).strip() if adjust.group(1) else int(adjust.group(2))
            return {"name": "optimize_focus_at_surface", "arguments": args}

        # Generic optimization: e.g. "optimize system parameters to minimize merit function".
        params = []
        if "aperture" in low or "beam_radius" in low or "stop" in low:
            params.append("aperture")
        if "wavelength" in low or "spectral" in low:
            params.append("wavelengths")
        if "field" in low:
            params.append("field")
        if not params:
            params = ["aperture"]
        return {
            "name": "run_ai_optimization",
            "arguments": {
                "parameter_names": params,
                "max_iterations": 15
            }
        }

    # 8. Import Zemax file: e.g. "import zemax file system.zmx" or "load zemax test.zmx" or "read zemax file test.zmx"
    if re.search(r"\b(?:import|load|open|read)\b", low) and ("zemax" in low or ".zmx" in low) and "replace" not in low and "catalog" not in low:
        filepath = _extract_zmx_path(raw, "imported_optical_system.zmx")
        return {
            "name": "import_zemax_file",
            "arguments": {
                "filepath": filepath
            }
        }

    # 9. Export Zemax file: e.g. "export system to zemax file system.zmx" or "save to zemax file output.zmx"
    if re.search(r"\b(?:export|save|write)\b", low) and ("zemax" in low or ".zmx" in low):
        filepath = _extract_zmx_path(raw, "exported_optical_system.zmx")
        return {
            "name": "export_zemax_file",
            "arguments": {
                "filepath": filepath
            }
        }

    # 10. Generative system design: e.g. "build a spectrometer with 785nm laser, 600 lpm grating, and 2x magnification"
    if any(k in low for k in ["build a spectrometer", "generate spectrometer", "design spectrometer", "create spectrometer", "build optical system"]):
        return {
            "name": "generate_optical_system",
            "arguments": {
                "prompt": raw
            }
        }

    # 11. Catalog query: e.g. "lookup AC254-050-C" or "Thorlabs AC254-050-C"
    m_cat = re.search(r"(?:lookup|specs? of|info for|thorlabs)\s+([A-Z0-9]{2,}[0-9A-Z\-]*)", raw, re.IGNORECASE)
    if m_cat:
        return {
            "name": "lookup_thorlabs_catalog",
            "arguments": {
                "part_number": m_cat.group(1).upper()
            }
        }

    # 12. Matrix propagation & Q-factor: e.g. "calculate matrix propagation and Q-factor"
    if "matrix" in low or "q-factor" in low or "q factor" in low:
        return {
            "name": "compute_matrix_propagation_and_q_factor",
            "arguments": {}
        }

    return None


class NemotronClient:
    """
    Client for interacting with NVIDIA Nemotron & Gemma models via Cloud APIs.
    Supports chat completion with structured tool-calling, multi-tier configuration,
    and demo integrity mode fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        family: Union[ModelFamily, str] = ModelFamily.NEMOTRON,
        tier: Union[ModelTier, str] = ModelTier.HIGH,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 4096,
        timeout_seconds: int = 120,
        demo_mode: Optional[bool] = None,
    ):
        # Auto-load .env
        find_and_load_dotenv()

        # Resolve Family
        if isinstance(family, str):
            try:
                self.family = ModelFamily(family.lower())
            except ValueError:
                self.family = ModelFamily.NEMOTRON
        else:
            self.family = family

        # Resolve Tier
        if isinstance(tier, str):
            try:
                self.tier = ModelTier(tier.lower())
            except ValueError:
                self.tier = ModelTier.HIGH
        else:
            self.tier = tier

        # Resolve Model
        if model:
            self.model = model
        else:
            if self.family == ModelFamily.GEMMA:
                self.model = GEMMA_TIER_MODEL_MAP.get(self.tier, GEMMA_TIER_MODEL_MAP[ModelTier.HIGH])
            else:
                self.model = TIER_MODEL_MAP.get(self.tier, TIER_MODEL_MAP[ModelTier.HIGH])

        # Resolve API Key
        if self.family == ModelFamily.GEMMA:
            self.api_key = (
                api_key
                or os.getenv("GEMMA_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("NVIDIA_API_KEY")
                or os.getenv("NVIDIA_NIM_API_KEY")
                or os.getenv("NEMOTRON_API_KEY")
            )
            self.base_url = (base_url or GEMMA_DEFAULT_BASE_URL).rstrip("/")
        else:
            self.api_key = (
                api_key
                or os.getenv("NVIDIA_API_KEY")
                or os.getenv("NVIDIA_NIM_API_KEY")
                or os.getenv("NEMOTRON_API_KEY")
            )
            self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

        # Demo mode resolution: defaults to demo integrity mode per task specification
        env_integrity = os.getenv("INTEGRITY_MODE", "demo").lower()
        if demo_mode is not None:
            self.demo_mode = demo_mode
        elif env_integrity == "live" and self.is_configured():
            self.demo_mode = False
        else:
            self.demo_mode = True

        # Test OpenAI SDK availability
        self._has_openai = False
        self._openai_client = None
        try:
            import openai
            self._has_openai = True
            if self.api_key and not self.demo_mode:
                self._openai_client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                )
        except ImportError:
            self._has_openai = False

    def is_configured(self) -> bool:
        """Returns True if a valid API key is present."""
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def set_tier(self, tier: Union[ModelTier, str]) -> None:
        """Switch active model tier (high/medium)."""
        if isinstance(tier, str):
            tier = ModelTier(tier.lower())
        self.tier = tier
        if self.family == ModelFamily.GEMMA:
            self.model = GEMMA_TIER_MODEL_MAP.get(self.tier, GEMMA_TIER_MODEL_MAP[ModelTier.HIGH])
        else:
            self.model = TIER_MODEL_MAP.get(self.tier, TIER_MODEL_MAP[ModelTier.HIGH])

    def set_family(self, family: Union[ModelFamily, str]) -> None:
        """Switch active model family (nemotron/gemma)."""
        if isinstance(family, str):
            family = ModelFamily(family.lower())
        self.family = family
        self.set_tier(self.tier)

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute chat completion with NVIDIA NIM or Gemma Cloud API,
        or demo mode when credentials are not supplied or in demo integrity mode.
        """
        if self.demo_mode or not self.is_configured():
            return self._demo_chat_completion(messages, tools=tools)

        try:
            if self._openai_client is not None:
                return self._call_with_openai(messages, tools=tools, tool_choice=tool_choice)
            else:
                return self._call_with_urllib(messages, tools=tools, tool_choice=tool_choice)
        except Exception as e:
            logger.warning(
                f"Cloud API request failed on model '{self.model}' ({e}); "
                f"gracefully falling back to demo/heuristic command parser."
            )
            return self._demo_chat_completion(messages, tools=tools)

    def _demo_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        High-fidelity demo mode: interprets natural language optical commands
        and provides realistic assistant tool calls and responses.
        """
        last_msg = messages[-1] if messages else {"role": "user", "content": ""}

        # If previous message was a tool result, synthesize a final answer
        if last_msg.get("role") == "tool":
            tool_name = last_msg.get("name", "tool")
            content = last_msg.get("content", "")
            return {
                "role": "assistant",
                "content": f"Successfully executed {tool_name}. Result summary:\n{content[:500]}",
                "tool_calls": [],
                "finish_reason": "stop"
            }

        # Otherwise, interpret user message
        user_content = last_msg.get("content", "")
        detected_tool = parse_optical_command_heuristics(user_content)

        if detected_tool:
            tool_name = detected_tool["name"]
            return {
                "role": "assistant",
                "content": f"Interpreted natural language command: '{user_content}'. Dispatching action via {tool_name}.",
                "tool_calls": [{
                    "id": "demo_call_001",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(detected_tool["arguments"])
                    }
                }],
                "finish_reason": "tool_calls"
            }

        # General response if no explicit command mapped
        return {
            "role": "assistant",
            "content": f"[{self.family.value.capitalize()} Demo Assistant] Received: '{user_content}'. Optical engine is ready for element manipulation, optimization, or generative design commands.",
            "tool_calls": [],
            "finish_reason": "stop"
        }

    def _call_with_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
    ) -> Dict[str, Any]:
        """Invoke chat completion using official openai package."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            response = self._openai_client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            result: Dict[str, Any] = {
                "role": msg.role or "assistant",
                "content": msg.content or "",
                "tool_calls": [],
                "finish_reason": choice.finish_reason,
            }

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })
            return result

        except Exception as e:
            logger.error(f"OpenAI client error on model {self.model}: {e}")
            return self._call_with_urllib(messages, tools=tools, tool_choice=tool_choice)

    def _call_with_urllib(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
    ) -> Dict[str, Any]:
        """Zero-dependency HTTP client using urllib.request."""
        import urllib.request
        import urllib.error

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": f"2D-Optics-{self.family.value.capitalize()}Agent/2.0",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                res_json = json.loads(body)

                choice = res_json["choices"][0]
                msg = choice["message"]

                result = {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content") or "",
                    "tool_calls": msg.get("tool_calls", []) or [],
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
                return result

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            logger.error(f"HTTP Error {e.code} from Cloud API: {err_body}")
            raise RuntimeError(f"Cloud API Error ({e.code}): {err_body}")
        except Exception as e:
            logger.error(f"Network error communicating with Cloud API: {e}")
            raise RuntimeError(f"Failed to connect to Cloud API at {endpoint}: {e}")


class GemmaClient(NemotronClient):
    """
    Client for interacting with Google Gemma models via Cloud APIs.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        tier: Union[ModelTier, str] = ModelTier.HIGH,
        base_url: Optional[str] = None,
        demo_mode: Optional[bool] = None,
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            family=ModelFamily.GEMMA,
            tier=tier,
            base_url=base_url or GEMMA_DEFAULT_BASE_URL,
            demo_mode=demo_mode,
            **kwargs
        )


# Convenience alias for unified client
CloudAiClient = NemotronClient
