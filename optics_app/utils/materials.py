"""
Optical Glass Catalog and Refractive Index Lookup Utility for V2 Engine.
Provides clean Sellmeier and Cauchy dispersion calculations with zero circular dependencies.
"""
from typing import Dict, Any, Union, Optional, List, Tuple
import math
import numpy as np


class SellmeierDispersion:
    """
    Three-term Sellmeier dispersion model:
    n^2(lambda) = 1 + sum_{i=1}^3 (B_i * lambda^2) / (lambda^2 - C_i)
    where lambda is in micrometers (um) and C_i is in um^2.
    """
    def __init__(self, B1: float, B2: float, B3: float, C1: float, C2: float, C3: float):
        self.B = [float(B1), float(B2), float(B3)]
        self.C = [float(C1), float(C2), float(C3)]

    def get_index(self, wavelength_um: float) -> float:
        wl = float(wavelength_um)
        if wl <= 0.0:
            raise ValueError(f"Wavelength must be positive, got {wavelength_um} um")
        wl2 = wl ** 2
        n2 = 1.0
        for b, c in zip(self.B, self.C):
            denom = wl2 - c
            if abs(denom) < 1e-12 or (c > 0 and abs(denom) / c < 1e-6):
                raise ValueError(f"Wavelength {wavelength_um} um near Sellmeier resonance pole {c}.")
            n2 += (b * wl2) / denom
        if n2 < 1.0:
            raise ValueError(f"Calculated refractive index n^2 ({n2}) < 1.0")
        return float(math.sqrt(n2))


class CauchyDispersion:
    """
    Cauchy dispersion model:
    n(lambda) = A + B / lambda^2 + C / lambda^4
    """
    def __init__(self, A: float, B: float = 0.0, C: float = 0.0):
        self.A = float(A)
        self.B = float(B)
        self.C = float(C)

    def get_index(self, wavelength_um: float) -> float:
        wl = float(wavelength_um)
        if wl <= 0.0:
            raise ValueError(f"Wavelength must be positive, got {wavelength_um} um")
        wl2 = wl ** 2
        return float(self.A + (self.B / wl2) + (self.C / (wl2 ** 2)))


class GlassCatalog:
    """
    Exhaustive optical glass catalog mapping material names and aliases
    to dispersion instances.
    """
    _BK7 = SellmeierDispersion(1.03961212, 0.231792344, 1.01046945, 0.00600069867, 0.0200179144, 103.560653)
    _SF11 = SellmeierDispersion(1.73759695, 0.313747346, 1.89878101, 0.013188707, 0.0623068142, 155.23629)
    _FUSED_SILICA = SellmeierDispersion(0.6961663, 0.4079426, 0.8974794, 0.004679148, 0.01351206, 97.9340025)
    _F2 = SellmeierDispersion(1.34533359, 0.209073176, 0.937357162, 0.00997743871, 0.0470450767, 111.886764)
    _SK16 = SellmeierDispersion(1.24355395, 0.310468846, 0.910046831, 0.007664188, 0.024941427, 98.4239845)
    _BALF14 = SellmeierDispersion(1.15657807, 0.26477019, 0.89973809, 0.007137837, 0.024479541, 102.50294)
    _SF2 = SellmeierDispersion(1.40307002, 0.231803944, 0.939056586, 0.010573176, 0.04948006, 112.4414)
    _SF4 = SellmeierDispersion(1.6377312, 0.2882898, 1.4883492, 0.0118836, 0.0573981, 139.7348)
    _SF5 = SellmeierDispersion(1.52481889, 0.25339644, 1.11603594, 0.011254756, 0.0541704, 117.1554)
    _SF10 = SellmeierDispersion(1.62153902, 0.256287842, 1.64447552, 0.01222415, 0.05957368, 147.468793)
    _NBAF10 = SellmeierDispersion(1.5851495, 0.143559385, 1.08521269, 0.00926681282, 0.0424489805, 105.613573)
    _NSF6 = SellmeierDispersion(1.77931763, 0.338149866, 2.08734474, 0.0133714182, 0.0617533621, 174.01759)
    # SCHOTT N-LAK22 (catalog 2023-01): nd=1.65113, Vd=55.89.
    _NLAK22 = SellmeierDispersion(1.14229781, 0.535138441, 1.04088385, 0.00585778594, 0.0198546147, 100.834017)
    _NLASF9 = SellmeierDispersion(2.00029547, 0.29892689, 1.80691843, 0.0121426, 0.0538736, 153.83365)
    _CAF2 = SellmeierDispersion(0.5675888, 0.4710914, 3.8484723, 0.00252643, 0.01007833, 1200.55597)
    _ZNSE = SellmeierDispersion(4.45813734, 0.467216334, 2.8956629, 0.04034468, 0.1531713, 2221.8223)
    _GERMANIUM = SellmeierDispersion(8.28156, 6.7288, 0.21307, 0.341942, 4479.62, 0.01)
    _SILICON = SellmeierDispersion(10.6684293, 0.0030434748, 1.54133408, 0.09091219, 1.2876602, 1218816.0)

    # Ohara molded aspheric and precision high-index glasses
    _SNPH1 = SellmeierDispersion(1.77708, 0.36647, 2.15572, 0.01385, 0.06522, 180.50)
    _SNPH11 = SellmeierDispersion(1.77708, 0.36647, 2.15572, 0.01385, 0.06522, 180.50)
    _SNPH2 = SellmeierDispersion(1.83226, 0.37053, 2.19045, 0.01452, 0.06811, 192.12)
    _SNPH3 = SellmeierDispersion(1.85412, 0.37521, 2.21540, 0.01490, 0.07010, 198.40)
    _SLAH64 = SellmeierDispersion(1.73358, 0.31295, 1.34119, 0.01315, 0.06234, 106.63)

    _NFK51A = SellmeierDispersion(0.971247817, 0.216901417, 0.904651666, 0.00472301997, 0.0153575612, 168.68133)
    _NSF1 = SellmeierDispersion(1.63581783, 0.25287707, 1.34138128, 0.011883584, 0.05432014, 138.83569)
    _NSF8 = SellmeierDispersion(1.51731692, 0.24522822, 1.10901923, 0.010839846, 0.051010375, 114.779774)
    _NSSK8 = SellmeierDispersion(1.42874136, 0.160107873, 0.992688005, 0.00936997577, 0.0427845341, 102.138656)
    _NKZFS11 = SellmeierDispersion(1.48835849, 0.222728442, 1.05607994, 0.0104996455, 0.0478673708, 104.148564)

    # Cauchy glasses & Media
    _B270 = CauchyDispersion(1.5230, 0.004, 0.0)
    _AIR = CauchyDispersion(1.000277, 0.0, 0.0)
    _VACUUM = CauchyDispersion(1.0, 0.0, 0.0)
    _MIRROR = CauchyDispersion(1.0, 0.0, 0.0)

    GLASSES: Dict[str, Any] = {
        "BK7": _BK7,
        "N-BK7": _BK7,
        "NBK7": _BK7,
        "N_BK7": _BK7,
        # SF11 / N-SF11
        "SF11": _SF11,
        "N-SF11": _SF11,
        "NSF11": _SF11,
        "N_SF11": _SF11,
        # FUSED SILICA
        "FUSED_SILICA": _FUSED_SILICA,
        "F_SILICA": _FUSED_SILICA,
        "FUSEDSILICA": _FUSED_SILICA,
        "FSILICA": _FUSED_SILICA,
        "SILICA": _FUSED_SILICA,
        # F2
        "F2": _F2,
        "N-F2": _F2,
        # SK16
        "SK16": _SK16,
        "N-SK16": _SK16,
        "NSK16": _SK16,
        "N_SK16": _SK16,
        # BALF14
        "BALF14": _BALF14,
        "N-BALF14": _BALF14,
        "NBALF14": _BALF14,
        "N_BALF14": _BALF14,
        # SF2, SF4, SF5, SF10
        "SF2": _SF2,
        "N-SF2": _SF2,
        "NSF2": _SF2,
        "SF4": _SF4,
        "N-SF4": _SF4,
        "NSF4": _SF4,
        "SF5": _SF5,
        "N-SF5": _SF5,
        "NSF5": _SF5,
        "N_SF5": _SF5,
        "SF10": _SF10,
        "N-SF10": _SF10,
        "NSF10": _SF10,
        # N-BAF10
        "N-BAF10": _NBAF10,
        "NBAF10": _NBAF10,
        "N_BAF10": _NBAF10,
        "BAF10": _NBAF10,
        # N-SF6 / N-SF6HT
        "N-SF6": _NSF6,
        "NSF6": _NSF6,
        "N_SF6": _NSF6,
        "SF6": _NSF6,
        "N-SF6HT": _NSF6,
        "NSF6HT": _NSF6,
        "N_SF6HT": _NSF6,
        # N-LAK22
        "N-LAK22": _NLAK22,
        "NLAK22": _NLAK22,
        "N_LAK22": _NLAK22,
        "LAK22": _NLAK22,
        # N-LASF9
        "N-LASF9": _NLASF9,
        "NLASF9": _NLASF9,
        "N_LASF9": _NLASF9,
        "LASF9": _NLASF9,
        # CaF2
        "CAF2": _CAF2,
        "CA_F2": _CAF2,
        "CA-F2": _CAF2,
        # ZnSe
        "ZNSE": _ZNSE,
        "ZN_SE": _ZNSE,
        "ZN-SE": _ZNSE,
        # Germanium / Silicon
        "GERMANIUM": _GERMANIUM,
        "GE": _GERMANIUM,
        "SILICON": _SILICON,
        "SI": _SILICON,
        # Ohara Glasses
        "S-NPH1": _SNPH1,
        "SNPH1": _SNPH1,
        "S_NPH1": _SNPH1,
        "S-NPH1_MOLD": _SNPH1,
        "SNPH1_MOLD": _SNPH1,
        "SNPH1MOLD": _SNPH1,
        "S_NPH1_MOLD": _SNPH1,
        "S-NPH11": _SNPH11,
        "SNPH11": _SNPH11,
        "S_NPH11": _SNPH11,
        "S-NPH11_MOLD": _SNPH11,
        "SNPH11_MOLD": _SNPH11,
        "SNPH11MOLD": _SNPH11,
        "S_NPH11_MOLD": _SNPH11,
        "S-NPH2": _SNPH2,
        "SNPH2": _SNPH2,
        "S-NPH3": _SNPH3,
        "SNPH3": _SNPH3,
        "S-LAH64": _SLAH64,
        "SLAH64": _SLAH64,
        # SCHOTT New Glasses
        "N-FK51A": _NFK51A,
        "NFK51A": _NFK51A,
        "N_FK51A": _NFK51A,
        "FK51A": _NFK51A,
        "N-SF1": _NSF1,
        "NSF1": _NSF1,
        "N_SF1": _NSF1,
        "SF1": _NSF1,
        "N-SF8": _NSF8,
        "NSF8": _NSF8,
        "N_SF8": _NSF8,
        "SF8": _NSF8,
        "N-SSK8": _NSSK8,
        "NSSK8": _NSSK8,
        "N_SSK8": _NSSK8,
        "SSK8": _NSSK8,
        "N-KZFS11": _NKZFS11,
        "NKZFS11": _NKZFS11,
        "N_KZFS11": _NKZFS11,
        "KZFS11": _NKZFS11,
        # B270
        "B270": _B270,
        "B-270": _B270,
        # Media
        "AIR": _AIR,
        "VACUUM": _VACUUM,
        "MIRROR": _MIRROR,
        "MIRR": _MIRROR,
        "REFL": _MIRROR,
    }

    @classmethod
    def get_glass(cls, name: str) -> Any:
        if not name:
            return cls.GLASSES["AIR"]
        if isinstance(name, (int, float)):
            return CauchyDispersion(float(name), 0.0, 0.0)
        clean_str = str(name).strip()
        clean_name = clean_str.upper().replace(" ", "_")
        if clean_name in cls.GLASSES:
            return cls.GLASSES[clean_name]

        # Try stripped variations (hyphens and underscores removed)
        alt_name = clean_name.replace("-", "").replace("_", "")
        if alt_name in cls.GLASSES:
            return cls.GLASSES[alt_name]

        # Strip prefixes like N-, N_, S-, S_, P-, P_
        for prefix in ["N-", "N_", "N", "S-", "S_", "S", "P-", "P_", "P"]:
            if clean_name.startswith(prefix) and len(clean_name) > len(prefix):
                stripped = clean_name[len(prefix):]
                if stripped in cls.GLASSES:
                    return cls.GLASSES[stripped]
                stripped_alt = stripped.replace("-", "").replace("_", "")
                if stripped_alt in cls.GLASSES:
                    return cls.GLASSES[stripped_alt]

        if alt_name.startswith("N") and len(alt_name) > 1:
            if alt_name[1:] in cls.GLASSES:
                return cls.GLASSES[alt_name[1:]]
        if alt_name.startswith("S") and len(alt_name) > 1:
            if alt_name[1:] in cls.GLASSES:
                return cls.GLASSES[alt_name[1:]]

        if clean_name.startswith("INDEX_"):
            try:
                val = float(clean_name[6:])
                return CauchyDispersion(val, 0.0, 0.0)
            except ValueError:
                pass
        elif clean_name.startswith("INDEX"):
            try:
                val = float(clean_name[5:])
                return CauchyDispersion(val, 0.0, 0.0)
            except ValueError:
                pass
        try:
            val = float(clean_str)
            return CauchyDispersion(val, 0.0, 0.0)
        except ValueError:
            pass

        raise ValueError(f"Unknown glass material: '{name}'")

    @classmethod
    def calculate_abbe_number(cls, glass_name: str) -> float:
        """Calculate Abbe V_d number: V_d = (n_d - 1) / (n_F - n_C)."""
        glass = cls.get_glass(glass_name)
        nd = glass.get_index(0.58756)  # Helium d line
        nf = glass.get_index(0.48613)  # Hydrogen F line
        nc = glass.get_index(0.65627)  # Hydrogen C line
        denom = nf - nc
        if abs(denom) < 1e-12:
            return float("inf")
        return float((nd - 1.0) / denom)

    @classmethod
    def calculate_partial_dispersion(
        cls, glass_name: str, wl_short: float = 0.48613, wl_mid: float = 0.58756, wl_long: float = 0.65627
    ) -> float:
        """
        Calculate relative partial dispersion P_{F,d} = (n_F - n_d) / (n_F - n_C).
        """
        glass = cls.get_glass(glass_name)
        nf = glass.get_index(wl_short)
        nd = glass.get_index(wl_mid)
        nc = glass.get_index(wl_long)
        denom = nf - nc
        if abs(denom) < 1e-12:
            return 0.0
        return float((nf - nd) / denom)

    @classmethod
    def get_glass_type(cls, glass_name: str) -> str:
        """Classify glass as Crown (V_d >= 50) or Flint (V_d < 50)."""
        vd = cls.calculate_abbe_number(glass_name)
        if math.isinf(vd) or math.isnan(vd):
            return "Medium"
        return "Crown" if vd >= 50.0 else "Flint"

    @classmethod
    def list_catalog_glasses(cls) -> List[Dict[str, Any]]:
        """
        Return structured metadata for all unique catalog optical glasses.
        """
        unique_names = [
            ("N-BK7", "SCHOTT", "Crown"),
            ("N-FK51A", "SCHOTT", "Crown"),
            ("FUSED_SILICA", "Corning/Heraeus", "Crown"),
            ("CAF2", "Crystal", "Crown"),
            ("N-SK16", "SCHOTT", "Crown"),
            ("N-BALF14", "SCHOTT", "Crown"),
            ("N-LAK22", "SCHOTT", "Crown"),
            ("B270", "SCHOTT", "Crown"),
            ("N-SSK8", "SCHOTT", "Flint"),
            ("N-BAF10", "SCHOTT", "Flint"),
            ("N-KZFS11", "SCHOTT", "Flint"),
            ("N-F2", "SCHOTT", "Flint"),
            ("N-SF2", "SCHOTT", "Flint"),
            ("N-SF5", "SCHOTT", "Flint"),
            ("N-SF8", "SCHOTT", "Flint"),
            ("N-SF1", "SCHOTT", "Flint"),
            ("N-SF4", "SCHOTT", "Flint"),
            ("N-SF10", "SCHOTT", "Flint"),
            ("N-SF11", "SCHOTT", "Flint"),
            ("N-SF6", "SCHOTT", "Flint"),
            ("N-LASF9", "SCHOTT", "Flint"),
            ("S-LAH64", "OHARA", "Flint"),
            ("S-NPH1", "OHARA", "Flint"),
            ("S-NPH2", "OHARA", "Flint"),
            ("ZNSE", "II-VI", "Infrared"),
            ("SILICON", "Semiconductor", "Infrared"),
            ("GERMANIUM", "Semiconductor", "Infrared"),
        ]
        results = []
        for name, mfr, expected_type in unique_names:
            try:
                glass = cls.get_glass(name)
                nd = float(glass.get_index(0.58756))
                nf = float(glass.get_index(0.48613))
                nc = float(glass.get_index(0.65627))
                vd = float(cls.calculate_abbe_number(name))
                pd = float(cls.calculate_partial_dispersion(name))
                gtype = "Crown" if vd >= 50.0 else "Flint"
                if "Infrared" in expected_type:
                    gtype = "Infrared"
                results.append({
                    "name": name,
                    "manufacturer": mfr,
                    "nd": nd,
                    "nf": nf,
                    "nc": nc,
                    "vd": vd,
                    "partial_dispersion": pd,
                    "type": gtype,
                })
            except Exception:
                continue
        return results


METALS: Dict[str, complex] = {
    "AL": complex(1.097, 6.79),
    "ALUMINUM": complex(1.097, 6.79),
    "AU": complex(0.18, 3.0),
    "GOLD": complex(0.18, 3.0),
    "AG": complex(0.05, 4.27),
    "SILVER": complex(0.05, 4.27),
    "CU": complex(0.25, 3.4),
    "COPPER": complex(0.25, 3.4),
}


def get_refractive_index(glass_name: str, wavelength_um: float = 0.5876) -> float:
    """
    Get refractive index n for any catalog material at a specific wavelength (um).
    """
    if glass_name is None:
        return 1.0
    if isinstance(glass_name, (int, float)):
        return float(glass_name)
    clean_str = str(glass_name).strip()
    if not clean_str or clean_str.upper() in ("AIR", "VACUUM", "MIRROR", "MIRR", "REFL"):
        return 1.000277 if clean_str.upper() == "AIR" else 1.0
    clean_name = clean_str.upper().replace(" ", "_")
    if clean_name in METALS:
        return float(METALS[clean_name].real)
    if clean_name.startswith("INDEX_"):
        try:
            return float(clean_name[6:])
        except ValueError:
            pass
    elif clean_name.startswith("INDEX"):
        try:
            return float(clean_name[5:])
        except ValueError:
            pass
    try:
        return float(clean_str)
    except ValueError:
        pass
    glass = GlassCatalog.get_glass(glass_name)
    return float(glass.get_index(wavelength_um))


def get_complex_refractive_index(
    glass_name: str,
    wavelength_um: float = 0.5876,
    extinction_coeff: float = 0.0
) -> complex:
    """
    Get complex refractive index n_tilde = n + ik for dielectric or metallic media.
    """
    if not glass_name:
        return complex(1.0, extinction_coeff)
    clean_name = glass_name.upper().strip().replace(" ", "_").replace("-", "")
    if clean_name in METALS:
        return METALS[clean_name]
    for key, val in METALS.items():
        if key in clean_name:
            return val
    try:
        n = get_refractive_index(glass_name, wavelength_um)
        return complex(n, extinction_coeff)
    except ValueError:
        return complex(1.0, extinction_coeff)
