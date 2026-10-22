"""
Multi-Wavelength Refractive Index Calculations (Sellmeier & Cauchy Formulas).
"""
import numpy as np

class SellmeierDispersion:
    def __init__(self, B1: float, B2: float, B3: float, C1: float, C2: float, C3: float):
        self.B = [float(B1), float(B2), float(B3)]
        self.C = [float(C1), float(C2), float(C3)]

    def get_index(self, wavelength_um: float) -> float:
        wl2 = float(wavelength_um)**2
        n2 = 1.0
        for b, c in zip(self.B, self.C):
            if abs(wl2 - c) < 1e-12 or (c > 0 and abs(wl2 - c) / c < 1e-6):
                raise ValueError(f"Wavelength {wavelength_um} um near Sellmeier resonance pole {c}.")
            n2 += (b * wl2) / (wl2 - c)
        if n2 < 1.0:
            raise ValueError(f"Calculated refractive index n^2 ({n2}) < 1.0")
        return float(np.sqrt(n2))

class CauchyDispersion:
    def __init__(self, A: float, B: float, C: float = 0.0):
        self.A = float(A)
        self.B = float(B)
        self.C = float(C)

    def get_index(self, wavelength_um: float) -> float:
        wl2 = float(wavelength_um)**2
        if wl2 == 0:
            raise ValueError("Wavelength cannot be zero.")
        return float(self.A + (self.B / wl2) + (self.C / (wl2**2)))

def get_refractive_index(glass_name: str, wavelength_um: float) -> float:
    from optics_app.utils.materials import GlassCatalog
    if not glass_name or glass_name.upper().strip() in ["AIR", "VACUUM", "MIRROR", "MIRR"]:
        return 1.000277 if (glass_name and glass_name.upper().strip() == "AIR") else 1.0
    glass = GlassCatalog.get_glass(glass_name)
    return glass.get_index(wavelength_um)
