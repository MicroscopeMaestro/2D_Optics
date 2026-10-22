import math
import numpy as np

def analyze_czerny_turner():
    print("=" * 70)
    print("CZERNY-TURNER & ALL-MIRROR OPTICAL EQUATIONS EVALUATION")
    print("=" * 70)

    # 1. Grating Equations (600 lines/mm)
    G = 600.0 # lines/mm
    d_um = 1000.0 / G # 1.66667 um
    print(f"1. Grating Pitch d = {d_um:.5f} um ({1000*d_um:.2f} nm)")

    # Analyze two candidate bands:
    # Band A: 785 - 880 nm (Raman 785nm)
    # Band B: 1100 - 1350 nm (NIR Fingerprint)
    bands = [
        ("785-880nm Raman", [0.785, 0.808, 0.830, 0.850, 0.880], 18.2),
        ("1100-1350nm NIR", [1.100, 1.225, 1.350], 12.0)
    ]

    for bname, wls, alpha_deg in bands:
        print(f"\n--- Band: {bname} (Incidence alpha = {alpha_deg:.2f} deg) ---")
        alpha_rad = math.radians(alpha_deg)
        sin_alpha = math.sin(alpha_rad)
        for wl in wls:
            sin_beta = wl / d_um - sin_alpha
            if abs(sin_beta) <= 1.0:
                beta_rad = math.asin(sin_beta)
                beta_deg = math.degrees(beta_rad)
                ang_disp = (1.0 / (d_um * math.cos(beta_rad))) # rad / um
                print(f"  wl = {wl:.3f} um: sin(beta) = {sin_beta:.4f} -> beta = {beta_deg:6.2f} deg | dbeta/dlambda = {ang_disp:.4f} rad/um ({math.degrees(ang_disp)*1e-3:.4f} deg/nm)")
            else:
                print(f"  wl = {wl:.3f} um: EVANESCENT (sin(beta) = {sin_beta:.4f})")

    # 2. Linear dispersion at DMD
    # W_disp = f2 * Delta_beta
    print("\n2. Linear Dispersion on DMD4500:")
    for f2 in [50.0, 75.0, 100.0]:
        for bname, wls, alpha_deg in bands:
            alpha_rad = math.radians(alpha_deg)
            sin_alpha = math.sin(alpha_rad)
            sin_b1 = wls[0] / d_um - sin_alpha
            sin_b2 = wls[-1] / d_um - sin_alpha
            b1 = math.asin(sin_b1)
            b2 = math.asin(sin_b2)
            d_beta = abs(b2 - b1)
            w_disp = f2 * math.tan(d_beta)
            print(f"  f2 = {f2:5.1f} mm | {bname:<16} | Delta_beta = {math.degrees(d_beta):5.2f} deg ({d_beta:.4f} rad) -> W_disp = {w_disp:.3f} mm")

    # 3. Coma compensation equation (Shafer / Fastie / Reader)
    print("\n3. Coma Cancellation Condition (Shafer):")
    # sin(theta1)/sin(theta2) = (R2/R1)^2 * (cos(theta2)/cos(theta1))^3 * (cos(alpha)/cos(beta_c))
    # Or in terms of f1, f2: sin(theta1)/sin(theta2) = (f1/f2)^3 * ...
    # Let's test standard mirror pairs:
    # Pair 1: f1 = 50 mm, f2 = 75 mm
    # Pair 2: f1 = 50 mm, f2 = 50 mm
    # Pair 3: f1 = 100 mm, f2 = 100 mm
    # Pair 4: f1 = 75 mm, f2 = 75 mm
    for (f1, f2) in [(50.0, 50.0), (50.0, 75.0), (75.0, 75.0), (100.0, 100.0)]:
        R1, R2 = 2*f1, 2*f2
        theta1_deg = 8.0
        theta1_rad = math.radians(theta1_deg)
        # For 785-880nm band: alpha=18.2 deg, beta_c = 10.70 deg
        alpha_rad = math.radians(18.2)
        beta_c_rad = math.asin(0.830 / d_um - math.sin(alpha_rad))
        anamorphic = math.cos(alpha_rad) / math.cos(beta_c_rad)
        
        # Target sin(theta2)
        # Using Reader equation: sin(theta2) = sin(theta1) * (R1/R2)^2 * (cos^3(theta1)/cos^3(theta2)) * (cos(beta_c)/cos(alpha))
        # Iterate to solve for theta2:
        th2 = theta1_rad
        for _ in range(20):
            th2 = math.asin(math.sin(theta1_rad) * ((R1/R2)**2) * ((math.cos(theta1_rad)/math.cos(th2))**3) * (1.0/anamorphic))
        print(f"  f1 = {f1:.0f} mm (R1={R1:.0f}), f2 = {f2:.0f} mm (R2={R2:.0f}) | theta1 = {theta1_deg:.1f} deg -> Optimal Coma theta2 = {math.degrees(th2):.2f} deg")

    # 4. Demagnifying Relay Systems
    print("\n4. Demagnifying Relay Configurations (5.0 mm DMD -> 0.8 mm Detector, m <= 0.16):")
    relay_pairs = [
        ("M3: CM254-150 (f=150), M4: CM254-025 (f=25)", 150.0, 25.0),
        ("M3: CM254-200 (f=200), M4: CM254-025 (f=25)", 200.0, 25.0),
        ("M3: CM254-100 (f=100), M4: CM254-015 (f=15)", 100.0, 15.0),
        ("M3: CM254-150 (f=150), M4: CM254-015 (f=15)", 150.0, 15.0),
        ("Single Mirror CM254-025 (f=25, s1=191.7, s2=28.8)", 25.0, None)
    ]
    for name, f3, f4 in relay_pairs:
        if f4 is not None:
            m = f4 / f3
            img_size = 5.0 * m
            img_radius = img_size / 2.0
            print(f"  {name:<50} | m = {m:.4f} ({1/m:.1f}:1) | Image Dia = {img_size:.3f} mm (r = {img_radius:.3f} mm <= 0.40 mm: {img_radius <= 0.40})")
        else:
            m = 0.15
            img_size = 5.0 * m
            img_radius = img_size / 2.0
            print(f"  {name:<50} | m = {m:.4f} ({1/m:.1f}:1) | Image Dia = {img_size:.3f} mm (r = {img_radius:.3f} mm <= 0.40 mm: {img_radius <= 0.40})")

analyze_czerny_turner()
