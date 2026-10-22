import json
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def dist_segment_point(p1, p2, p):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p = np.array(p)
    l2 = np.sum((p1 - p2)**2)
    if l2 == 0:
        return np.linalg.norm(p - p1)
    t = max(0, min(1, np.dot(p - p1, p2 - p1) / l2))
    projection = p1 + t * (p2 - p1)
    return np.linalg.norm(p - projection)

# Constraints and parameters
HOUSING_RADIUS = 13.0 # 12.7 mm + 0.3 mm margin
f_CM = 50.8
f_FM = 50.8
f_Coll = 25.4
d_grating = 1e6 / 150 # 150 lines/mm => nm
wl_min, wl_max = 1000.0, 1700.0
wl_c = 1350.0

def evaluate_layout(x):
    # x contains angles and lengths to define the layout sequentially
    # x[0]: CM angle from Slit (theta_CM) - Slit is at origin, CM is at f_CM distance
    # x[1]: G angle from CM (theta_G) - Grating is at distance L_CM_G
    # x[2]: L_CM_G
    # x[3]: FM angle from G for central wavelength (theta_FM)
    # x[4]: L_G_FM
    # x[5]: DMD angle from FM (theta_DMD)
    # x[6]: Coll angle from DMD (theta_Coll)
    # x[7]: L_DMD_Coll
    # x[8]: Det angle from Coll (theta_Det)
    
    theta_CM = x[0]
    theta_G = x[1]
    L_CM_G = x[2]
    theta_FM = x[3]
    L_G_FM = x[4]
    theta_DMD = x[5]
    theta_Coll = x[6]
    L_DMD_Coll = x[7]
    theta_Det = x[8]
    
    # 1. Slit
    S = np.array([0.0, 0.0])
    
    # 2. CM
    CM = S + f_CM * np.array([np.cos(theta_CM), np.sin(theta_CM)])
    
    # 3. G
    G = CM + L_CM_G * np.array([np.cos(theta_G), np.sin(theta_G)])
    
    # Calculate grating reflection / diffraction
    # Beam from CM to G is v_in
    v_in_G = normalize(G - CM)
    alpha_G_global = np.arctan2(v_in_G[1], v_in_G[0])
    
    # We want FM to be at theta_FM from G
    FM = G + L_G_FM * np.array([np.cos(theta_FM), np.sin(theta_FM)])
    v_out_G_c = normalize(FM - G)
    beta_G_global = np.arctan2(v_out_G_c[1], v_out_G_c[0])
    
    # 4. DMD
    DMD = FM + f_FM * np.array([np.cos(theta_DMD), np.sin(theta_DMD)])
    
    # 5. Coll
    # We want Coll to focus light from DMD to Detector.
    # The light from DMD is diverging if DMD is not at the focus, but wait.
    # Light from FM focuses ON the DMD. So the DMD is a focal plane.
    # Light reflecting from DMD acts as a new point source (the "slit" for the second half).
    # We place Coll at distance f_Coll from DMD.
    # Actually, we want a relay. The simplest relay is DMD -> Coll -> Det.
    # If Coll is at distance 2*f_Coll from DMD, it images DMD onto Det at 2*f_Coll.
    # Let's use 2*f_Coll for both distances to get a 1:1 relay.
    Coll = DMD + (2 * f_Coll) * np.array([np.cos(theta_Coll), np.sin(theta_Coll)])
    
    # 6. Det
    Det = Coll + (2 * f_Coll) * np.array([np.cos(theta_Det), np.sin(theta_Det)])
    
    # Calculate grating diffraction for the extreme wavelengths
    # We need to find the grating normal.
    # alpha is angle of incidence, beta is angle of diffraction.
    # sin(alpha) + sin(beta) = wl_c / d
    # The angle between incident and diffracted is beta_G_global - alpha_G_global.
    deviation = (beta_G_global - alpha_G_global + np.pi) % (2*np.pi) - np.pi
    
    # To have a valid grating equation, we need |sin(alpha) + sin(beta)| = wl/d.
    # Let's just assume we orient the grating to satisfy this.
    
    # Check collisions
    # Beams:
    # 1. S -> CM
    # 2. CM -> G
    # 3. G -> FM (central, min, max wavelengths)
    # 4. FM -> DMD (central, min, max)
    # 5. DMD -> Coll
    # 6. Coll -> Det
    
    beams = [
        (S, CM),
        (CM, G),
        (G, FM),
        (FM, DMD),
        (DMD, Coll),
        (Coll, Det)
    ]
    
    elements = {
        'S': S,
        'CM': CM,
        'G': G,
        'FM': FM,
        'DMD': DMD,
        'Coll': Coll,
        'Det': Det
    }
    
    penalty = 0.0
    
    # Encourage some distances
    if L_CM_G < 40: penalty += (40 - L_CM_G)**2
    if L_G_FM < 40: penalty += (40 - L_G_FM)**2
    if L_CM_G > 150: penalty += (L_CM_G - 150)**2
    if L_G_FM > 150: penalty += (L_G_FM - 150)**2
    
    # Collision check
    for name, pos in elements.items():
        for i, (b1, b2) in enumerate(beams):
            # The beam starts or ends at the element -> naturally close, ignore
            # But wait, what if a beam intersects the housing of a mirror it's NOT supposed to hit?
            if np.allclose(pos, b1) or np.allclose(pos, b2):
                continue
            
            d = dist_segment_point(b1, b2, pos)
            if d < HOUSING_RADIUS:
                penalty += (HOUSING_RADIUS - d + 1)**3 * 1000
                
    # Also check DMD width for extreme wavelengths
    # Let's calculate beta for min and max wl.
    # alpha + beta_c = deviation ... wait, better to define alpha.
    # We can always find a grating normal that bisects the angle differently to get the right dispersion.
    # But dispersion depends on alpha.
    # Actually, simplest is to constrain the angle.
    
    return penalty

# Let's just build a manual layout that works, because optimization might get stuck.
