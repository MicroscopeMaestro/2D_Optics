import json
import numpy as np
import matplotlib.pyplot as plt

def rot_mat(deg):
    rad = np.radians(deg)
    return np.array([[np.cos(rad), -np.sin(rad)],
                     [np.sin(rad),  np.cos(rad)]])

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

def calculate_layout():
    # Parameters
    f_CM = 50.8
    f_FM = 40.0
    f_Coll = 25.4
    
    # We want demagnification from DMD to Detector to be at least 4.41 / 0.8 = 5.51
    # 1/s_o + 1/s_i = 1/f_Coll
    # M = s_o / s_i = 5.6
    M = 5.6
    s_o = (M + 1) * f_Coll
    s_i = (1 + 1/M) * f_Coll
    
    fold_CM = 25.0
    L_CM_G = 75.0
    
    S = np.array([0.0, 0.0])
    CM = np.array([f_CM, 0.0])
    
    v_CM_G = rot_mat(180 - fold_CM).dot(np.array([1.0, 0.0]))
    G = CM + L_CM_G * v_CM_G
    
    alpha = 30.0
    d = 1000000.0 / 150.0 
    lambda_c = 1350.0
    
    sin_beta_c = lambda_c / d - np.sin(np.radians(alpha))
    beta_c = np.degrees(np.arcsin(sin_beta_c))
    deviation_angle = alpha - beta_c
    
    v_G_FM = rot_mat(deviation_angle).dot(-v_CM_G)
    
    L_G_FM = 75.0
    FM = G + L_G_FM * v_G_FM
    
    fold_FM = -30.0
    v_FM_DMD = rot_mat(180 - fold_FM).dot(-v_G_FM)
    DMD = FM + f_FM * v_FM_DMD
    
    L_DMD_Coll = s_o
    fold_DMD = 35.0
    v_DMD_Coll = rot_mat(180 - fold_DMD).dot(-v_FM_DMD)
    Coll = DMD + L_DMD_Coll * v_DMD_Coll
    
    L_Coll_Det = s_i
    fold_Coll = -40.0
    v_Coll_Det = rot_mat(180 - fold_Coll).dot(-v_DMD_Coll)
    Det = Coll + L_Coll_Det * v_Coll_Det
    
    elements = {
        'Slit': S,
        'CollimatingMirror': CM,
        'Grating': G,
        'FocusingMirror': FM,
        'DMD': DMD,
        'CollectorMirror': Coll,
        'Detector': Det
    }
    
    lambda_min, lambda_max = 1000.0, 1700.0
    sin_b_min = lambda_min / d - np.sin(np.radians(alpha))
    b_min = np.degrees(np.arcsin(sin_b_min))
    sin_b_max = lambda_max / d - np.sin(np.radians(alpha))
    b_max = np.degrees(np.arcsin(sin_b_max))
    
    angle_min = b_min - beta_c
    angle_max = b_max - beta_c
    
    spread_min = f_FM * np.tan(np.radians(angle_min))
    spread_max = f_FM * np.tan(np.radians(angle_max))
    total_spread = abs(spread_max - spread_min)
    
    print(f"Spread on DMD: {total_spread:.2f} mm")
    print(f"Spread on Detector: {total_spread / M:.2f} mm")
    
    HOUSING_RADIUS = 12.7
    MARGIN = 0.3
    SAFE_RADIUS = HOUSING_RADIUS + MARGIN
    
    beams = [
        ('Slit->CM', S, CM),
        ('CM->G', CM, G),
        ('G->FM', G, FM),
        ('FM->DMD', FM, DMD),
        ('DMD->Coll', DMD, Coll),
        ('Coll->Det', Coll, Det)
    ]
    
    v_G_FM_min = rot_mat(deviation_angle + angle_min).dot(-v_CM_G)
    v_G_FM_max = rot_mat(deviation_angle + angle_max).dot(-v_CM_G)
    
    FM_min = G + L_G_FM * v_G_FM_min
    FM_max = G + L_G_FM * v_G_FM_max
    
    DMD_min = DMD + np.array([-v_FM_DMD[1], v_FM_DMD[0]]) * spread_min
    DMD_max = DMD + np.array([-v_FM_DMD[1], v_FM_DMD[0]]) * spread_max
    
    beams.extend([
        ('G->FM_min', G, FM_min),
        ('G->FM_max', G, FM_max),
        ('FM_min->DMD_min', FM_min, DMD_min),
        ('FM_max->DMD_max', FM_max, DMD_max)
    ])
    
    collision_detected = False
    for name, pos in elements.items():
        for b_name, p1, p2 in beams:
            if name == 'FocusingMirror' and ('FM' in b_name):
                continue
            if name == 'Grating' and ('G' in b_name):
                continue
            if name == 'DMD' and ('DMD' in b_name):
                continue
            if name == 'CollectorMirror' and ('Coll' in b_name):
                continue
            if name == 'Detector' and ('Det' in b_name):
                continue
                
            if np.allclose(pos, p1) or np.allclose(pos, p2):
                continue
                
            d_dist = dist_segment_point(p1, p2, pos)
            if d_dist < SAFE_RADIUS:
                print(f"COLLISION: Beam {b_name} hits {name} (distance {d_dist:.2f} < {SAFE_RADIUS})")
                collision_detected = True

    if not collision_detected:
        print("NO COLLISIONS DETECTED")
        out_elements = {k: v.tolist() for k, v in elements.items()}
        with open('spectrometer_nir_mirror.json', 'w') as f:
            json.dump(out_elements, f, indent=4)
        print("Coordinates:")
        for k, v in out_elements.items():
            print(f"{k}: {v[0]:.2f}, {v[1]:.2f}")
    else:
        print("WARNING: Collisions were detected. Layout not saved.")
        
if __name__ == '__main__':
    calculate_layout()
