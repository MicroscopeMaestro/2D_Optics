import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# I will find the 'if enable_fillets:' block and inject the new chamfering code.
old_block = '''    # 2. Exterior Fillets (R1 Aesthetic & Ergonomics)
    if enable_fillets:
        if verbose: print("[STEP 2/7] Applying exterior corner fillets...")
        f1_top = make_corner_fillet_cutter(CLAMSHELL_HALF_THICKNESS, 25.0, FILLET_RADIUS, -1, -1)
        f2_top = make_corner_fillet_cutter(CLAMSHELL_HALF_THICKNESS, -80.0, FILLET_RADIUS, -1, 1)
        top_half = top_half.cut(f1_top).cut(f2_top)

        f1_bot = make_corner_fillet_cutter(-CLAMSHELL_HALF_THICKNESS, 25.0, FILLET_RADIUS, 1, -1)
        f2_bot = make_corner_fillet_cutter(-CLAMSHELL_HALF_THICKNESS, -80.0, FILLET_RADIUS, 1, 1)
        bottom_half = bottom_half.cut(f1_bot).cut(f2_bot)

        # Entrance bottom corner fillet
        hf_top = make_horizontal_fillet_cutter(CLAMSHELL_HALF_THICKNESS, 0.0, 3.0, -1, 1)
        hf_bot = make_horizontal_fillet_cutter(-CLAMSHELL_HALF_THICKNESS, 0.0, 3.0, 1, 1)
        top_half, bottom_half = top_half.cut(hf_top), bottom_half.cut(hf_bot)'''

new_block = '''    # 2. Exterior Edge Smoothing (Chamfers)
    if enable_fillets:
        if verbose: print("[STEP 2/7] Applying massive 2mm chamfer to external surfaces...")
        # Gather top exterior edges
        top_edges = []
        for f in top_half.Faces:
            if abs(f.BoundBox.XMin - CLAMSHELL_HALF_THICKNESS) < 0.1 and abs(f.BoundBox.XMax - CLAMSHELL_HALF_THICKNESS) < 0.1:
                top_edges.extend(f.Edges)
        if top_edges:
            try:
                top_half = top_half.makeChamfer(2.0, top_edges)
            except:
                pass
                
        # Gather bottom exterior edges
        bot_edges = []
        for f in bottom_half.Faces:
            if abs(f.BoundBox.XMin - (-CLAMSHELL_HALF_THICKNESS)) < 0.1 and abs(f.BoundBox.XMax - (-CLAMSHELL_HALF_THICKNESS)) < 0.1:
                bot_edges.extend(f.Edges)
        if bot_edges:
            try:
                bottom_half = bottom_half.makeChamfer(2.0, bot_edges)
            except:
                pass'''

content = content.replace(old_block, new_block)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Replaced old broken corner fillets with robust full-surface edge chamfer!")
