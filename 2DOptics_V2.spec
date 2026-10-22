# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['launch_v2.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('v2/catalogs/presets', 'v2/catalogs/presets'),
        ('assets', 'assets'),
    ],
    hiddenimports=[
        'v2.engine.illumination',
        'v2.gui.illumination_dock',
        'v2.engine.extended_merit',
        'v2.ai.tools',
        'v2.ai.nemotron_client',
        'matplotlib.backends.backend_qtagg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='2DOptics_V2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='assets/icon.ico',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='2DOptics_V2',
)
