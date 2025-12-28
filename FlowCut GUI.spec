# -*- mode: python ; coding: utf-8 -*-
import certifi

a = Analysis(
    ['flowcut_gui_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        # SSL証明書をバンドル（macOSでHTTPSリクエストを行うため必須）
        (certifi.where(), 'certifi'),
    ],
    hiddenimports=['certifi'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FlowCut GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/FlowCut.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FlowCut GUI',
)
app = BUNDLE(
    coll,
    name='FlowCut GUI.app',
    icon='assets/FlowCut.icns',
    bundle_identifier=None,
)
