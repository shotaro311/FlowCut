# -*- mode: python ; coding: utf-8 -*-

"""
Windows 向け FlowCut GUI パッケージング用の PyInstaller レシピ（one-folder 想定）。

- エントリポイント: flowcut_gui_launcher.py
- アイコン: assets/FlowCut.ico （既存アイコンから生成した .ico を配置して利用する想定）
- 出力: dist/FlowCut/FlowCut.exe を含むフォルダ一式
"""

hiddenimports = ['whisper']

a = Analysis(
    ['flowcut_gui_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('config', 'config')],
    hiddenimports=hiddenimports,
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
    name='FlowCut',
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
    # Windows では .ico を指定する想定（assets/FlowCut.iconset から作成した FlowCut.ico を配置）
    icon='assets/FlowCut.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FlowCut',
)
