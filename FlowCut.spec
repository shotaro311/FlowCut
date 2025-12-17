# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import glob

datas = [('config', 'config')]

# ffmpegバイナリを 'ffmpeg_bin' サブディレクトリに配置
# これにより、PyAVのffmpegライブラリとの競合を避ける
binaries = [('/opt/homebrew/bin/ffmpeg', 'ffmpeg_bin')]

# Homebrew版ffmpegライブラリもffmpeg_binに配置（ffmpegバイナリが参照する）
ffmpeg_libs = glob.glob('/opt/homebrew/Cellar/ffmpeg/*/lib/*.dylib')
for lib in ffmpeg_libs:
    binaries.append((lib, 'ffmpeg_bin'))

hiddenimports = ['mlx_whisper', 'mlx']

# mlx_whisper
tmp_ret = collect_all('mlx_whisper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# mlx
tmp_ret = collect_all('mlx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# 注意: collect_all('av') は使用しない
# PyAVのffmpegライブラリがHomebrew ffmpegと競合するため


a = Analysis(
    ['flowcut_gui_launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['av'],  # PyAVを除外（ffmpegライブラリ競合を避けるため）
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
    icon=['assets/FlowCut.icns'],
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
app = BUNDLE(
    coll,
    name='FlowCut.app',
    icon='assets/FlowCut.icns',
    bundle_identifier=None,
)
