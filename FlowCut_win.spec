# -*- mode: python ; coding: utf-8 -*-

"""
Windows 向け FlowCut GUI パッケージング用の PyInstaller レシピ（one-folder 想定）。

- エントリポイント: flowcut_gui_launcher.py
- アイコン: assets/FlowCut.ico （既存アイコンから生成した .ico を配置して利用する想定）
- 出力: dist/FlowCut/FlowCut.exe を含むフォルダ一式
"""

from PyInstaller.utils.hooks import collect_all
import os

# whisper パッケージ一式（コード + assets）をまとめて同梱する。
datas = [('config', 'config')]
binaries = []
hiddenimports = ['whisper', 'faster_whisper']

ROOT_DIR = os.path.abspath(globals().get("specpath") or os.getcwd())


def _find_ffmpeg_exe() -> str | None:
    for key in ("FLOWCUT_FFMPEG_EXE", "FFMPEG_EXE"):
        raw_path = os.environ.get(key)
        if not raw_path:
            continue
        candidate = os.path.expandvars(raw_path)
        if not os.path.isabs(candidate):
            candidate = os.path.join(ROOT_DIR, candidate)
        if os.path.exists(candidate):
            return candidate
    bundled = os.path.join(ROOT_DIR, "assets", "ffmpeg", "ffmpeg.exe")
    if os.path.exists(bundled):
        return bundled
    return None


ffmpeg_exe = _find_ffmpeg_exe()
if ffmpeg_exe:
    binaries.append((ffmpeg_exe, "ffmpeg_bin"))

# Optional: include license texts for redistributed ffmpeg binary.
ffmpeg_licenses_dir = os.path.join(ROOT_DIR, "assets", "ffmpeg", "licenses")
if os.path.isdir(ffmpeg_licenses_dir):
    datas.append((ffmpeg_licenses_dir, os.path.join("licenses", "ffmpeg")))

tmp_ret = collect_all('whisper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('faster_whisper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# さらに、実際の whisper パッケージの場所から mel_filters.npz を特定して、
# 実行時に参照される `_internal\\whisper\\assets\\mel_filters.npz` へ明示的に配置する。
try:
    import whisper  # type: ignore

    whisper_pkg_dir = os.path.dirname(whisper.__file__)
    whisper_mel_filters = os.path.join(whisper_pkg_dir, "assets", "mel_filters.npz")
    if os.path.exists(whisper_mel_filters):
        datas.append(
            (whisper_mel_filters, os.path.join("_internal", "whisper", "assets"))
        )
except Exception:
    # build 環境で whisper が import できない場合は、通常の collect_all のみで進める
    pass


a = Analysis(
    ['flowcut_gui_launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Prefer system-installed VC++ runtime DLLs over bundled ones.
# Some native stacks (torch / onnxruntime / ctranslate2) can become unstable when
# older VC runtime DLLs are bundled into the app directory.
_VC_RUNTIME_DLLS = {
    "msvcp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
}
try:
    a.binaries = [
        entry
        for entry in a.binaries
        if os.path.basename(entry[0]).lower() not in _VC_RUNTIME_DLLS
    ]
except Exception:
    # If filtering fails for some reason, continue with the original binaries list.
    pass
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
