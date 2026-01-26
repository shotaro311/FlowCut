param(
    [string]$PythonExe = ".venv\\Scripts\\python.exe",
    [switch]$InstallDeps
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..") | Select-Object -ExpandProperty Path
$pythonPath = if (Test-Path $PythonExe) {
    $PythonExe
} else {
    Join-Path $repoRoot $PythonExe
}

if (-not (Test-Path $pythonPath)) {
    throw "Python executable not found: $pythonPath"
}

$ffmpegLocal = Join-Path $repoRoot "assets\\ffmpeg\\ffmpeg.exe"
if (-not $env:FLOWCUT_FFMPEG_EXE -and (Test-Path $ffmpegLocal)) {
    $env:FLOWCUT_FFMPEG_EXE = $ffmpegLocal
}

Push-Location $repoRoot
try {
    if ($InstallDeps) {
        & $pythonPath -m pip install -r requirements-dev.txt
    }

    & $pythonPath -m PyInstaller FlowCut_win.spec --clean --noconfirm

    $zipPath = Join-Path $repoRoot "dist\\FlowCut-win.zip"
    if (Test-Path $zipPath) {
        Remove-Item $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $repoRoot "dist\\FlowCut") -DestinationPath $zipPath
} finally {
    Pop-Location
}
