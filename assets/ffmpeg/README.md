# ffmpeg (Windows)

Place a Windows build of `ffmpeg.exe` at:

```
assets/ffmpeg/ffmpeg.exe
```

`FlowCut_win.spec` bundles this into `ffmpeg_bin/` so the GUI can find it at
runtime. You can also override the path with an environment variable:

- `FLOWCUT_FFMPEG_EXE` (preferred)
- `FFMPEG_EXE` (fallback)

Recommendation for redistribution: use an LGPL build, not a GPL build.
