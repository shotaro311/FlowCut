# src/gui

CLI ベースの文字起こしパイプライン (`src.pipeline.poc.execute_poc_run`) を **Tkinter GUI** から呼び出すための薄いラッパーレイヤーです。GUI側でロジックを再実装せず、既存の PoC パイプラインをそのまま利用することを目的としています。

- メインエントリ: `run_gui()`（`python -m src.cli.main gui` から起動）
- メインウィンドウ実装: `MainWindow` (`src/gui/app.py`)
- パイプライン実行ラッパ: `GuiController` (`src/gui/controller.py`)

## 起動方法

```bash
python -m src.cli.main gui
```
