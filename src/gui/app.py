"""Tkinter-based minimal GUI to run the existing CLI pipeline."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.gui.controller import GuiController


class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Flow Cut GUI (Phase 1)")
        self.root.geometry("480x260")

        self.controller = GuiController(ui_dispatch=self._dispatch_to_ui)
        self.selected_file: Path | None = None

        self.status_var = tk.StringVar(value="待機中")
        self.file_var = tk.StringVar(value="音声ファイル: 未選択")
        self.output_var = tk.StringVar(value="")

        self._build_widgets()

    def _build_widgets(self) -> None:
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        file_label = ttk.Label(main_frame, textvariable=self.file_var, anchor=tk.W)
        file_label.pack(fill=tk.X, pady=(0, 12))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 12))

        select_button = ttk.Button(button_frame, text="ファイルを選択", command=self.select_file)
        select_button.pack(side=tk.LEFT)

        self.run_button = ttk.Button(button_frame, text="実行", command=self.run_pipeline)
        self.run_button.pack(side=tk.LEFT, padx=(8, 0))

        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(0, 12))

        status_row = ttk.Frame(main_frame)
        status_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(status_row, text="ステータス:").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.status_var).pack(side=tk.LEFT, padx=(4, 0))

        self.output_label = ttk.Label(main_frame, textvariable=self.output_var, foreground="#0b4f6c")
        self.output_label.pack(fill=tk.X)

    def select_file(self) -> None:
        path = filedialog.askopenfilename(
            title="音声ファイルを選択",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac"), ("All Files", "*.*")],
        )
        if path:
            self.selected_file = Path(path)
            self.file_var.set(f"音声ファイル: {self.selected_file}")

    def run_pipeline(self) -> None:
        if not self.selected_file:
            messagebox.showerror("エラー", "音声ファイルを選択してください。")
            return

        self._set_running_state(True)
        self.status_var.set("処理中…")
        self.output_var.set("")
        self.progress.start(10)

        self.controller.run_pipeline(
            self.selected_file,
            on_start=lambda: None,
            on_success=self._on_success,
            on_error=self._on_error,
            on_finish=self._on_finish,
        )

    def _on_success(self, output_paths: list[Path]) -> None:
        self.status_var.set("完了しました")
        if output_paths:
            self.output_var.set(f"出力: {output_paths[-1]}")

    def _on_error(self, exc: Exception) -> None:
        self.status_var.set("エラーが発生しました")
        messagebox.showerror("処理失敗", str(exc))

    def _on_finish(self) -> None:
        self.progress.stop()
        self._set_running_state(False)

    def _set_running_state(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.run_button.config(state=state)

    def _dispatch_to_ui(self, func) -> None:
        self.root.after(0, func)


def run_gui() -> None:
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


__all__ = ["run_gui", "MainWindow"]
