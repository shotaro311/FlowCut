"""Tkinter-based minimal GUI to run the existing CLI pipeline."""
from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from src.config import reload_settings
from src.gui.config import get_config
from src.gui.controller import GuiController
from src.gui.workflow_panel import WorkflowPanel


class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        # アプリタイトルはシンプルに「FlowCut」とする
        self.root.title("FlowCut")
        self.root.geometry("640x800")  # ウィンドウサイズ
        
        # 設定マネージャー
        self.config = get_config()
        self._apply_api_settings_from_config()
        
        # コントローラー
        self.controller = GuiController(ui_dispatch=self._dispatch_to_ui)
        
        # ワークフローパネルの管理
        self.workflow_panels: dict[str, WorkflowPanel] = {}
        
        self._build_widgets()
        self._setup_workflows()

    def _build_widgets(self) -> None:
        """メインウィジェットを構築する。"""
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ヘッダー
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 16))
        
        title_label = ttk.Label(
            header_frame, 
            text="FlowCut", 
            font=("", 14, "bold")
        )
        title_label.pack(side=tk.LEFT)

        api_button = ttk.Button(
            header_frame,
            text="API設定",
            command=self._open_api_settings_dialog,
        )
        api_button.pack(side=tk.RIGHT)
        
        # ワークフローパネル用のスクロール可能なフレーム
        self.workflow_frame = ttk.Frame(main_frame)
        self.workflow_frame.pack(fill=tk.BOTH, expand=True)
        
        # フッター
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=(16, 0))
        
        # 終了時の確認設定
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _apply_api_settings_from_config(self) -> None:
        """設定ファイルからAPIキーなどを環境変数に反映する。"""
        google_key = self.config.get_google_api_key()
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key
        reload_settings()

    def _reload_llm_profiles_in_workflows(self) -> None:
        """すべてのワークフローパネルでLLMプロファイルを再読み込みする。"""
        for panel in self.workflow_panels.values():
            panel.reload_llm_profiles()

    def _setup_workflows(self) -> None:
        """ワークフローパネルをセットアップする。"""
        # ワークフロー1
        panel1 = WorkflowPanel(self.workflow_frame, "1", self.controller, self.root)
        panel1.pack(fill=tk.X, pady=(0, 8))
        self.workflow_panels["1"] = panel1
        
        # ワークフロー2
        panel2 = WorkflowPanel(self.workflow_frame, "2", self.controller, self.root)
        panel2.pack(fill=tk.X, pady=(0, 8))
        self.workflow_panels["2"] = panel2
    
    def _dispatch_to_ui(self, func) -> None:
        """UIスレッドで関数を実行する。"""
        self.root.after(0, func)
    
    def _on_closing(self) -> None:
        """ウィンドウクローズ時の処理。"""
        # 実行中のワークフローがある場合は確認
        running_workflows = [
            wf_id for wf_id, panel in self.workflow_panels.items() 
            if panel.is_running
        ]
        
        if running_workflows:
            result = messagebox.askyesno(
                "確認", 
                f"ワークフロー {', '.join(running_workflows)} が実行中です。\n終了してもよろしいですか？"
            )
            if not result:
                return
        
        self.root.destroy()

    def _open_api_settings_dialog(self) -> None:
        """Google APIキーの設定ダイアログを開く。"""
        dialog = tk.Toplevel(self.root)
        dialog.title("API設定")
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Google APIキー:").grid(row=0, column=0, sticky=tk.W)

        current_key = self.config.get_google_api_key() or ""
        var = tk.StringVar(value=current_key)
        entry = ttk.Entry(frame, textvariable=var, show="*")
        entry.grid(row=0, column=1, sticky=tk.EW, padx=(8, 0))
        frame.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(12, 0), sticky=tk.E)

        def on_save() -> None:
            key = var.get().strip()
            if not key:
                messagebox.showerror("エラー", "Google APIキーを入力してください。")
                return
            self.config.set_google_api_key(key)
            os.environ["GOOGLE_API_KEY"] = key
            reload_settings()
            self._reload_llm_profiles_in_workflows()
            messagebox.showinfo("情報", "Google APIキーを保存しました。")
            dialog.destroy()

        def on_cancel() -> None:
            dialog.destroy()

        save_button = ttk.Button(button_frame, text="保存", command=on_save)
        save_button.pack(side=tk.RIGHT)

        cancel_button = ttk.Button(button_frame, text="キャンセル", command=on_cancel)
        cancel_button.pack(side=tk.RIGHT, padx=(0, 8))

        entry.focus_set()


def run_gui() -> None:
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


__all__ = ["run_gui", "MainWindow"]
