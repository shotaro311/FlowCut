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
        self.root.title("Flow Cut")
        self.root.geometry("640x800")  # 2パネル用にサイズを拡張
        
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
            text="Flow Cut", 
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
        openai_key = self.config.get_openai_api_key()
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        anthropic_key = self.config.get_anthropic_api_key()
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        reload_settings()

    def _reload_llm_profiles_in_workflows(self) -> None:
        """すべてのワークフローパネルでLLMプロファイルを再読み込みする。"""
        for panel in self.workflow_panels.values():
            panel.reload_llm_profiles()

    def _setup_workflows(self) -> None:
        """ワークフローパネルをセットアップする。"""
        # ワークフローパネル
        self.workflow_panels = {}
        panel = WorkflowPanel(self.workflow_frame, "1", self.controller, self.root)
        panel.pack(fill=tk.X, pady=(0, 8))
        self.workflow_panels["1"] = panel
    
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
        """APIキーの設定ダイアログを開く。"""
        dialog = tk.Toplevel(self.root)
        dialog.title("API設定")
        dialog.geometry("450x200")
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)

        # Google APIキー
        ttk.Label(frame, text="Google APIキー:").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        google_var = tk.StringVar(value=self.config.get_google_api_key() or "")
        google_entry = ttk.Entry(frame, textvariable=google_var, show="*", width=40)
        google_entry.grid(row=0, column=1, sticky=tk.EW, padx=(8, 0), pady=(0, 8))

        # OpenAI APIキー
        ttk.Label(frame, text="OpenAI APIキー:").grid(row=1, column=0, sticky=tk.W, pady=(0, 8))
        openai_var = tk.StringVar(value=self.config.get_openai_api_key() or "")
        openai_entry = ttk.Entry(frame, textvariable=openai_var, show="*", width=40)
        openai_entry.grid(row=1, column=1, sticky=tk.EW, padx=(8, 0), pady=(0, 8))

        # Anthropic APIキー
        ttk.Label(frame, text="Anthropic APIキー:").grid(row=2, column=0, sticky=tk.W, pady=(0, 8))
        anthropic_var = tk.StringVar(value=self.config.get_anthropic_api_key() or "")
        anthropic_entry = ttk.Entry(frame, textvariable=anthropic_var, show="*", width=40)
        anthropic_entry.grid(row=2, column=1, sticky=tk.EW, padx=(8, 0), pady=(0, 8))

        # ボタンフレーム
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(12, 0), sticky=tk.E)

        def on_save() -> None:
            google_key = google_var.get().strip()
            openai_key = openai_var.get().strip()
            anthropic_key = anthropic_var.get().strip()

            # 少なくとも1つのAPIキーが設定されていることを確認
            if not google_key and not openai_key and not anthropic_key:
                messagebox.showerror("エラー", "少なくとも1つのAPIキーを入力してください。")
                return

            # Google
            if google_key:
                self.config.set_google_api_key(google_key)
                os.environ["GOOGLE_API_KEY"] = google_key
            # OpenAI
            if openai_key:
                self.config.set_openai_api_key(openai_key)
                os.environ["OPENAI_API_KEY"] = openai_key
            # Anthropic
            if anthropic_key:
                self.config.set_anthropic_api_key(anthropic_key)
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key

            reload_settings()
            self._reload_llm_profiles_in_workflows()
            messagebox.showinfo("情報", "APIキーを保存しました。")
            dialog.destroy()

        def on_cancel() -> None:
            dialog.destroy()

        save_button = ttk.Button(button_frame, text="保存", command=on_save)
        save_button.pack(side=tk.RIGHT)

        cancel_button = ttk.Button(button_frame, text="キャンセル", command=on_cancel)
        cancel_button.pack(side=tk.RIGHT, padx=(0, 8))

        google_entry.focus_set()


def run_gui() -> None:
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


__all__ = ["run_gui", "MainWindow"]
