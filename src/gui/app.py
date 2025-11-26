"""Tkinter-based minimal GUI to run the existing CLI pipeline."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from src.gui.config import get_config
from src.gui.controller import GuiController
from src.gui.workflow_panel import WorkflowPanel


class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Flow Cut GUI (Phase 1) - 並列処理対応")
        self.root.geometry("640x800")  # 2パネル用にサイズを拡張
        
        # 設定マネージャー
        self.config = get_config()
        
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
            text="Flow Cut - 並列処理対応GUI", 
            font=("", 14, "bold")
        )
        title_label.pack(side=tk.LEFT)
        
        info_label = ttk.Label(
            header_frame, 
            text="最大2つのワークフローを同時実行できます",
            foreground="#666666"
        )
        info_label.pack(side=tk.RIGHT)
        
        # ワークフローパネル用のスクロール可能なフレーム
        self.workflow_frame = ttk.Frame(main_frame)
        self.workflow_frame.pack(fill=tk.BOTH, expand=True)
        
        # フッター
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=(16, 0))
        
        # 終了時の確認設定
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

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


def run_gui() -> None:
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


__all__ = ["run_gui", "MainWindow"]
