"""個別のワークフロー処理パネルを管理するウィジェット。"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, Any, List

from src.llm.profiles import list_models_by_provider
from src.gui.config import get_config


def _get_all_models() -> List[str]:
    """全プロバイダーのモデルを統合したリストを返す。"""
    models_by_provider = list_models_by_provider()
    all_models = []
    for models in models_by_provider.values():
        all_models.extend(models)
    # 重複を除去してソート
    return sorted(set(all_models))


def _get_provider_for_model(model: str) -> str:
    """モデル名からプロバイダーを判定する。"""
    model_lower = model.lower()
    if model_lower.startswith("gemini"):
        return "google"
    elif model_lower.startswith("gpt"):
        return "openai"
    elif model_lower.startswith("claude"):
        return "anthropic"
    # デフォルトはgoogle
    return "google"


class WorkflowPanel(ttk.Frame):
    """個別のワークフロー処理を管理するパネル。"""

    def __init__(self, parent: tk.Widget, workflow_id: str, controller, root: tk.Tk) -> None:
        super().__init__(parent, padding=8)
        self.workflow_id = workflow_id
        self.controller = controller
        self.root = root
        
        # 状態変数
        self.selected_file: Path | None = None
        self.output_dir: Path | None = None
        self.is_running = False
        
        # 設定マネージャー（共有）
        self.config = get_config()
        
        # UI変数
        self.file_var = tk.StringVar(value="音声ファイル: 未選択")
        self.output_dir_var = tk.StringVar(value="保存先フォルダ: output/ （デフォルト）")
        self.output_var = tk.StringVar(value="")
        self.metrics_var = tk.StringVar(value="")
        self.phase_var = tk.StringVar(value="")
        self.base_phase_var = tk.StringVar(value="")
        self.rolling_dots = 0
        self.rolling_timer_id = None
        
        # Pass1-4モデル変数
        self.pass1_model_var = tk.StringVar()
        self.pass2_model_var = tk.StringVar()
        self.pass3_model_var = tk.StringVar()
        self.pass4_model_var = tk.StringVar()
        self.start_delay_var = tk.StringVar(value="0.0")
        
        # Pass5関連変数
        self.pass5_enabled_var = tk.BooleanVar(value=False)
        self.pass5_max_chars_var = tk.StringVar(value="17")
        self.pass5_model_var = tk.StringVar()
        
        # 全モデルリスト
        self._all_models = _get_all_models()
        self._pass_model_combos: list[ttk.Combobox] = []
        
        # フレームのタイトル
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(title_frame, text=f"ワークフロー {workflow_id}", font=("", 10, "bold")).pack(side=tk.LEFT)
        
        self._build_widgets()
        self._load_initial_settings()

    def _build_widgets(self) -> None:
        """ウィジェットを構築する。"""
        # ファイル選択
        file_frame = ttk.Frame(self)
        file_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(file_frame, textvariable=self.file_var, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.run_button = ttk.Button(
            file_frame,
            text="▶ 実行",
            command=self.run_pipeline,
            style="Run.TButton",
            width=8,
        )
        self.run_button.pack(side=tk.RIGHT, padx=(8, 0))
        
        select_button = ttk.Button(file_frame, text="ファイルを選択", command=self.select_file)
        select_button.pack(side=tk.RIGHT)
        
        # 出力先選択
        output_frame = ttk.Frame(self)
        output_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(output_frame, textvariable=self.output_dir_var, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        select_output_button = ttk.Button(output_frame, text="保存先を変更", command=self.select_output_dir)
        select_output_button.pack(side=tk.RIGHT)
        
        # LLMオプション
        options_frame = ttk.LabelFrame(self, text="LLMオプション")
        options_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Pass1-4のモデル選択（常に表示、全モデルから選択可能）
        for pass_name, var in [
            ("Pass1:", self.pass1_model_var),
            ("Pass2:", self.pass2_model_var),
            ("Pass3:", self.pass3_model_var),
            ("Pass4:", self.pass4_model_var),
        ]:
            row = ttk.Frame(options_frame)
            row.pack(fill=tk.X, pady=(2, 2))
            ttk.Label(row, text=pass_name, width=8).pack(side=tk.LEFT)
            combo = ttk.Combobox(
                row,
                textvariable=var,
                values=self._all_models,
                state="readonly",
                width=30,
            )
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            combo.bind("<<ComboboxSelected>>", self._on_model_changed)
            self._pass_model_combos.append(combo)
        
        # 開始遅延
        delay_row = ttk.Frame(options_frame)
        delay_row.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(delay_row, text="開始遅延(秒):").pack(side=tk.LEFT)
        start_delay_entry = ttk.Entry(
            delay_row,
            textvariable=self.start_delay_var,
            width=6,
        )
        start_delay_entry.pack(side=tk.LEFT, padx=(4, 0))
        start_delay_entry.bind("<FocusOut>", self._on_start_delay_changed)
        ttk.Label(delay_row, text="例: 0.2", foreground="#888888").pack(side=tk.LEFT, padx=(4, 0))
        
        # Pass5設定
        pass5_frame = ttk.Frame(options_frame)
        pass5_frame.pack(fill=tk.X, pady=(8, 2))
        
        pass5_check = ttk.Checkbutton(
            pass5_frame,
            text="Pass5: 長行改行",
            variable=self.pass5_enabled_var,
            command=self._on_pass5_enabled_changed,
        )
        pass5_check.pack(side=tk.LEFT)
        
        ttk.Label(pass5_frame, text="モデル:").pack(side=tk.LEFT, padx=(16, 0))
        pass5_model_combo = ttk.Combobox(
            pass5_frame,
            textvariable=self.pass5_model_var,
            values=self._all_models,
            state="readonly",
            width=20,
        )
        pass5_model_combo.pack(side=tk.LEFT, padx=(4, 0))
        pass5_model_combo.bind("<<ComboboxSelected>>", self._on_pass5_model_changed)
        
        # Pass5文字数
        pass5_chars_row = ttk.Frame(options_frame)
        pass5_chars_row.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(pass5_chars_row, text="        ").pack(side=tk.LEFT)  # インデント
        ttk.Label(pass5_chars_row, text="文字数:").pack(side=tk.LEFT)
        pass5_chars_entry = ttk.Entry(
            pass5_chars_row,
            textvariable=self.pass5_max_chars_var,
            width=4,
        )
        pass5_chars_entry.pack(side=tk.LEFT, padx=(4, 0))
        pass5_chars_entry.bind("<FocusOut>", self._on_pass5_chars_changed)
        ttk.Label(pass5_chars_row, text="文字超過時に改行", foreground="#888888").pack(side=tk.LEFT, padx=(4, 0))
        
        # プログレスバー
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(4, 4))
        
        # ステータス表示
        status_row = ttk.Frame(self)
        status_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(status_row, text="ステータス:").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.phase_var).pack(side=tk.LEFT, padx=(4, 0))
        
        # 出力とメトリクス表示
        self.output_label = ttk.Label(self, textvariable=self.output_var, foreground="#0b4f6c")
        self.output_label.pack(fill=tk.X, pady=(0, 2))
        
        self.metrics_label = ttk.Label(self, textvariable=self.metrics_var, foreground="#555555")
        self.metrics_label.pack(fill=tk.X)

    def _load_initial_settings(self) -> None:
        """初期設定を読み込む。"""
        # Pass1-4モデル設定を読み込み
        default_models = {
            "pass1": "gemini-2.5-pro",
            "pass2": "gemini-2.5-pro", 
            "pass3": "gemini-2.5-flash",
            "pass4": "gemini-2.5-flash",
            "pass5": "claude-sonnet-4-5",
        }
        
        pass1 = self.config.get_pass_model("pass1", default_models["pass1"])
        pass2 = self.config.get_pass_model("pass2", default_models["pass2"])
        pass3 = self.config.get_pass_model("pass3", default_models["pass3"])
        pass4 = self.config.get_pass_model("pass4", default_models["pass4"])
        pass5 = self.config.get_pass_model("pass5", default_models["pass5"])
        
        self.pass1_model_var.set(pass1 if pass1 in self._all_models else default_models["pass1"])
        self.pass2_model_var.set(pass2 if pass2 in self._all_models else default_models["pass2"])
        self.pass3_model_var.set(pass3 if pass3 in self._all_models else default_models["pass3"])
        self.pass4_model_var.set(pass4 if pass4 in self._all_models else default_models["pass4"])
        self.pass5_model_var.set(pass5 if pass5 in self._all_models else default_models["pass5"])
        
        # 開始遅延
        delay = self.config.get_start_delay()
        self.start_delay_var.set(str(delay))
        
        # Pass5設定
        self.pass5_enabled_var.set(self.config.get_pass5_enabled())
        self.pass5_max_chars_var.set(str(self.config.get_pass5_max_chars()))
        
        # 保存先フォルダ設定
        saved_output_dir = self.config.get_output_dir()
        if saved_output_dir and saved_output_dir.exists():
            self.output_dir = saved_output_dir
            self.output_dir_var.set(f"保存先フォルダ: {self.output_dir}")
        else:
            self.output_dir = Path("output")
            self.output_dir_var.set("保存先フォルダ: output/ （デフォルト）")

    # --- イベントハンドラ（設定保存） ---

    def _on_model_changed(self, event: object) -> None:
        """Pass1-4モデル変更時に設定を保存。"""
        self.config.set_pass_model("pass1", self.pass1_model_var.get())
        self.config.set_pass_model("pass2", self.pass2_model_var.get())
        self.config.set_pass_model("pass3", self.pass3_model_var.get())
        self.config.set_pass_model("pass4", self.pass4_model_var.get())

    def _on_start_delay_changed(self, event: object) -> None:
        """開始遅延変更時に設定を保存。"""
        delay = self._get_start_delay()
        self.config.set_start_delay(delay)

    def _on_pass5_enabled_changed(self) -> None:
        """Pass5有効/無効変更時に設定を保存。"""
        self.config.set_pass5_enabled(self.pass5_enabled_var.get())

    def _on_pass5_model_changed(self, event: object) -> None:
        """Pass5モデル変更時に設定を保存。"""
        self.config.set_pass_model("pass5", self.pass5_model_var.get())

    def _on_pass5_chars_changed(self, event: object) -> None:
        """Pass5文字数変更時に設定を保存。"""
        chars = self._get_pass5_max_chars()
        self.config.set_pass5_max_chars(chars)

    def select_file(self) -> None:
        """音声ファイルを選択する。"""
        path = filedialog.askopenfilename(
            title="音声ファイルを選択",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac"), ("All Files", "*.*")],
        )
        if path:
            self.selected_file = Path(path)
            self.file_var.set(f"音声ファイル: {self.selected_file.name}")

    def select_output_dir(self) -> None:
        """保存先フォルダを選択する。"""
        path = filedialog.askdirectory(title="保存先フォルダを選択")
        if path:
            self.output_dir = Path(path)
            self.output_dir_var.set(f"保存先フォルダ: {self.output_dir}")
            self.config.set_output_dir(self.output_dir)

    def run_pipeline(self) -> None:
        """パイプラインを実行する。"""
        if not self.selected_file:
            messagebox.showerror("エラー", "音声ファイルを選択してください。")
            return
        
        if self.is_running:
            messagebox.showinfo("情報", "このワークフローは実行中です。")
            return
        
        self._set_running_state(True)
        self.phase_var.set("準備中")
        self.output_var.set("")
        self.progress["value"] = 0
        self._start_rolling_animation()
        
        # モデル名からプロバイダーを判定（Pass1のモデルで判定）
        pass1_model = self.pass1_model_var.get()
        provider = _get_provider_for_model(pass1_model)
        
        # Pass5用のプロバイダーとモデル
        pass5_model = self.pass5_model_var.get()
        pass5_provider = _get_provider_for_model(pass5_model)
        
        # コントローラーで並列実行
        self.controller.run_workflow(
            workflow_id=self.workflow_id,
            audio_path=self.selected_file,
            subtitle_dir=self.output_dir,
            llm_provider=provider,
            llm_profile=None,  # プロファイル廃止
            pass1_model=self.pass1_model_var.get().strip() or None,
            pass2_model=self.pass2_model_var.get().strip() or None,
            pass3_model=self.pass3_model_var.get().strip() or None,
            pass4_model=self.pass4_model_var.get().strip() or None,
            start_delay=self._get_start_delay(),
            enable_pass5=self.pass5_enabled_var.get(),
            pass5_max_chars=self._get_pass5_max_chars(),
            pass5_model=pass5_model,
            pass5_provider=pass5_provider,
            on_start=self._on_start,
            on_success=self._on_success,
            on_error=self._on_error,
            on_finish=self._on_finish,
            on_progress=self._on_progress,
        )

    def _on_start(self) -> None:
        """開始時のコールバック。"""
        pass

    def _on_success(self, output_paths: list[Path], metrics: dict | None) -> None:
        """成功時のコールバック。"""
        self._stop_rolling_animation()
        self.phase_var.set("完了")
        self.progress["value"] = 100
        if output_paths:
            self.output_var.set(f"出力: {output_paths[-1].name}")
        
        # メトリクス表示
        if metrics:
            total_tokens = metrics.get("total_tokens") or 0
            total_cost = float(metrics.get("total_cost_usd") or 0.0)
            elapsed_sec = float(metrics.get("total_elapsed_sec") or 0.0)
            self.metrics_var.set(
                f"トークン: {total_tokens:,} / コスト: ${total_cost:.4f} / 時間: {self._format_elapsed(elapsed_sec)}"
            )

    def _on_error(self, exc: Exception) -> None:
        """エラー時のコールバック。"""
        self._stop_rolling_animation()
        self.phase_var.set("エラー")
        self.progress["value"] = 0
        messagebox.showerror("エラー", str(exc))

    def _on_finish(self) -> None:
        """終了時のコールバック。"""
        self._set_running_state(False)

    def _on_progress(self, phase: str, progress: int) -> None:
        """進捗更新時のコールバック。"""
        self.base_phase_var.set(phase)
        self.progress["value"] = progress

    def _set_running_state(self, running: bool) -> None:
        """実行状態を設定する。"""
        self.is_running = running
        self.run_button.config(state=tk.DISABLED if running else tk.NORMAL)

    def _start_rolling_animation(self) -> None:
        """ローリングアニメーションを開始する。"""
        self.rolling_dots = 0
        self._update_rolling_animation()

    def _update_rolling_animation(self) -> None:
        """ローリングアニメーションを更新する。"""
        if not self.is_running:
            return
        self.rolling_dots = (self.rolling_dots % 3) + 1
        base_phase = self.base_phase_var.get() or "処理中"
        self.phase_var.set(base_phase + "." * self.rolling_dots)
        self.rolling_timer_id = self.root.after(500, self._update_rolling_animation)

    def _stop_rolling_animation(self) -> None:
        """ローリングアニメーションを停止する。"""
        if self.rolling_timer_id:
            self.root.after_cancel(self.rolling_timer_id)
            self.rolling_timer_id = None
        current_phase = self.phase_var.get()
        if current_phase:
            self.phase_var.set(current_phase.rstrip('.').rstrip())

    def _format_elapsed(self, seconds: float) -> str:
        """経過時間をフォーマットする。"""
        total = int(round(max(0.0, seconds)))
        minutes, sec = divmod(total, 60)
        if minutes > 0:
            return f"{minutes}分{sec}秒"
        return f"{sec}秒"

    def _get_start_delay(self) -> float:
        """start_delay入力値を取得する。無効な値は0.0を返す。"""
        try:
            value = float(self.start_delay_var.get().strip())
            return max(0.0, value)  # 負の値は0.0に
        except (ValueError, TypeError):
            return 0.0

    def _get_pass5_max_chars(self) -> int:
        """Pass5の文字数閾値を取得する。無効な値や8未満はデフォルト17を返す。"""
        try:
            value = int(self.pass5_max_chars_var.get().strip())
            if value < 8:
                return 17  # 最小8文字未満はデフォルト
            return value
        except (ValueError, TypeError):
            return 17

    def reload_llm_profiles(self) -> None:
        """LLMモデル一覧を再読み込みする。"""
        self._all_models = _get_all_models()
        for combo in self._pass_model_combos:
            combo["values"] = self._all_models


__all__ = ["WorkflowPanel"]
