"""個別のワークフロー処理パネルを管理するウィジェット。"""
from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, Any

from src.config.settings import get_settings
from src.llm.profiles import get_profile, list_profiles, list_models_by_provider
from src.gui.config import get_config


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
        
        # LLM関連変数
        self.llm_provider_var = tk.StringVar()
        self.llm_profile_var = tk.StringVar()
        self.pass1_model_var = tk.StringVar()
        self.pass2_model_var = tk.StringVar()
        self.pass3_model_var = tk.StringVar()
        self.pass4_model_var = tk.StringVar()
        self.advanced_visible = tk.BooleanVar(value=False)
        self.save_logs_var = tk.BooleanVar(value=False)
        
        # プロファイルとモデル
        self._profiles = list_profiles()
        self._models_by_provider = list_models_by_provider()
        self._pass_model_combos: list[ttk.Combobox] = []
        
        # フレームのタイトル
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(title_frame, text=f"ワークフロー {workflow_id}", font=("", 10, "bold")).pack(side=tk.LEFT)
        
        self._build_widgets()
        self._load_initial_settings()
        self._apply_status_text_colors()

    def _build_widgets(self) -> None:
        """ウィジェットを構築する。"""
        # 緑色ボタンスタイルを設定
        self._setup_button_styles()
        
        # ファイル選択 + 実行ボタン
        file_frame = ttk.Frame(self)
        file_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(file_frame, textvariable=self.file_var, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 実行ボタン（緑色）- ttkボタンにカスタムスタイルを適用
        style = ttk.Style()
        style.configure(
            "Run.TButton",
            font=("", 10, "bold"),
        )
        # macOSでは背景色が効かないため、テキストで区別
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
        
        # プロバイダー表示
        provider_row = ttk.Frame(options_frame)
        provider_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(provider_row, text="LLMプロバイダー:").pack(side=tk.LEFT)
        ttk.Label(provider_row, textvariable=self.llm_provider_var).pack(side=tk.LEFT, padx=(4, 0))
        
        # プロファイル選択
        profile_row = ttk.Frame(options_frame)
        profile_row.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(profile_row, text="モデルプリセット:").pack(side=tk.LEFT)
        profile_names = sorted(self._profiles.keys())
        profile_combo = ttk.Combobox(
            profile_row,
            textvariable=self.llm_profile_var,
            values=profile_names,
            state="readonly",
            width=16,
        )
        profile_combo.pack(side=tk.LEFT, padx=(4, 0))
        profile_combo.bind("<<ComboboxSelected>>", self._on_profile_changed)
        
        # 詳細設定トグル
        advanced_row = ttk.Frame(options_frame)
        advanced_row.pack(fill=tk.X, pady=(2, 2))
        advanced_check = ttk.Checkbutton(
            advanced_row,
            text="詳細設定",
            variable=self.advanced_visible,
            command=self._toggle_advanced,
        )
        advanced_check.pack(side=tk.LEFT)

        # ログ保存トグル（デバッグ用）
        save_logs_check = ttk.Checkbutton(
            advanced_row,
            text="ログ保存",
            variable=self.save_logs_var,
        )
        save_logs_check.pack(side=tk.LEFT, padx=(12, 0))
        
        # 詳細設定エリア
        self.advanced_frame = ttk.Frame(options_frame)
        for idx, (label_text, var) in enumerate([
            ("Pass1:", self.pass1_model_var),
            ("Pass2:", self.pass2_model_var),
            ("Pass3:", self.pass3_model_var),
            ("Pass4:", self.pass4_model_var),
        ]):
            row = ttk.Frame(self.advanced_frame)
            row.pack(fill=tk.X, pady=(1, 1))
            ttk.Label(row, text=label_text, width=8).pack(side=tk.LEFT)
            combo = ttk.Combobox(
                row,
                textvariable=var,
                values=self._get_models_for_current_provider(),
                state="readonly",
            )
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._pass_model_combos.append(combo)
        
        # プログレスバー
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(4, 4))
        
        # ステータス表示
        status_row = ttk.Frame(self)
        status_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(status_row, text="ステータス:").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.phase_var).pack(side=tk.LEFT, padx=(4, 0))
        
        # 出力とメトリクス表示
        self.output_label = ttk.Label(self, textvariable=self.output_var)
        self.output_label.pack(fill=tk.X, pady=(0, 2))
        
        self.metrics_label = ttk.Label(self, textvariable=self.metrics_var)
        self.metrics_label.pack(fill=tk.X)

    def _apply_status_text_colors(self) -> None:
        """ダークモードでも読めるように、状態表示の文字色を調整する。"""
        style = ttk.Style()
        bg = (
            style.lookup("TFrame", "background")
            or style.lookup(".", "background")
            or self.root.cget("background")
        )
        try:
            r, g, b = self.root.winfo_rgb(bg)
            luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 65535.0
            is_dark = luminance < 0.5
        except Exception:
            # 判定に失敗した場合は、従来の色味（ライト想定）でフォールバック
            is_dark = False

        output_color = "#6cb6ff" if is_dark else "#0b4f6c"
        metrics_color = "#d6d6d6" if is_dark else "#555555"

        self.output_label.configure(foreground=output_color)
        self.metrics_label.configure(foreground=metrics_color)

    def _setup_button_styles(self) -> None:
        """ボタンスタイルを設定する。"""
        # ttkスタイルはプラットフォームによっては背景色が効かないため、
        # 実行ボタンは標準のtk.Buttonを使用する
        pass

    def _load_initial_settings(self) -> None:
        """初期設定を読み込む。"""
        # プロファイル設定
        if self._profiles:
            initial_profile = "default" if "default" in self._profiles else sorted(self._profiles.keys())[0]
            self.llm_profile_var.set(initial_profile)
            self.llm_provider_var.set(self._profiles[initial_profile].provider or "google")
            self._apply_profile_to_pass_models(initial_profile)
        else:
            self.llm_provider_var.set("google")
        
        # 保存先フォルダ設定
        saved_output_dir = self.config.get_output_dir()
        if saved_output_dir and saved_output_dir.exists():
            self.output_dir = saved_output_dir
            self.output_dir_var.set(f"保存先フォルダ: {self.output_dir}")
        else:
            self.output_dir = Path("output")
            self.output_dir_var.set("保存先フォルダ: output/ （デフォルト）")

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
        path = filedialog.askdirectory(title="SRTの保存先フォルダを選択")
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

        provider = (self.llm_provider_var.get() or "").strip().lower()
        if provider == "google":
            # 失敗を握りつぶすと「完了なのにトークン/コストが0」のように見えるため、
            # GUI側で先に設定漏れを検知して止める。
            if not get_settings().llm.google_api_key and not os.getenv("GOOGLE_API_KEY"):
                messagebox.showerror(
                    "エラー",
                    "Google APIキーが未設定です。\n画面上部の「API設定」から設定してください。",
                )
                return
        
        self._set_running_state(True)
        self.phase_var.set("準備中")
        self.base_phase_var.set("準備中")
        self.output_var.set("")
        self.metrics_var.set("")
        self.progress["value"] = 0
        self._start_rolling_animation()
        
        # コントローラーで並列実行
        self.controller.run_workflow(
            workflow_id=self.workflow_id,
            audio_path=self.selected_file,
            subtitle_dir=self.output_dir,
            llm_provider=self.llm_provider_var.get() or None,
            llm_profile=self.llm_profile_var.get() or None,
            pass1_model=self.pass1_model_var.get().strip() or None,
            pass2_model=self.pass2_model_var.get().strip() or None,
            pass3_model=self.pass3_model_var.get().strip() or None,
            pass4_model=self.pass4_model_var.get().strip() or None,
            save_logs=bool(self.save_logs_var.get()),
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
            total_elapsed_sec = float(metrics.get("total_elapsed_sec") or 0.0)
            metrics_files_found = int(metrics.get("metrics_files_found") or 0)
            time_str = self._format_elapsed(total_elapsed_sec)

            wait_elapsed_sec = metrics.get("wait_elapsed_sec")
            processing_elapsed_sec = metrics.get("processing_elapsed_sec")
            wait_str = self._format_elapsed(float(wait_elapsed_sec)) if isinstance(wait_elapsed_sec, (int, float)) else None
            processing_str = (
                self._format_elapsed(float(processing_elapsed_sec)) if isinstance(processing_elapsed_sec, (int, float)) else None
            )

            lines: list[str] = []
            if wait_str is not None and processing_str is not None:
                lines.append(f"時間: {time_str}（待機 {wait_str} / 実処理 {processing_str}）")
            else:
                lines.append(f"時間: {time_str}")

            if metrics_files_found <= 0:
                lines.append("トークン: - / コスト: -（メトリクス未取得）")
                self.metrics_var.set("\n".join(lines))
                return

            suffix = ""
            if int(total_tokens) <= 0 and total_cost <= 0.0:
                suffix = "（LLM未実行/usage未取得の可能性）"
            lines.append(f"トークン: {int(total_tokens)} / コスト: ${total_cost:.3f}{suffix}")

            per_runner = metrics.get("per_runner") or {}
            if isinstance(per_runner, dict) and per_runner:
                ordered_slugs = sorted(per_runner.keys())
                for slug in ordered_slugs:
                    info = per_runner.get(slug) or {}
                    if not isinstance(info, dict):
                        continue
                    transcribe_time = info.get("transcribe_time") or "-"
                    llm_two_pass_time = info.get("llm_two_pass_time") or "-"
                    durations = info.get("pass_durations") or {}
                    if not isinstance(durations, dict):
                        durations = {}

                    p1 = durations.get("pass1", "-")
                    p2 = durations.get("pass2", "-")
                    p3 = durations.get("pass3", "-")
                    p4 = durations.get("pass4", "-")

                    lines.append(f"[{slug}] 文字起こし: {transcribe_time} / LLM合計: {llm_two_pass_time}")

                    pass_line = f"[{slug}] Pass1: {p1} / Pass2: {p2} / Pass3: {p3} / Pass4: {p4}"
                    p5 = durations.get("pass5")
                    if isinstance(p5, str) and p5.strip():
                        pass_line += f" / Pass5: {p5.strip()}"
                    lines.append(pass_line)

            self.metrics_var.set("\n".join(lines))
        else:
            self.metrics_var.set("")

    def _on_error(self, exc: Exception) -> None:
        """エラー時のコールバック。"""
        self._stop_rolling_animation()
        self.phase_var.set(f"エラー: {str(exc)}")
        self.progress["value"] = 100
        messagebox.showerror("処理失敗", str(exc))

    def _on_finish(self) -> None:
        """完了時のコールバック。"""
        self._stop_rolling_animation()
        self._set_running_state(False)

    def _on_progress(self, phase_name: str, progress: int) -> None:
        """プログレス更新のコールバック。"""
        self.progress["value"] = progress
        self.base_phase_var.set(phase_name)
        self.phase_var.set(phase_name)

    def _set_running_state(self, running: bool) -> None:
        """実行状態を設定する。"""
        self.is_running = running
        state = tk.DISABLED if running else tk.NORMAL
        self.run_button.configure(state=state)
        # テキストも変更して状態を明示
        if running:
            self.run_button.configure(text="処理中...")
        else:
            self.run_button.configure(text="▶ 実行")

    def _start_rolling_animation(self) -> None:
        """ローリングアニメーションを開始する。"""
        self.rolling_dots = 0
        self._update_rolling_dots()

    def _update_rolling_dots(self) -> None:
        """ローリングドットを更新する。"""
        base_phase = self.base_phase_var.get()
        if base_phase and not base_phase.endswith("完了") and "エラー" not in base_phase:
            dots = "." * (self.rolling_dots % 4)
            self.phase_var.set(f"{base_phase} {dots}")
            self.rolling_dots += 1
            self.rolling_timer_id = self.root.after(500, self._update_rolling_dots)

    def _stop_rolling_animation(self) -> None:
        """ローリングアニメーションを停止する。"""
        if self.rolling_timer_id:
            self.root.after_cancel(self.rolling_timer_id)
            self.rolling_timer_id = None
        current_phase = self.phase_var.get()
        if current_phase:
            self.phase_var.set(current_phase.rstrip('.').rstrip())

    def _on_profile_changed(self, _event: object) -> None:
        """プロファイル変更時の処理。"""
        name = self.llm_profile_var.get()
        if name:
            self._apply_profile_to_pass_models(name)
            self.config.set_llm_profile(name)

    def _toggle_advanced(self) -> None:
        """詳細設定の表示/非表示を切り替える。"""
        if self.advanced_visible.get():
            self.advanced_frame.pack(fill=tk.X, pady=(4, 2))
        else:
            self.advanced_frame.pack_forget()

    def _get_models_for_current_provider(self) -> list[str]:
        """現在のプロバイダーのモデル一覧を取得する。"""
        provider = self.llm_provider_var.get() or ""
        models = sorted(self._models_by_provider.get(provider, set()))
        return models

    def _apply_profile_to_pass_models(self, profile_name: str) -> None:
        """プロファイルをPassモデルに適用する。"""
        profile = get_profile(profile_name)
        if profile is None:
            return
        self.pass1_model_var.set(profile.pass1_model or "")
        self.pass2_model_var.set(profile.pass2_model or "")
        self.pass3_model_var.set(profile.pass3_model or "")
        self.pass4_model_var.set(profile.pass4_model or "")
        self._refresh_advanced_model_choices()

    def reload_llm_profiles(self) -> None:
        """LLMプロファイルとモデル一覧を再読み込みする。"""
        self._profiles = list_profiles()
        self._models_by_provider = list_models_by_provider()
        current_profile = self.llm_profile_var.get()
        if current_profile and current_profile in self._profiles:
            provider = self._profiles[current_profile].provider or "google"
            self.llm_provider_var.set(provider)
            self._apply_profile_to_pass_models(current_profile)
        elif self._profiles:
            initial_profile = "default" if "default" in self._profiles else sorted(self._profiles.keys())[0]
            self.llm_profile_var.set(initial_profile)
            provider = self._profiles[initial_profile].provider or "google"
            self.llm_provider_var.set(provider)
            self._apply_profile_to_pass_models(initial_profile)
        else:
            self.llm_provider_var.set("")
        self._refresh_advanced_model_choices()

    def _refresh_advanced_model_choices(self) -> None:
        """詳細設定のモデル選択肢を更新する。"""
        models = self._get_models_for_current_provider()
        for combo in self._pass_model_combos:
            combo["values"] = models

    def _format_elapsed(self, seconds: float) -> str:
        """経過時間をフォーマットする。"""
        total = int(round(max(0.0, seconds)))
        minutes, sec = divmod(total, 60)
        if minutes > 0:
            return f"{minutes}分{sec}秒"
        return f"{sec}秒"


__all__ = ["WorkflowPanel"]
