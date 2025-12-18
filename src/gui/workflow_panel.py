"""個別のワークフロー処理パネルを管理するウィジェット。"""
from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.config.settings import get_settings
from src.llm.profiles import get_profile, list_models_by_provider, reload_profiles
from src.llm.workflows.registry import get_workflow, list_workflows
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
        self.file_var = tk.StringVar(value="メディア: 未選択")
        self.output_dir_var = tk.StringVar(value="保存先フォルダ: output/ （デフォルト）")
        self.output_var = tk.StringVar(value="")
        self.metrics_var = tk.StringVar(value="")
        self.phase_var = tk.StringVar(value="")
        self.base_phase_var = tk.StringVar(value="")
        self._last_output_dir: Path | None = None
        self.rolling_dots = 0
        self.rolling_timer_id = None
        
        # LLM関連変数
        self.workflow_var = tk.StringVar(value="workflow1")
        self.pass1_model_var = tk.StringVar()
        self.pass2_model_var = tk.StringVar()
        self.pass3_model_var = tk.StringVar()
        self.pass4_model_var = tk.StringVar()
        self.start_delay_var = tk.StringVar(value="0.2")
        self.advanced_visible = tk.BooleanVar(value=False)
        self.save_logs_var = tk.BooleanVar(value=False)
        self.keep_extracted_audio_var = tk.BooleanVar(value=False)
        self.notify_on_complete_var = tk.BooleanVar(value=False)
        self.pass5_enabled_var = tk.BooleanVar(value=False)
        self.pass5_max_chars_var = tk.StringVar(value="17")
        self.pass5_model_var = tk.StringVar()
        
        # モデル一覧
        self._models_by_provider = list_models_by_provider()
        self._pass_model_combos: list[ttk.Combobox] = []
        self._pass5_model_combo: ttk.Combobox | None = None
        
        # フレームのタイトル
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(title_frame, text=f"スロット {workflow_id}", font=("", 10, "bold")).pack(side=tk.LEFT)
        
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
        
        # ワークフロー選択
        workflow_row = ttk.Frame(options_frame)
        workflow_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(workflow_row, text="ワークフロー:").pack(side=tk.LEFT)
        workflow_options = [wf.slug for wf in list_workflows()]
        workflow_combo = ttk.Combobox(
            workflow_row,
            textvariable=self.workflow_var,
            values=workflow_options,
            state="readonly",
            width=16,
        )
        workflow_combo.pack(side=tk.LEFT, padx=(4, 0))
        workflow_combo.bind("<<ComboboxSelected>>", self._on_workflow_changed)
        ttk.Label(workflow_row, text="workflow2: 校正/最適化  workflow3: カスタム", foreground="#888888").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        
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
            command=self._on_save_logs_changed,
        )
        save_logs_check.pack(side=tk.LEFT, padx=(12, 0))

        pass5_check = ttk.Checkbutton(
            advanced_row,
            text="Pass5（長行改行）",
            variable=self.pass5_enabled_var,
            command=self._toggle_pass5,
        )
        pass5_check.pack(side=tk.LEFT, padx=(12, 0))

        keep_audio_check = ttk.Checkbutton(
            advanced_row,
            text="抽出音声を保存",
            variable=self.keep_extracted_audio_var,
            command=self._on_keep_extracted_audio_changed,
        )
        keep_audio_check.pack(side=tk.LEFT, padx=(12, 0))

        notify_check = ttk.Checkbutton(
            advanced_row,
            text="完了通知",
            variable=self.notify_on_complete_var,
            command=self._on_notify_on_complete_changed,
        )
        notify_check.pack(side=tk.LEFT, padx=(12, 0))
        
        # 詳細設定エリア
        self.advanced_frame = ttk.Frame(options_frame)
        self._advanced_pass_models_frame = ttk.Frame(self.advanced_frame)
        self._advanced_pass_models_frame.pack(fill=tk.X)
        self._render_advanced_pass_models()

        delay_row = ttk.Frame(self.advanced_frame)
        delay_row.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(delay_row, text="開始遅延:", width=8).pack(side=tk.LEFT)
        delay_entry = ttk.Entry(
            delay_row,
            textvariable=self.start_delay_var,
            width=6,
        )
        delay_entry.pack(side=tk.LEFT, padx=(0, 4))
        delay_entry.bind("<FocusOut>", self._on_start_delay_changed)
        ttk.Label(delay_row, text="秒（例: 0.2）", foreground="#888888").pack(side=tk.LEFT)

        # Pass5 設定
        self.pass5_frame = ttk.Frame(options_frame)
        pass5_model_row = ttk.Frame(self.pass5_frame)
        pass5_model_row.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(pass5_model_row, text="Pass5:", width=8).pack(side=tk.LEFT)
        self._pass5_model_combo = ttk.Combobox(
            pass5_model_row,
            textvariable=self.pass5_model_var,
            values=self._get_all_models(),
            state="readonly",
        )
        self._pass5_model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._pass5_model_combo.bind("<<ComboboxSelected>>", self._on_pass5_model_changed)

        pass5_chars_row = ttk.Frame(self.pass5_frame)
        pass5_chars_row.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(pass5_chars_row, text="", width=8).pack(side=tk.LEFT)
        ttk.Label(pass5_chars_row, text="文字数:").pack(side=tk.LEFT)
        pass5_chars_entry = ttk.Entry(
            pass5_chars_row,
            textvariable=self.pass5_max_chars_var,
            width=6,
        )
        pass5_chars_entry.pack(side=tk.LEFT, padx=(4, 0))
        pass5_chars_entry.bind("<FocusOut>", self._on_pass5_max_chars_changed)
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
        output_row = ttk.Frame(self)
        output_row.pack(fill=tk.X, pady=(0, 2))
        self.output_label = ttk.Label(output_row, textvariable=self.output_var, anchor=tk.W)
        self.output_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.open_output_button = ttk.Button(
            output_row,
            text="開く",
            command=self._open_output_folder,
            state=tk.DISABLED,
            width=6,
        )
        self.open_output_button.pack(side=tk.RIGHT, padx=(8, 0))

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
        self.workflow_var.set(self.config.get_workflow())

        all_models = set(self._get_all_models())
        saved_pass1 = self.config.get_pass_model("pass1", "").strip()
        saved_pass2 = self.config.get_pass_model("pass2", "").strip()
        saved_pass3 = self.config.get_pass_model("pass3", "").strip()
        saved_pass4 = self.config.get_pass_model("pass4", "").strip()
        saved_pass5 = (self.config.get_pass5_model() or "").strip()

        if saved_pass1 not in all_models:
            saved_pass1 = ""
        if saved_pass2 not in all_models:
            saved_pass2 = ""
        if saved_pass3 not in all_models:
            saved_pass3 = ""
        if saved_pass4 not in all_models:
            saved_pass4 = ""
        if saved_pass5 and saved_pass5 not in all_models:
            saved_pass5 = ""

        providers = sorted(self._models_by_provider.keys())
        provider = self._get_provider_for_model(saved_pass1) or (self.config.get_llm_provider() or "").strip().lower()
        if provider not in providers:
            provider = "google" if "google" in providers else (providers[0] if providers else "google")

        pass1 = saved_pass1 or (self._get_default_model_for_provider(provider, "pass1") or "")
        if pass1 and pass1 not in all_models:
            pass1 = ""
        if not pass1 and all_models:
            pass1 = sorted(all_models)[0]
        self.pass1_model_var.set(pass1)
        if pass1:
            self.config.set_pass_model("pass1", pass1)

        main_provider = self._get_provider_for_model(pass1) or provider
        self.config.set_llm_provider(main_provider)

        def pick(pass_name: str, saved: str) -> str:
            if saved and self._get_provider_for_model(saved) == main_provider:
                return saved
            return self._get_default_model_for_provider(main_provider, pass_name) or ""

        pass2 = pick("pass2", saved_pass2)
        pass3 = pick("pass3", saved_pass3)
        pass4 = pick("pass4", saved_pass4)
        self.pass2_model_var.set(pass2)
        self.pass3_model_var.set(pass3)
        self.pass4_model_var.set(pass4)
        if pass2:
            self.config.set_pass_model("pass2", pass2)
        if pass3:
            self.config.set_pass_model("pass3", pass3)
        if pass4:
            self.config.set_pass_model("pass4", pass4)

        self.pass5_model_var.set(saved_pass5)
        pass5_provider = self._get_provider_for_model(saved_pass5) if saved_pass5 else main_provider
        self.config.set_pass5_provider(pass5_provider)

        self._render_advanced_pass_models()
        self._refresh_advanced_model_choices()
        self.start_delay_var.set(str(self.config.get_start_delay()))
        self.save_logs_var.set(bool(self.config.get_save_logs()))
        self.keep_extracted_audio_var.set(bool(self.config.get_keep_extracted_audio()))
        self.notify_on_complete_var.set(bool(self.config.get_notify_on_complete()))
        
        # 保存先フォルダ設定
        saved_output_dir = self.config.get_output_dir()
        if saved_output_dir and saved_output_dir.exists():
            self.output_dir = saved_output_dir
            self.output_dir_var.set(f"保存先フォルダ: {self.output_dir}")
        else:
            self.output_dir = Path("output")
            self.output_dir_var.set("保存先フォルダ: output/ （デフォルト）")

        self.pass5_enabled_var.set(bool(self.config.get_pass5_enabled()))
        self.pass5_max_chars_var.set(str(self.config.get_pass5_max_chars()))
        self._toggle_pass5()

    def select_file(self) -> None:
        """メディアファイル（音声/動画）を選択する。"""
        path = filedialog.askopenfilename(
            title="音声/動画ファイルを選択",
            filetypes=[
                ("Media Files", "*.wav *.mp3 *.m4a *.flac *.mp4 *.mov *.mkv *.avi *.webm"),
                ("Audio Files", "*.wav *.mp3 *.m4a *.flac"),
                ("Video Files", "*.mp4 *.mov *.mkv *.avi *.webm"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.selected_file = Path(path)
            self.file_var.set(f"メディア: {self.selected_file.name}")

    def select_output_dir(self) -> None:
        """保存先フォルダを選択する。"""
        path = filedialog.askdirectory(title="SRTの保存先フォルダを選択")
        if path:
            self.output_dir = Path(path)
            self.output_dir_var.set(f"保存先フォルダ: {self.output_dir}")
            self.config.set_output_dir(self.output_dir)

    def _open_output_folder(self) -> None:
        target = self._last_output_dir or self.output_dir
        if target is None:
            return
        if not target.exists():
            messagebox.showerror("エラー", "出力フォルダが見つかりません。")
            return
        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", str(target)], check=False)
            elif os.name == "nt":
                os.startfile(str(target))  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(target)], check=False)
        except Exception as exc:
            messagebox.showerror("エラー", f"フォルダを開けませんでした: {exc}")

    def run_pipeline(self) -> None:
        """パイプラインを実行する。"""
        if not self.selected_file:
            messagebox.showerror("エラー", "音声/動画ファイルを選択してください。")
            return
        
        if self.is_running:
            messagebox.showinfo("情報", "このワークフローは実行中です。")
            return

        pass1_model = (self.pass1_model_var.get() or "").strip()
        provider = self._get_provider_for_model(pass1_model)
        if not provider:
            messagebox.showerror("エラー", f"Pass1のモデルからプロバイダーを判定できません: {pass1_model}")
            return
        self.config.set_llm_provider(provider)

        model_vars: dict[str, tk.StringVar] = {
            "pass1": self.pass1_model_var,
            "pass2": self.pass2_model_var,
            "pass3": self.pass3_model_var,
            "pass4": self.pass4_model_var,
        }
        for pass_name in self._get_active_pass_names():
            if pass_name == "pass1":
                continue
            model = (model_vars[pass_name].get() or "").strip()
            if model and self._get_provider_for_model(model) != provider:
                messagebox.showerror(
                    "エラー",
                    "Pass1と同じプロバイダーのモデルを選択してください。\n"
                    f"Pass1: {pass1_model}（{provider}）\n"
                    f"{pass_name}: {model}",
                )
                return

        settings = get_settings().llm

        enable_pass5 = bool(self.pass5_enabled_var.get())
        pass5_model = (self.pass5_model_var.get() or "").strip() or None
        pass5_provider = provider
        if enable_pass5:
            if pass5_model:
                inferred = self._get_provider_for_model(pass5_model)
                if not inferred:
                    messagebox.showerror("エラー", f"Pass5のモデルからプロバイダーを判定できません: {pass5_model}")
                    return
                pass5_provider = inferred

        for check_provider in {provider, pass5_provider} if enable_pass5 else {provider}:
            if check_provider == "google":
                # 失敗を握りつぶすと「完了なのにトークン/コストが0」のように見えるため、
                # GUI側で先に設定漏れを検知して止める。
                if not settings.google_api_key and not os.getenv("GOOGLE_API_KEY"):
                    messagebox.showerror(
                        "エラー",
                        "Google APIキーが未設定です。\n画面上部の「API設定」から設定してください。",
                    )
                    return
            elif check_provider == "openai":
                if not settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
                    messagebox.showerror(
                        "エラー",
                        "OpenAI APIキーが未設定です。\n画面上部の「API設定」から設定してください。",
                    )
                    return
            elif check_provider == "anthropic":
                if not settings.anthropic_api_key and not os.getenv("ANTHROPIC_API_KEY"):
                    messagebox.showerror(
                        "エラー",
                        "Anthropic APIキーが未設定です。\n画面上部の「API設定」から設定してください。",
                    )
                    return

        self.config.set_pass5_provider(pass5_provider)
        
        self._set_running_state(True)
        self.phase_var.set("準備中")
        self.base_phase_var.set("準備中")
        self.output_var.set("")
        self.metrics_var.set("")
        self._last_output_dir = None
        self.open_output_button.configure(state=tk.DISABLED)
        self.progress["value"] = 0
        self._start_rolling_animation()

        pass5_max_chars = 17
        if enable_pass5:
            try:
                pass5_max_chars = int((self.pass5_max_chars_var.get() or "").strip())
            except (TypeError, ValueError):
                messagebox.showerror("エラー", "Pass5の文字数が不正です（例: 17）")
                self._stop_rolling_animation()
                self._set_running_state(False)
                return
            if pass5_max_chars < 8:
                messagebox.showerror("エラー", "Pass5の文字数は 8 以上を指定してください。")
                self._stop_rolling_animation()
                self._set_running_state(False)
                return

        start_delay = self._get_start_delay()
        
        # コントローラーで並列実行
        self.controller.run_workflow(
            workflow_id=self.workflow_id,
            audio_path=self.selected_file,
            subtitle_dir=self.output_dir,
            llm_provider=provider,
            llm_profile=None,
            workflow=self.workflow_var.get() or "workflow1",
            pass1_model=self.pass1_model_var.get().strip() or None,
            pass2_model=self.pass2_model_var.get().strip() or None,
            pass3_model=self.pass3_model_var.get().strip() or None,
            pass4_model=self.pass4_model_var.get().strip() or None,
            start_delay=start_delay,
            keep_extracted_audio=bool(self.keep_extracted_audio_var.get()),
            enable_pass5=enable_pass5,
            pass5_max_chars=pass5_max_chars,
            pass5_provider=pass5_provider if enable_pass5 else None,
            pass5_model=pass5_model,
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
            srt_path = next((p for p in output_paths if p.suffix.lower() == ".srt"), None)
            last_path = srt_path or output_paths[-1]
            display = f"{last_path.parent.name}/{last_path.name}" if last_path.parent.name else last_path.name
            self.output_var.set(f"出力: {display}")
            self._last_output_dir = last_path.parent
            if self._last_output_dir.exists():
                self.open_output_button.configure(state=tk.NORMAL)

        if self.notify_on_complete_var.get():
            try:
                from src.utils.notification import send_notification

                output_name = output_paths[-1].name if output_paths else "完了"
                send_notification("FlowCut", f"字幕生成が完了しました: {output_name}")
            except Exception:
                pass
        
        # メトリクス表示
        if metrics:
            total_prompt_tokens = int(metrics.get("total_prompt_tokens") or 0)
            total_completion_tokens = int(metrics.get("total_completion_tokens") or 0)
            total_tokens = metrics.get("total_tokens") or 0
            total_cost = float(metrics.get("total_cost_usd") or 0.0)
            cost_available = bool(metrics.get("cost_available")) if "cost_available" in metrics else True
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
                lines.append("トークン: - / - / - / コスト: -（メトリクス未取得）")
                self.metrics_var.set("\n".join(lines))
                return

            suffix = ""
            if int(total_tokens) <= 0:
                suffix = "（LLM未実行/usage未取得の可能性）"
            elif not cost_available:
                suffix = "（単価未設定）"
            cost_str = f"${total_cost:.3f}" if cost_available else "-"
            lines.append(
                f"トークン: {total_prompt_tokens} / {total_completion_tokens} / {int(total_tokens)} / コスト: {cost_str}{suffix}"
            )

            per_runner = metrics.get("per_runner") or {}
            pass5_enabled = bool(metrics.get("pass5_enabled"))
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
                    pass_metrics = info.get("pass_metrics") or {}
                    if not isinstance(pass_metrics, dict):
                        pass_metrics = {}

                    lines.append(f"[{slug}] 文字起こし: {transcribe_time} / LLM合計: {llm_two_pass_time}")

                    available_labels: set[str] = set()
                    available_labels.update(durations.keys())
                    available_labels.update(pass_metrics.keys())
                    preferred_order = [
                        "pass1",
                        "pass2to4",
                        "pass2",
                        "pass3",
                        "pass4",
                        "pass4_fast",
                        "pass4_fallback",
                        "pass5",
                    ]
                    pass_labels = [label for label in preferred_order if label in available_labels]
                    for label in sorted(available_labels):
                        if label not in pass_labels:
                            pass_labels.append(label)

                    for label in pass_labels:
                        duration = durations.get(label)
                        duration_str = duration.strip() if isinstance(duration, str) and duration.strip() else "-"
                        usage = pass_metrics.get(label) or {}
                        if not isinstance(usage, dict):
                            usage = {}
                        prompt = usage.get("prompt_tokens")
                        completion = usage.get("completion_tokens")
                        total = usage.get("total_tokens")
                        if isinstance(prompt, (int, float)) and isinstance(completion, (int, float)) and isinstance(total, (int, float)):
                            token_str = f"{int(prompt)} / {int(completion)} / {int(total)}"
                        else:
                            token_str = "- / - / -"
                        cost = usage.get("cost_total_usd")
                        cost_str = f"${float(cost):.3f}" if isinstance(cost, (int, float)) else "-"
                        if label == "pass2to4":
                            pass_name = "Pass2-4"
                        elif label == "pass4_fast":
                            pass_name = "Pass4（fast）"
                        elif label == "pass4_fallback":
                            pass_name = "Pass4（fallback）"
                        else:
                            pass_name = label.replace("pass", "Pass")
                        lines.append(f"[{slug}] {pass_name}: {duration_str} / トークン: {token_str} / コスト: {cost_str}")

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

    def _on_workflow_changed(self, _event: object) -> None:
        self.config.set_workflow(self.workflow_var.get())
        self._render_advanced_pass_models()

    def _on_start_delay_changed(self, _event: object) -> None:
        self.config.set_start_delay(self._get_start_delay())

    def _on_save_logs_changed(self) -> None:
        self.config.set_save_logs(bool(self.save_logs_var.get()))

    def _on_keep_extracted_audio_changed(self) -> None:
        self.config.set_keep_extracted_audio(bool(self.keep_extracted_audio_var.get()))

    def _on_notify_on_complete_changed(self) -> None:
        self.config.set_notify_on_complete(bool(self.notify_on_complete_var.get()))

    def _toggle_advanced(self) -> None:
        """詳細設定の表示/非表示を切り替える。"""
        if self.advanced_visible.get():
            self.advanced_frame.pack(fill=tk.X, pady=(4, 2))
        else:
            self.advanced_frame.pack_forget()

    def _toggle_pass5(self) -> None:
        enabled = bool(self.pass5_enabled_var.get())
        self.config.set_pass5_enabled(enabled)
        if enabled:
            self.pass5_frame.pack(fill=tk.X, pady=(4, 2))
        else:
            self.pass5_frame.pack_forget()

    def _get_all_models(self) -> list[str]:
        models: set[str] = set()
        for bucket in self._models_by_provider.values():
            models.update(bucket)
        return sorted(models)

    def _get_provider_for_model(self, model: str) -> str | None:
        target = model.strip()
        if not target:
            return None
        for provider, models in self._models_by_provider.items():
            if target in models:
                return provider
        return None

    def _get_default_model_for_provider(self, provider: str, pass_name: str) -> str | None:
        models = set(self._models_by_provider.get(provider, set()))
        if not models:
            return None

        profile_name = "default" if provider == "google" else provider
        profile = get_profile(profile_name)
        if profile is not None:
            candidate_map = {
                "pass1": profile.pass1_model,
                "pass2": profile.pass2_model,
                "pass3": profile.pass3_model,
                "pass4": profile.pass4_model,
            }
            candidate = (candidate_map.get(pass_name) or "").strip()
            if candidate and candidate in models:
                return candidate

        return sorted(models)[0]

    def _get_active_pass_names(self) -> list[str]:
        wf = get_workflow(self.workflow_var.get())
        if wf.two_call_enabled and wf.pass2to4_prompt is not None:
            return ["pass1", "pass2"]
        return ["pass1", "pass2", "pass3", "pass4"]

    def _sync_pass_models_to_provider(self, provider: str) -> None:
        mapping: dict[str, tk.StringVar] = {
            "pass1": self.pass1_model_var,
            "pass2": self.pass2_model_var,
            "pass3": self.pass3_model_var,
            "pass4": self.pass4_model_var,
        }
        for pass_name in ("pass2", "pass3", "pass4"):
            var = mapping[pass_name]
            current = (var.get() or "").strip()
            if current and self._get_provider_for_model(current) == provider:
                continue
            default = self._get_default_model_for_provider(provider, pass_name) or ""
            if default:
                var.set(default)
                self.config.set_pass_model(pass_name, default)

    def _on_pass_model_changed(self, pass_name: str) -> None:
        mapping = {
            "pass1": self.pass1_model_var,
            "pass2": self.pass2_model_var,
            "pass3": self.pass3_model_var,
            "pass4": self.pass4_model_var,
        }
        var = mapping.get(pass_name)
        if var is None:
            return
        model = (var.get() or "").strip()
        if model:
            self.config.set_pass_model(pass_name, model)
        if pass_name == "pass1" and model:
            provider = self._get_provider_for_model(model)
            if provider:
                self.config.set_llm_provider(provider)
                self.config.set_pass5_provider(self._get_provider_for_model(self.pass5_model_var.get()) or provider)
                self._sync_pass_models_to_provider(provider)

    def reload_llm_profiles(self) -> None:
        """LLMモデル一覧を再読み込みする。"""
        reload_profiles()
        self._models_by_provider = list_models_by_provider()
        all_models = set(self._get_all_models())

        pass1 = (self.pass1_model_var.get() or "").strip()
        if pass1 not in all_models:
            pass1 = ""
        if not pass1 and all_models:
            providers = sorted(self._models_by_provider.keys())
            provider = (self.config.get_llm_provider() or "").strip().lower()
            if provider not in providers:
                provider = "google" if "google" in providers else (providers[0] if providers else "google")
            pass1 = self._get_default_model_for_provider(provider, "pass1") or sorted(all_models)[0]
            self.pass1_model_var.set(pass1)
            self.config.set_pass_model("pass1", pass1)

        main_provider = self._get_provider_for_model(pass1) or self.config.get_llm_provider()
        if main_provider:
            self.config.set_llm_provider(main_provider)
            self._sync_pass_models_to_provider(main_provider)

        pass5_model = (self.pass5_model_var.get() or "").strip()
        if pass5_model and pass5_model not in all_models:
            self.pass5_model_var.set("")
            self.config.set_pass5_model(None)
            pass5_model = ""
        self.config.set_pass5_provider(self._get_provider_for_model(pass5_model) or main_provider)

        self._render_advanced_pass_models()
        self._refresh_advanced_model_choices()

    def _refresh_advanced_model_choices(self) -> None:
        """詳細設定のモデル選択肢を更新する。"""
        models = self._get_all_models()
        for combo in self._pass_model_combos:
            combo["values"] = models
        if self._pass5_model_combo is not None:
            self._pass5_model_combo["values"] = models

    def _render_advanced_pass_models(self) -> None:
        """ワークフローに応じて、詳細設定の Pass モデル欄を構築する。"""
        frame = getattr(self, "_advanced_pass_models_frame", None)
        if frame is None:
            return

        for child in frame.winfo_children():
            child.destroy()
        self._pass_model_combos = []

        wf = get_workflow(self.workflow_var.get())
        if wf.two_call_enabled and wf.pass2to4_prompt is not None:
            rows = [
                ("Pass1:", "pass1", self.pass1_model_var),
                ("Pass2-4:", "pass2", self.pass2_model_var),
            ]
        else:
            rows = [
                ("Pass1:", "pass1", self.pass1_model_var),
                ("Pass2:", "pass2", self.pass2_model_var),
                ("Pass3:", "pass3", self.pass3_model_var),
                ("Pass4:", "pass4", self.pass4_model_var),
            ]

        models = self._get_all_models()
        for label_text, pass_name, var in rows:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=(1, 1))
            ttk.Label(row, text=label_text, width=8).pack(side=tk.LEFT)
            combo = ttk.Combobox(
                row,
                textvariable=var,
                values=models,
                state="readonly",
            )
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            combo.bind("<<ComboboxSelected>>", lambda _event, name=pass_name: self._on_pass_model_changed(name))
            self._pass_model_combos.append(combo)

    def _on_pass5_max_chars_changed(self, _event: object) -> None:
        raw = (self.pass5_max_chars_var.get() or "").strip()
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 17
        if value < 8:
            value = 8
        self.pass5_max_chars_var.set(str(value))
        self.config.set_pass5_max_chars(value)

    def _on_pass5_model_changed(self, _event: object) -> None:
        model = (self.pass5_model_var.get() or "").strip()
        self.config.set_pass5_model(model or None)

    def _get_start_delay(self) -> float:
        raw = (self.start_delay_var.get() or "").strip()
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, value)

    def _format_elapsed(self, seconds: float) -> str:
        """経過時間をフォーマットする。"""
        total = int(round(max(0.0, seconds)))
        minutes, sec = divmod(total, 60)
        if minutes > 0:
            return f"{minutes}分{sec}秒"
        return f"{sec}秒"


__all__ = ["WorkflowPanel"]
