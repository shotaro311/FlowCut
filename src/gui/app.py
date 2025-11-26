"""Tkinter-based minimal GUI to run the existing CLI pipeline."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.llm.profiles import get_profile, list_profiles, list_models_by_provider
from src.gui.controller import GuiController


class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Flow Cut GUI (Phase 1)")
        self.root.geometry("520x360")

        self.controller = GuiController(ui_dispatch=self._dispatch_to_ui)
        self.selected_file: Path | None = None
        self.output_dir: Path | None = None

        # 状態表示用の変数
        self.status_var = tk.StringVar(value="待機中")
        self.file_var = tk.StringVar(value="音声ファイル: 未選択")
        self.output_dir_var = tk.StringVar(value="保存先フォルダ: output/ （デフォルト）")
        self.output_var = tk.StringVar(value="")
        self.metrics_var = tk.StringVar(value="")  # 総トークン数・コスト・処理時間表示用

        # LLM関連（プリセット＋詳細設定）
        self.llm_provider_var = tk.StringVar()
        self.llm_profile_var = tk.StringVar()
        self.pass1_model_var = tk.StringVar()
        self.pass2_model_var = tk.StringVar()
        self.pass3_model_var = tk.StringVar()
        self.pass4_model_var = tk.StringVar()
        self.advanced_visible = tk.BooleanVar(value=False)

        # プロファイル定義をロード（プロバイダーはプロファイル側の設定に従う）
        self._profiles = list_profiles()
        # プロファイルからプロバイダー別の既知モデル一覧を作る
        self._models_by_provider = list_models_by_provider()
        # 詳細設定で使うコンボボックス参照を保持
        self._pass_model_combos: list[ttk.Combobox] = []
        if self._profiles:
            # default があればそれを優先
            initial_profile = "default" if "default" in self._profiles else sorted(self._profiles.keys())[0]
            self.llm_profile_var.set(initial_profile)
            # プロバイダーはプロファイル側に合わせる（例: default=google/Gemini）
            self.llm_provider_var.set(self._profiles[initial_profile].provider or "google")
            self._apply_profile_to_pass_models(initial_profile)
        else:
            # フォールバック: プロバイダーは google(Gemini) 固定
            self.llm_provider_var.set("google")

        self._build_widgets()

    def _build_widgets(self) -> None:
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        file_label = ttk.Label(main_frame, textvariable=self.file_var, anchor=tk.W)
        file_label.pack(fill=tk.X, pady=(0, 8))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 8))

        select_button = ttk.Button(button_frame, text="ファイルを選択", command=self.select_file)
        select_button.pack(side=tk.LEFT)

        self.run_button = ttk.Button(button_frame, text="実行", command=self.run_pipeline)
        self.run_button.pack(side=tk.LEFT, padx=(8, 0))

        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=(0, 12))
        output_label = ttk.Label(output_frame, textvariable=self.output_dir_var, anchor=tk.W)
        output_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        select_output_button = ttk.Button(output_frame, text="保存先を変更", command=self.select_output_dir)
        select_output_button.pack(side=tk.RIGHT)

        # モデルプリセット＋詳細設定
        options_frame = ttk.LabelFrame(main_frame, text="LLMオプション（プリセット＋詳細設定）")
        options_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 12))

        # プロバイダー表示（GUIからは変更しない・プロファイル側で決まる）
        provider_row = ttk.Frame(options_frame)
        provider_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(provider_row, text="LLMプロバイダー:").pack(side=tk.LEFT)
        ttk.Label(provider_row, textvariable=self.llm_provider_var).pack(side=tk.LEFT, padx=(4, 0))

        # プロファイル（プリセット）選択
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
            text="詳細設定（Passごとのモデル名を直接指定）",
            variable=self.advanced_visible,
            command=self._toggle_advanced,
        )
        advanced_check.pack(side=tk.LEFT)

        # 詳細設定エリア（初期は非表示）
        self.advanced_frame = ttk.Frame(options_frame)
        # Pass1〜4用の入力欄（プルダウン形式）
        for idx, (label_text, var) in enumerate(
            [
                ("Pass1モデル:", self.pass1_model_var),
                ("Pass2モデル:", self.pass2_model_var),
                ("Pass3モデル:", self.pass3_model_var),
                ("Pass4モデル:", self.pass4_model_var),
            ]
        ):
            row = ttk.Frame(self.advanced_frame)
            row.pack(fill=tk.X, pady=(2 if idx == 0 else 1, 1))
            ttk.Label(row, text=label_text, width=10).pack(side=tk.LEFT)
            combo = ttk.Combobox(
                row,
                textvariable=var,
                values=self._get_models_for_current_provider(),
                state="readonly",
            )
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._pass_model_combos.append(combo)

        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(0, 12))

        status_row = ttk.Frame(main_frame)
        status_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(status_row, text="ステータス:").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.status_var).pack(side=tk.LEFT, padx=(4, 0))

        self.output_label = ttk.Label(main_frame, textvariable=self.output_var, foreground="#0b4f6c")
        self.output_label.pack(fill=tk.X, pady=(0, 2))

        self.metrics_label = ttk.Label(main_frame, textvariable=self.metrics_var, foreground="#555555")
        self.metrics_label.pack(fill=tk.X)

    def select_file(self) -> None:
        path = filedialog.askopenfilename(
            title="音声ファイルを選択",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac"), ("All Files", "*.*")],
        )
        if path:
            self.selected_file = Path(path)
            self.file_var.set(f"音声ファイル: {self.selected_file}")

    def select_output_dir(self) -> None:
        path = filedialog.askdirectory(title="SRTの保存先フォルダを選択")
        if path:
            self.output_dir = Path(path)
            self.output_dir_var.set(f"保存先フォルダ: {self.output_dir}")

    def run_pipeline(self) -> None:
        # デバッグ用ログ（GUIコマンドが呼ばれているか確認）
        print("[gui] run_pipeline called, selected_file=", self.selected_file)
        if not self.selected_file:
            messagebox.showerror("エラー", "音声ファイルを選択してください。")
            return

        self._set_running_state(True)
        self.status_var.set("処理中…")
        self.output_var.set("")
        self.progress.start(10)

        self.controller.run_pipeline(
            self.selected_file,
            subtitle_dir=self.output_dir,
            llm_provider=self.llm_provider_var.get() or None,
            llm_profile=self.llm_profile_var.get() or None,
            pass1_model=self.pass1_model_var.get().strip() or None,
            pass2_model=self.pass2_model_var.get().strip() or None,
            pass3_model=self.pass3_model_var.get().strip() or None,
            pass4_model=self.pass4_model_var.get().strip() or None,
            on_start=lambda: None,
            on_success=self._on_success,
            on_error=self._on_error,
            on_finish=self._on_finish,
        )

    def _on_success(self, output_paths: list[Path], metrics: dict | None) -> None:
        self.status_var.set("完了しました")
        if output_paths:
            self.output_var.set(f"出力: {output_paths[-1]}")
        # メトリクスサマリの表示
        if metrics:
            total_tokens = metrics.get("total_tokens") or 0
            total_cost = float(metrics.get("total_cost_usd") or 0.0)
            elapsed_sec = float(metrics.get("total_elapsed_sec") or 0.0)
            time_str = self._format_elapsed(elapsed_sec)
            self.metrics_var.set(
                f"総トークン数: {int(total_tokens)} / 概算APIコスト: ${total_cost:.3f} / 総処理時間: {time_str}"
            )
        else:
            self.metrics_var.set("")

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

    # --- LLMプリセット / 詳細設定用のヘルパー ---

    def _get_models_for_current_provider(self) -> list[str]:
        provider = self.llm_provider_var.get() or ""
        models = sorted(self._models_by_provider.get(provider, set()))
        return models

    def _format_elapsed(self, seconds: float) -> str:
        total = int(round(max(0.0, seconds)))
        minutes, sec = divmod(total, 60)
        if minutes > 0:
            return f"{minutes}分{sec}秒"
        return f"{sec}秒"

    def _apply_profile_to_pass_models(self, profile_name: str) -> None:
        """プロファイルを読み込み、詳細欄のPass1〜4モデルに反映する（空なら触らない）。"""
        profile = get_profile(profile_name)
        if profile is None:
            return
        # プリセット変更時に一括で上書きする。
        self.pass1_model_var.set(profile.pass1_model or "")
        self.pass2_model_var.set(profile.pass2_model or "")
        self.pass3_model_var.set(profile.pass3_model or "")
        self.pass4_model_var.set(profile.pass4_model or "")
        # モデル候補を再設定（プロバイダーに紐づく一覧）
        self._refresh_advanced_model_choices()

    def _on_profile_changed(self, _event: object) -> None:
        name = self.llm_profile_var.get()
        if name:
            self._apply_profile_to_pass_models(name)

    def _toggle_advanced(self) -> None:
        """詳細設定エリアの表示/非表示を切り替える。"""
        if self.advanced_visible.get():
            self.advanced_frame.pack(fill=tk.X, pady=(4, 2))
        else:
            self.advanced_frame.pack_forget()

    def _on_provider_changed(self, _event: object) -> None:
        """プロバイダー変更時に、詳細設定用のモデル候補を更新する。"""
        self._refresh_advanced_model_choices()

    def _refresh_advanced_model_choices(self) -> None:
        """現在のプロバイダーに応じて、Pass1〜4のモデル選択肢を更新する。"""
        models = self._get_models_for_current_provider()
        for combo in self._pass_model_combos:
            combo["values"] = models


def run_gui() -> None:
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


__all__ = ["run_gui", "MainWindow"]
