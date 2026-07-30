//! `eframe::App` implementation — the egui UI rendering loop.
//!
//! Single-window layout, minimal styling. Tray-side menu events
//! (Show / Start / Stop / Quit) come in via the channel the
//! `tray` module installs; we drain it on each `update()` call.

use eframe::egui;

use crate::state::{AppState, EnrollmentOrigin};
use crate::tray::{TrayCommand, TrayHandles};

/// Guided enrollment text deliberately mixes vowels, voiced/unvoiced
/// consonants, numbers and sentence rhythms. Reading a fixed varied
/// passage for the full capture is substantially more repeatable than
/// five seconds of arbitrary speech.
const ENROLLMENT_READING_TEXT: &str = "今日は少し早起きをして、窓を開けたら涼しい風が入りました。\
赤い車と青い自転車が交差点を通ります。三、七、八、九と数えながら、普段どおりの声で話します。\
これから友達とゲームをして、夜は温かい飲み物を楽しみます。明日の予定を確認したあと、\
静かな部屋で本を読み、最後に深呼吸をします。";

pub struct MellonellaApp {
    state: AppState,
    tray: Option<TrayHandles>,
    /// Whether the main window is currently visible. Tracked
    /// separately so the tray menu's "Show / Hide" can flip it
    /// without going through the OS close request loop.
    window_visible: bool,
}

impl MellonellaApp {
    pub fn new(state: AppState, tray: Option<TrayHandles>) -> Self {
        Self {
            state,
            tray,
            window_visible: true,
        }
    }

    fn drain_tray_commands(&mut self, ctx: &egui::Context) {
        let Some(tray) = self.tray.as_ref() else {
            return;
        };
        while let Some(cmd) = tray.try_recv() {
            match cmd {
                TrayCommand::Show => {
                    self.window_visible = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                }
                TrayCommand::Toggle => {
                    if self.state.is_running() {
                        self.state.stop();
                    } else {
                        self.state.start();
                    }
                }
                TrayCommand::Quit => {
                    self.state.stop();
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
    }

    fn render_central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("声紋マイク");
            ui.add_space(8.0);

            self.render_enrollment_section(ui);
            ui.add_space(8.0);
            ui.separator();
            ui.add_space(6.0);
            self.render_device_row(ui);
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(8.0);
            self.render_run_controls(ui);
            ui.add_space(8.0);
            self.render_meters(ui);
            ui.add_space(6.0);
            self.render_error_row(ui);
            ui.add_space(6.0);
            self.render_settings_panel(ui);
        });
    }

    /// Top-of-window enrollment summary:
    /// - During an active recording: progress bar
    /// - No pool loaded: big "welcome" wizard panel asking the user
    ///   to record their voice (first-run experience)
    /// - Pool loaded: compact "● Profile" pill + Re-enroll button
    fn render_enrollment_section(&mut self, ui: &mut egui::Ui) {
        if self.state.is_recording() {
            ui.label("声紋登録:");
            self.render_recording_progress(ui);
            return;
        }
        if self.state.pool.is_some() {
            self.render_profile_pill(ui);
            ui.add_space(6.0);
            ui.weak(
                "再登録するときは、開始後に次の文章を20秒間、普段どおりの声で繰り返してください。",
            );
            ui.label(ENROLLMENT_READING_TEXT);
        } else {
            self.render_first_run_wizard(ui);
        }
    }

    /// Compact "● Profile loaded" indicator + Re-enroll + Test voice
    /// (live monitor toggle) buttons. The power-user "From WAV / Load
    /// / Save JSON" buttons live in the Settings panel; the main UI
    /// exposes only the actions the user would actually want from
    /// here.
    ///
    /// **Test voice** toggles a `LiveSession` that loops the mic
    /// straight through the gate to the selected output device — the
    /// most direct way to confirm "is my voice being recognised right
    /// now and reaching the speaker". Clicking again stops it.
    fn render_profile_pill(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let origin_short = match &self.state.origin {
                EnrollmentOrigin::None => "本人の声".to_string(),
                EnrollmentOrigin::Mic { secs } => format!("本人の声（{secs}秒）"),
                EnrollmentOrigin::AutoLoaded(_) => "本人の声（保存済み）".to_string(),
            };
            ui.label(
                egui::RichText::new(format!("● 声紋: {origin_short}"))
                    .color(egui::Color32::from_rgb(80, 200, 120)),
            );
            ui.label(format!("· 特徴点 {}個", self.state.pool_anchors));
            if self.state.pool_f0_mu > 0.0 {
                ui.weak(format!(
                    "· F0 μ={:.0} Hz σ={:.0} Hz",
                    self.state.pool_f0_mu, self.state.pool_f0_sigma
                ));
            }
            // Re-enroll / Test buttons live at the right-hand edge.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let recording = self.state.is_recording();
                let running = self.state.is_running();
                let record_secs = self.state.record_duration_secs;
                if ui
                    .add_enabled(!recording && !running, egui::Button::new("声紋を再登録"))
                    .on_hover_text("保存済みの声紋をすべて捨てて、新しい録音で置き換えます。")
                    .clicked()
                {
                    self.state.start_recording(record_secs, false);
                }
                if ui
                    .add_enabled(
                        !recording && !running,
                        egui::Button::new("＋声紋を追加登録"),
                    )
                    .on_hover_text(
                        "保存済みの声紋を残したまま、今の声を追加で登録します。\
                         朝と夜など、声の調子が変わる時間帯ごとに登録しておくと\
                         本人判定が安定します。",
                    )
                    .clicked()
                {
                    self.state.start_recording(record_secs, true);
                }
                let (label, hover) = if running {
                    ("■ フィルター停止", "Discordへ送る音声処理を停止します。")
                } else {
                    (
                        "▶ フィルター開始",
                        "マイク → 声紋抽出 → 選択した出力、の順で処理を開始します。",
                    )
                };
                if ui
                    .add_enabled(!recording, egui::Button::new(label))
                    .on_hover_text(hover)
                    .clicked()
                {
                    if running {
                        self.state.stop();
                    } else {
                        self.state.start();
                    }
                }
            });
        });
        if let Some(quality) = self.state.enrollment_quality.as_deref() {
            let colour = if quality.contains("要再登録") {
                egui::Color32::from_rgb(220, 130, 70)
            } else {
                egui::Color32::from_rgb(80, 200, 120)
            };
            ui.colored_label(colour, quality);
        }
    }

    /// First-run welcome / set-up-your-voice panel. Shown when no pool
    /// has been loaded yet (either because the user just installed the
    /// app, or they cleared `~/.config/mellonella/enrollment.json`).
    fn render_first_run_wizard(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .inner_margin(egui::Margin::same(16))
            .fill(ui.visuals().faint_bg_color)
            .corner_radius(egui::CornerRadius::same(6))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("最初に自分の声を登録します").heading());
                ui.add_space(4.0);
                ui.label(
                    "次の文章を20秒間、普段どおりの声で繰り返してください。\
                     十分な特徴点が取れた録音だけをPC内の声紋として保存します。",
                );
                ui.add_space(8.0);
                ui.label(egui::RichText::new(ENROLLMENT_READING_TEXT).strong());
                ui.add_space(12.0);
                let record_secs = self.state.record_duration_secs;
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new(format!(
                                    "🎙  自分の声を登録 ({record_secs:.0}秒)"
                                ))
                                .strong(),
                            )
                            .min_size(egui::vec2(220.0, 36.0)),
                        )
                        .clicked()
                    {
                        self.state.start_recording(record_secs, false);
                    }
                    ui.add(
                        egui::DragValue::new(&mut self.state.record_duration_secs)
                            .speed(0.5)
                            .range(15.0..=30.0)
                            .suffix(" s")
                            .fixed_decimals(0),
                    )
                    .on_hover_text("録音時間（15～30秒）");
                });
                ui.add_space(4.0);
                ui.weak(
                    "雑音や友達の声が入らない状態で登録してください。声紋はこのPC内だけに保存され、\
                     次回から自動で読み込まれます。特徴点が6個未満なら以前の声紋を残します。",
                );
            });
    }

    fn render_recording_progress(&mut self, ui: &mut egui::Ui) {
        let (elapsed, target, progress) = self
            .state
            .recorder
            .as_ref()
            .map_or((0.0, self.state.record_duration_secs, 0.0), |r| {
                (r.elapsed_seconds(), r.target_seconds(), r.progress())
            });
        ui.horizontal(|ui| {
            ui.label(format!("録音中 {elapsed:.1} / {target:.1}秒"));
            ui.add(
                egui::ProgressBar::new(progress)
                    .desired_width(120.0)
                    .desired_height(14.0),
            );
            if ui.button("キャンセル").clicked() {
                self.state.cancel_recording();
            }
        });
        ui.add_space(6.0);
        ui.label(egui::RichText::new(ENROLLMENT_READING_TEXT).strong());
    }

    fn render_device_row(&mut self, ui: &mut egui::Ui) {
        let busy = self.state.is_running() || self.state.is_recording();
        ui.horizontal(|ui| {
            ui.label("マイク入力:");
            let current = self
                .state
                .selected_input
                .clone()
                .unwrap_or_else(|| "(既定)".into());
            ui.add_enabled_ui(!busy, |ui| {
                egui::ComboBox::from_id_salt("input_combo")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.state.selected_input, None, "(既定)");
                        for d in &self.state.available_inputs {
                            ui.selectable_value(
                                &mut self.state.selected_input,
                                Some(d.name.clone()),
                                &d.name,
                            );
                        }
                    });
            });
        });
        ui.horizontal(|ui| {
            ui.label("Discordへ渡す出力:");
            let current = self
                .state
                .selected_output
                .clone()
                .unwrap_or_else(|| "(既定)".into());
            ui.add_enabled_ui(!busy, |ui| {
                egui::ComboBox::from_id_salt("output_combo")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.state.selected_output, None, "(既定)");
                        for d in &self.state.available_outputs {
                            ui.selectable_value(
                                &mut self.state.selected_output,
                                Some(d.name.clone()),
                                &d.name,
                            );
                        }
                    });
            });
            if ui.add_enabled(!busy, egui::Button::new("再読込")).clicked() {
                self.state.refresh_devices();
            }
        });
    }

    /// Capability indicators: which optional ONNX backends are wired
    /// up (DFN3 NS / TSE). The Test voice button on the profile pill
    /// is the only way to start / stop the live monitor, so this row
    /// is informational only.
    fn render_run_controls(&mut self, ui: &mut egui::Ui) {
        let dfn3_available = self.state.dfn3_available();
        let sepformer_available = self.state.sepformer_available();
        let tse_available = self.state.tse_available();
        ui.horizontal(|ui| {
            ui.weak(if dfn3_available {
                "● ノイズ除去"
            } else {
                "○ ノイズ除去モデル未設定"
            });
            ui.separator();
            ui.weak(if sepformer_available {
                "● 強力2話者分離（声紋選択）"
            } else if tse_available {
                "● 標準の本人音声抽出"
            } else {
                "○ 本人音声の抽出モデル未設定"
            });
        });
    }

    /// Step 18: live level meter + gate light. Renders two thin
    /// progress bars (input RMS, output RMS) and a coloured circle
    /// for the gate state. RMS is mapped via a log-ish scale so
    /// quiet speech (~-30 dBFS) still shows movement.
    fn render_meters(&self, ui: &mut egui::Ui) {
        let running = self.state.is_running();
        let in_rms = self.state.input_rms();
        let out_rms = self.state.output_rms();
        let gate_on = self.state.gate_on();

        ui.horizontal(|ui| {
            ui.label("入力:");
            ui.add(
                egui::ProgressBar::new(rms_to_bar(in_rms))
                    .desired_width(160.0)
                    .desired_height(10.0)
                    .fill(egui::Color32::from_rgb(80, 160, 220)),
            );
            ui.label("出力:");
            ui.add(
                egui::ProgressBar::new(rms_to_bar(out_rms))
                    .desired_width(160.0)
                    .desired_height(10.0)
                    .fill(egui::Color32::from_rgb(180, 180, 80)),
            );
            let (gate_label, gate_colour) = if !running {
                ("○ 停止", egui::Color32::DARK_GRAY)
            } else if gate_on {
                ("● 本人音声", egui::Color32::from_rgb(80, 200, 120))
            } else {
                ("○ 抑制中", egui::Color32::from_rgb(160, 80, 80))
            };
            ui.label(egui::RichText::new(gate_label).color(gate_colour));
        });
    }

    /// Step 19: collapsible "Settings" section with sliders for
    /// the user-tunable gate / envelope / refresh-cadence
    /// parameters. Gate controls are lock-free and update during a
    /// running session; model-path changes still take effect on the
    /// next start because they rebuild ONNX sessions.
    fn render_settings_panel(&mut self, ui: &mut egui::Ui) {
        let running = self.state.is_running();
        egui::CollapsingHeader::new("詳細設定")
            .default_open(false)
            .show(ui, |ui| {
                if running {
                    ui.weak("モデルの変更は次回フィルター開始時に反映されます。");
                }
                if self.state.sepformer_available() {
                    self.render_separator_controls(ui, running);
                } else {
                    self.render_gate_controls(ui, running);
                }
                self.render_tse_model_controls(ui);
            });
    }

    fn render_separator_controls(&mut self, ui: &mut egui::Ui, running: bool) {
        ui.separator();
        ui.label(egui::RichText::new("本人判定のしきい値（強力2話者分離）").strong());
        let mut threshold = self.state.separator_tuning.threshold();
        if ui
            .add(
                egui::Slider::new(&mut threshold, 0.20..=0.80)
                    .step_by(0.01)
                    .text("しきい値"),
            )
            .changed()
        {
            self.state.separator_tuning.set_threshold(threshold);
            self.state.save_separator_threshold();
        }
        let last_score = self.state.separator_tuning.last_best_score();
        if running && last_score > 0.0 {
            let passing = last_score >= threshold;
            ui.colored_label(
                if passing {
                    egui::Color32::from_rgb(80, 200, 120)
                } else {
                    egui::Color32::from_rgb(160, 80, 80)
                },
                format!(
                    "直近1秒の本人スコア: {last_score:.2}（{}）",
                    if passing { "通過" } else { "抑制" }
                ),
            );
        } else {
            ui.weak("フィルター実行中は直近1秒の本人スコアがここに表示されます。");
        }
        ui.weak(
            "一人で話しながらスコアを確認し、その少し下まで下げてください。\
             下げるほど他人の声も通りやすくなります。変更は即時反映されます。",
        );
    }

    fn render_gate_controls(&mut self, ui: &mut egui::Ui, running: bool) {
        ui.separator();
        ui.label(egui::RichText::new("本人判定のしきい値").strong());
        let mut threshold = self.state.gate_tuning.threshold();
        let mut hangover_ms = self.state.gate_tuning.hangover_ms();
        let mut release_ms = self.state.gate_tuning.release_ms();
        let changed = ui
            .add(
                egui::Slider::new(&mut threshold, 0.15..=0.85)
                    .step_by(0.01)
                    .text("厳しさ"),
            )
            .changed()
            | ui.add(
                egui::Slider::new(&mut hangover_ms, 100.0..=1_200.0)
                    .step_by(25.0)
                    .suffix(" ms")
                    .text("途切れ保護"),
            )
            .changed()
            | ui.add(
                egui::Slider::new(&mut release_ms, 30.0..=400.0)
                    .step_by(10.0)
                    .suffix(" ms")
                    .text("閉じ方"),
            )
            .changed();
        if changed {
            self.state.gate_tuning.set_threshold(threshold);
            self.state.gate_tuning.set_hangover_ms(hangover_ms);
            self.state.gate_tuning.set_release_ms(release_ms);
            self.state.save_gate_settings();
        }

        ui.horizontal(|ui| {
            ui.label("プリセット:");
            if ui.small_button("途切れにくい").clicked() {
                self.state.set_gate_preset(0.36, 700.0, 180.0);
            }
            if ui.small_button("標準").clicked() {
                self.state.set_gate_preset(0.45, 500.0, 120.0);
            }
            if ui.small_button("他人を厳しく").clicked() {
                self.state.set_gate_preset(0.58, 300.0, 80.0);
            }
        });
        self.render_gate_score(ui, running);
        ui.weak(
            "変更は実行中でも即時反映・自動保存されます。本人まで抑制される場合は厳しさを\
             0.03ずつ下げるか途切れ保護を伸ばし、友達が通る場合は厳しさを0.03ずつ上げます。\
             二人同時に話す区間は別の本人抽出モデルが自動処理します。",
        );
    }

    fn render_gate_score(&self, ui: &mut egui::Ui, running: bool) {
        let last_score = self.state.gate_tuning.last_score();
        if running && last_score != 0.0 {
            let effective = self.state.gate_tuning.effective_threshold();
            let passing = last_score >= effective;
            ui.colored_label(
                if passing {
                    egui::Color32::from_rgb(80, 200, 120)
                } else {
                    egui::Color32::from_rgb(220, 120, 80)
                },
                format!(
                    "現在の本人スコア: {last_score:.2} / しきい値 {effective:.2}（{}）",
                    if passing { "通過" } else { "抑制" }
                ),
            );
        } else if running {
            ui.weak("本人スコアを測定中です（発話開始から約0.75秒）。");
        }
    }

    fn render_tse_model_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label(egui::RichText::new("本人音声の抽出モデル").strong());
        ui.horizontal(|ui| {
            let label = self.state.tse_onnx_path.as_deref().map_or_else(
                || "(未選択)".to_string(),
                |p| {
                    p.file_name().map_or_else(
                        || p.display().to_string(),
                        |n| n.to_string_lossy().into_owned(),
                    )
                },
            );
            if ui.button("モデルを選択…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("ONNX model", &["onnx"])
                    .pick_file()
                {
                    self.state.tse_onnx_path = Some(path);
                }
            }
            if ui.button("モデルをダウンロード").clicked() {
                self.state.fetch_tse_from_hf();
            }
            ui.label(label);
            if self.state.tse_onnx_path.is_some() && ui.small_button("解除").clicked() {
                self.state.tse_onnx_path = None;
            }
        });
    }

    fn render_error_row(&self, ui: &mut egui::Ui) {
        if let Some(err) = &self.state.last_error {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), err);
        }
    }
}

impl eframe::App for MellonellaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll the live session for fresh counters / worker errors,
        // and the recorder for completion / progress, before
        // rendering so the UI reflects the latest state.
        self.state.poll_session();
        self.state.poll_recorder();
        self.drain_tray_commands(ctx);

        // Step 17: minimise-to-tray. When the user clicks the
        // window's close button AND the tray is available,
        // intercept the close so the live session keeps running in
        // the background. Without a tray (Linux without
        // AppIndicator, etc.) we fall through to the OS default
        // (close = quit) so users aren't trapped in a headless app.
        if ctx.input(|i| i.viewport().close_requested()) && self.tray.is_some() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
            self.window_visible = false;
        }

        // Step 17: keep the tray icon's visual state in sync with
        // the live-session state.
        if let Some(tray) = self.tray.as_mut() {
            tray.set_running(self.state.is_running());
        }

        self.render_central_panel(ctx);

        // Repaint at ~10 Hz so the counter / recording progress
        // displays stay alive even without UI interaction.
        if self.state.is_running() || self.state.is_recording() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

/// Map RMS in `[0, 1]` to a `[0, 1]` progress-bar reading via a
/// pseudo-log scale: -60 dBFS → 0.0, 0 dBFS → 1.0. Conversational
/// speech sits around -25 dBFS which lights about half the bar —
/// the sweet spot for a level meter that "moves visibly" without
/// pinning at the top.
fn rms_to_bar(rms: f32) -> f32 {
    if rms <= 1e-6 {
        return 0.0;
    }
    let db = 20.0 * rms.log10();
    ((db + 60.0) / 60.0).clamp(0.0, 1.0)
}
