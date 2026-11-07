#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod network;
mod output;

#[cfg(test)]
mod session_tests;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use eframe::egui;
use network::{Session, SessionOptions};
use output::OutputStore;
use std::{net::SocketAddr, path::PathBuf, time::Duration};

#[derive(Parser)]
#[command(version, about = "StreamTrans 录音客户端与实时网页输出")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:3000")]
    api: String,
    /// 与服务端一致的可选请求 key；也可通过 STREAMTRANS_REQUEST_KEY 设置。
    #[arg(long, env = "STREAMTRANS_REQUEST_KEY", hide_env_values = true)]
    request_key: Option<String>,
    #[arg(long, default_value = "127.0.0.1:3001")]
    web_bind: SocketAddr,
    #[arg(long)]
    css: Option<PathBuf>,
    #[arg(long)]
    list_devices: bool,
    #[arg(long)]
    headless: bool,
    /// Replay an existing 16 kHz mono PCM16 WAV without opening a microphone.
    #[arg(long, requires = "headless")]
    wav: Option<PathBuf>,
    #[arg(long, default_value = "en")]
    target: String,
}

fn main() {
    attach_console_if_cli();
    if let Err(error) = run() {
        let message = format!("客户端启动失败：{error:#}");
        eprintln!("{message}");
        if !std::env::args().any(|arg| arg == "--headless" || arg == "--list-devices") {
            show_error(&message);
        }
        std::process::exit(1);
    }
}

#[cfg(windows)]
fn attach_console_if_cli() {
    if std::env::args().any(|arg| {
        matches!(
            arg.as_str(),
            "--headless" | "--list-devices" | "--help" | "-h" | "--version" | "-V"
        )
    }) {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn AttachConsole(process_id: u32) -> i32;
        }
        unsafe {
            AttachConsole(u32::MAX);
        }
    }
}

#[cfg(not(windows))]
fn attach_console_if_cli() {}

fn run() -> Result<()> {
    let args = Args::parse();
    if args.list_devices {
        println!("{}", serde_json::to_string_pretty(&audio::list_devices()?)?);
        return Ok(());
    }
    network::validate_request_key(args.request_key.as_deref())?;
    ensure!(
        args.web_bind.ip().is_loopback(),
        "网页输出仅允许绑定本机回环地址"
    );
    let css = args.css.unwrap_or(
        std::env::current_exe()?
            .parent()
            .context("无法定位程序目录")?
            .join("output.css"),
    );
    output::prepare_css(&css)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(2)
        .build()?;
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind(args.web_bind))
        .with_context(|| {
            format!(
                "无法监听网页地址 {}，请使用 --web-bind 更换端口",
                args.web_bind
            )
        })?;
    let web_address = listener.local_addr()?;
    let store = OutputStore::new();
    let router = output::router(store.clone(), css.clone());
    let web_store = store.clone();
    runtime.spawn(async move {
        if let Err(error) = axum::serve(listener, router).await {
            web_store.set_error(Some(format!("网页输出服务中断：{error}")));
        }
    });
    if args.headless {
        println!("网页输出：http://{web_address}/\nCSS：{}", css.display());
        let session = if let Some(wav) = args.wav {
            let target = (args.target != "none").then_some(args.target);
            store.set_visibility(true, target.is_some());
            Some(Session::start(
                SessionOptions {
                    api: args.api,
                    request_key: args.request_key,
                    device_id: "default".into(),
                    target,
                    wav: Some(wav),
                },
                store.clone(),
            )?)
        } else {
            None
        };
        runtime
            .block_on(tokio::signal::ctrl_c())
            .context("无法等待退出信号")?;
        if let Some(session) = session {
            session.stop();
            runtime.block_on(async {
                let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
                while !session.is_finished() && tokio::time::Instant::now() < deadline {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            });
        }
    } else {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([820.0, 700.0])
                .with_min_inner_size([480.0, 440.0]),
            ..Default::default()
        };
        eframe::run_native(
            "StreamTrans · 实时转写与翻译",
            options,
            Box::new(move |cc| {
                let mut fonts = egui::FontDefinitions::default();
                fonts.font_data.insert(
                    "noto_sc".into(),
                    std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
                        "../assets/NotoSansSC.otf"
                    ))),
                );
                for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                    fonts
                        .families
                        .entry(family)
                        .or_default()
                        .push("noto_sc".into());
                }
                cc.egui_ctx.set_fonts(fonts);
                Ok(Box::new(ClientApp::new(
                    args.api,
                    args.request_key,
                    args.target,
                    web_address,
                    css,
                    store,
                )))
            }),
        )
        .map_err(|error| anyhow::anyhow!("无法打开桌面界面：{error}"))?;
    }
    Ok(())
}

const LANGUAGES: &[(&str, &str)] = &[
    ("none", "不翻译（仅识别）"),
    ("en", "英语"),
    ("zh", "中文"),
    ("ja", "日语"),
    ("ko", "韩语"),
    ("fr", "法语"),
    ("de", "德语"),
    ("es", "西班牙语"),
    ("ru", "俄语"),
    ("pt", "葡萄牙语"),
    ("ar", "阿拉伯语"),
    ("it", "意大利语"),
];

struct ClientApp {
    api: String,
    request_key: String,
    devices: Vec<audio::DeviceInfo>,
    device: String,
    target: String,
    show_original: bool,
    show_translation: bool,
    store: OutputStore,
    session: Option<Session>,
    web_url: String,
    css: PathBuf,
    display: String,
}

impl ClientApp {
    fn new(
        api: String,
        request_key: Option<String>,
        target: String,
        web: SocketAddr,
        css: PathBuf,
        store: OutputStore,
    ) -> Self {
        let mut app = Self {
            api,
            request_key: request_key.unwrap_or_default(),
            devices: Vec::new(),
            device: "default".into(),
            target,
            show_original: true,
            show_translation: true,
            store,
            session: None,
            web_url: format!("http://{web}/"),
            css,
            display: String::new(),
        };
        if !LANGUAGES.iter().any(|(code, _)| *code == app.target) {
            app.target = "en".into();
        }
        if app.target == "none" {
            app.show_translation = false;
        }
        app.refresh_devices();
        app
    }

    fn refresh_devices(&mut self) {
        match audio::list_devices() {
            Ok(devices) => {
                if !devices.iter().any(|device| device.id == self.device) {
                    self.device = "default".into();
                }
                self.devices = devices;
            }
            Err(error) => self
                .store
                .set_error(Some(format!("无法读取录音设备：{error:#}"))),
        }
    }

    fn start(&mut self) {
        self.store
            .set_visibility(self.show_original, self.show_translation);
        let target = (self.target != "none").then(|| self.target.clone());
        match Session::start(
            SessionOptions {
                api: self.api.clone(),
                request_key: (!self.request_key.is_empty()).then(|| self.request_key.clone()),
                device_id: self.device.clone(),
                target,
                wav: None,
            },
            self.store.clone(),
        ) {
            Ok(session) => self.session = Some(session),
            Err(error) => self
                .store
                .set_error(Some(format!("无法开始录音：{error:#}"))),
        }
    }
}

impl eframe::App for ClientApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.session.as_ref().is_some_and(Session::is_finished) {
            self.session = None;
        }
        let active = self.session.is_some();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("实时转写与翻译");
            ui.label("连续录音发送至服务端，由 FSMN-VAD 断句后识别。");
            ui.add_space(12.0);
            ui.add_enabled_ui(!active, |ui| {
                ui.label("API 地址");
                ui.add(
                    egui::TextEdit::singleline(&mut self.api)
                        .desired_width(f32::INFINITY)
                        .hint_text("http://127.0.0.1:3000"),
                );
                ui.label("请求 key（可选）");
                ui.add(
                    egui::TextEdit::singleline(&mut self.request_key)
                        .password(true)
                        .desired_width(f32::INFINITY)
                        .hint_text("与服务端配置一致；未设置时留空"),
                );
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label("录音设备");
                    let current = self
                        .devices
                        .iter()
                        .find(|device| device.id == self.device)
                        .map(|device| device.name.as_str())
                        .unwrap_or("系统默认设备");
                    egui::ComboBox::from_id_salt("input-device")
                        .width(300.0)
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for device in &self.devices {
                                ui.selectable_value(
                                    &mut self.device,
                                    device.id.clone(),
                                    &device.name,
                                );
                            }
                        });
                    if ui.button("刷新设备").clicked() {
                        self.refresh_devices();
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("目标语言");
                    let current = LANGUAGES
                        .iter()
                        .find(|(code, _)| *code == self.target)
                        .map(|(_, name)| *name)
                        .unwrap_or("英语");
                    egui::ComboBox::from_id_salt("target-language")
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for &(code, name) in LANGUAGES {
                                ui.selectable_value(&mut self.target, code.to_owned(), name);
                            }
                        });
                });
            });
            ui.add_space(8.0);
            if self.target == "none" {
                self.show_original = true;
                self.show_translation = false;
            }
            ui.horizontal_wrapped(|ui| {
                ui.add_enabled(
                    self.target != "none",
                    egui::Checkbox::new(&mut self.show_original, "显示原文"),
                );
                ui.add_enabled(
                    self.target != "none",
                    egui::Checkbox::new(&mut self.show_translation, "显示译文"),
                );
            });
            self.store
                .set_visibility(self.show_original, self.show_translation);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        !active,
                        egui::Button::new(if self.target == "none" {
                            "开始语音识别"
                        } else {
                            "开始自动翻译"
                        })
                        .min_size(egui::vec2(130.0, 32.0)),
                    )
                    .clicked()
                {
                    self.start();
                }
                if ui
                    .add_enabled(
                        active,
                        egui::Button::new("停止录音").min_size(egui::vec2(100.0, 32.0)),
                    )
                    .clicked()
                    && let Some(session) = &self.session
                {
                    session.stop();
                }
                if ui.button("清空结果").clicked() {
                    self.store.clear();
                }
            });
            ui.add_space(8.0);
            let snapshot = self.store.snapshot();
            ui.label(format!("状态：{}", snapshot.status));
            if let Some(error) = &snapshot.error {
                ui.colored_label(ui.visuals().error_fg_color, error);
            }
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.label("文本输出");
                if ui.button("复制文本").clicked() {
                    ctx.copy_text(output::render_text(&snapshot));
                }
                ui.hyperlink_to("打开实时网页", &self.web_url);
            });
            self.display = output::render_text(&snapshot);
            ui.small(format!("网页：{}　每 500 ms 刷新", self.web_url));
            ui.small(format!("CSS：{}", self.css.display()));
            ui.add_space(4.0);
            egui::ScrollArea::vertical()
                .id_salt("output-scroll")
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let mut text = self.display.as_str();
                    ui.add(
                        egui::TextEdit::multiline(&mut text)
                            .desired_width(f32::INFINITY)
                            .desired_rows(16)
                            .font(egui::TextStyle::Body),
                    );
                });
        });
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}

#[cfg(windows)]
fn show_error(message: &str) {
    #[link(name = "user32")]
    unsafe extern "system" {
        fn MessageBoxW(
            hwnd: *mut std::ffi::c_void,
            text: *const u16,
            caption: *const u16,
            flags: u32,
        ) -> i32;
    }
    let text: Vec<u16> = message.encode_utf16().chain(Some(0)).collect();
    let caption: Vec<u16> = "StreamTrans".encode_utf16().chain(Some(0)).collect();
    unsafe {
        MessageBoxW(std::ptr::null_mut(), text.as_ptr(), caption.as_ptr(), 0x10);
    }
}

#[cfg(not(windows))]
fn show_error(_message: &str) {}
