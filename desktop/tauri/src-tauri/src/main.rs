// Depth Wizard desktop shell: the existing Vite/Three.js frontend in a Tauri window,
// with the frozen Python backend (desktop/freeze_trial/, PyInstaller one-folder)
// started as a sidecar process on 127.0.0.1:8765 and stopped when the app exits.
//
// The one-folder backend (an executable plus its _internal/ folder) ships as a bundle
// resource and is spawned directly: Tauri's externalBin only takes single-file
// binaries, and a PyInstaller one-file build would unpack ~800 MB on every launch.
//
// DW2_E2E=1 (test only) runs src/e2e.js in the page: it drives the real UI
// (Library -> tile -> START GENERATION) and reports through the e2e_report command.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::webview::PageLoadEvent;
use tauri::{AppHandle, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_updater::UpdaterExt;

const PORT: u16 = 8765;

struct Sidecar(Mutex<Option<Child>>);

fn backend_exe(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    // Tauri refuses to resolve paths when the app's own path contains a symlink (a macOS
    // security check), e.g. an app run from /var/... instead of /private/var/... .
    let res = app.path().resource_dir().map_err(|e| {
        format!("can't locate the app's resources ({e}); run Depth Wizard from a normal folder such as /Applications (not a symlinked path)")
    })?;
    Ok(res.join("dw2-backend").join(if cfg!(windows) { "dw2-backend.exe" } else { "dw2-backend" }))
}

fn stop_sidecar(app: &AppHandle) {
    if let Some(mut child) = app.state::<Sidecar>().0.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Tauri's updater plugin: on launch, fetch the signed version manifest (plugins.updater
/// in tauri.conf.json); if a newer version exists, ask; on yes, download, verify the
/// signature against the bundled public key, install, stop the sidecar, restart.
/// DW2_UPDATE_AUTO_ACCEPT=1 skips the prompt (tests only).
async fn check_for_update(app: AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let current = app.package_info().version.to_string();
    let Some(update) = app.updater()?.check().await? else {
        eprintln!("update check: up to date ({current})");
        return Ok(());
    };
    eprintln!("update check: {} available (current {current})", update.version);
    let accept = if std::env::var("DW2_UPDATE_AUTO_ACCEPT").as_deref() == Ok("1") {
        eprintln!("update check: auto-accepted (test)");
        true
    } else {
        let dialog = app
            .dialog()
            .message(format!(
                "Depth Wizard {} is available (you have {current}).\n\n{}\n\nUpdate now? The app restarts when it is installed.",
                update.version,
                update.body.clone().unwrap_or_default()
            ))
            .title("Update available")
            .kind(MessageDialogKind::Info)
            .buttons(MessageDialogButtons::OkCancelCustom("Update".into(), "Later".into()));
        eprintln!("update check: prompting");
        tauri::async_runtime::spawn_blocking(move || dialog.blocking_show()).await?
    };
    if !accept {
        eprintln!("update check: postponed by the user");
        return Ok(());
    }
    // download_and_install verifies the signature after the download and refuses to install on a mismatch
    update.download_and_install(|_, _| {}, || eprintln!("update check: downloaded")).await?;
    eprintln!("update check: installed {}; restarting", update.version);
    stop_sidecar(&app);
    app.restart();
}

#[tauri::command]
fn e2e_report(app: tauri::AppHandle, result: String) {
    if let Ok(path) = std::env::var("DW2_E2E_OUT") {
        let _ = std::fs::write(path, &result);
    }
    println!("E2E {result}");
    app.exit(0);
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .manage(Sidecar(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![e2e_report])
        .setup(|app| {
            let exe = backend_exe(app.handle())?;
            // Never talk to a stale or foreign server that already holds the port.
            if TcpStream::connect(("127.0.0.1", PORT)).is_ok() {
                return Err(format!("port {PORT} is already in use; is another Depth Wizard backend running?").into());
            }
            let t0 = Instant::now();
            let mut child = Command::new(&exe)
                .env("DW2_PORT", PORT.to_string())
                // the tiered tile library shipped with the app (desktop/tiles/build_bundle.py)
                .env("DW2_LIBRARY_BUNDLE", app.path().resource_dir()?.join("library"))
                // the webview's own origin (macOS/Linux: tauri://localhost; Windows: http(s)://tauri.localhost)
                .env("CORS_ORIGINS", "tauri://localhost,http://tauri.localhost,https://tauri.localhost")
                .current_dir(exe.parent().expect("backend dir"))
                .spawn()
                .map_err(|e| format!("could not start the backend {}: {e}", exe.display()))?;
            while TcpStream::connect(("127.0.0.1", PORT)).is_err() && t0.elapsed() < Duration::from_secs(120) {
                if let Ok(Some(status)) = child.try_wait() {
                    return Err(format!("the backend exited during startup ({status})").into());
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            *app.state::<Sidecar>().0.lock().unwrap() = Some(child);
            eprintln!("backend listening after {:.1}s", t0.elapsed().as_secs_f32());

            let e2e = std::env::var("DW2_E2E").ok().map(|_| include_str!("e2e.js"));
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("Depth Wizard")
                .inner_size(1600.0, 1000.0)
                .on_page_load(move |window, payload| {
                    if payload.event() == PageLoadEvent::Finished {
                        if let Some(js) = e2e {
                            let _ = window.eval(js);
                        }
                    }
                })
                .build()?;
            if std::env::var("DW2_E2E").is_err() {
                let handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    if let Err(e) = check_for_update(handle).await {
                        eprintln!("update check failed: {e}");
                    }
                });
            }
            eprintln!("Depth Wizard {}", app.package_info().version);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the app")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                stop_sidecar(app);
            }
        });
}
