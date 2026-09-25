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
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

const PORT: u16 = 8765;

struct Sidecar(Mutex<Option<Child>>);

fn backend_exe(app: &tauri::AppHandle) -> PathBuf {
    let dir = app.path().resource_dir().expect("resource dir").join("dw2-backend");
    dir.join(if cfg!(windows) { "dw2-backend.exe" } else { "dw2-backend" })
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
        .manage(Sidecar(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![e2e_report])
        .setup(|app| {
            let exe = backend_exe(app.handle());
            // Never talk to a stale or foreign server that already holds the port.
            if TcpStream::connect(("127.0.0.1", PORT)).is_ok() {
                return Err(format!("port {PORT} is already in use; is another Depth Wizard backend running?").into());
            }
            let t0 = Instant::now();
            let mut child = Command::new(&exe)
                .env("DW2_PORT", PORT.to_string())
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
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the app")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(mut child) = app.state::<Sidecar>().0.lock().unwrap().take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        });
}
