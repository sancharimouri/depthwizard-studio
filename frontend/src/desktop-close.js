// ============================================================
// UNSAVED-WORK GUARD FOR WINDOW / TAB CLOSE
//
// Desktop (Tauri v2): a real custom confirmation is possible. The window's
// close request is intercepted and the same in-app modal as in-app
// navigation (job count + save options) decides it. Uses the global API
// (`app.withGlobalTauri: true`), so the web build needs no Tauri packages.
// This repo has no Tauri shell yet (no src-tauri/); the hook activates only
// when window.__TAURI__ is present.
//
// Web: browsers allow only their own generic beforeunload prompt on tab or
// window close (no custom text or buttons), so that is all the web build
// does — it doesn't try to fake custom content there.
// ============================================================

export async function installCloseGuard({ hasUnsaved, confirmUnsaved, isUnloadAllowed = () => false }) {
    const tauriWindow = window.__TAURI__?.window;

    if (tauriWindow?.getCurrentWindow) {
        const win = tauriWindow.getCurrentWindow();
        // Tauri v2 awaits this handler, then destroys the window unless the
        // event was prevented.
        await win.onCloseRequested(async event => {
            if (!hasUnsaved()) {
                return;
            }
            const choice = await confirmUnsaved("quit");
            if (choice === "cancel") {
                event.preventDefault();
            }
        });
        return "tauri";
    }

    window.addEventListener("beforeunload", event => {
        if (hasUnsaved() && !isUnloadAllowed()) {
            event.preventDefault();
            event.returnValue = ""; // older browsers need a (ignored) value
        }
    });
    return "web";
}
