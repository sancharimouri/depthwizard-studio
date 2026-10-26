// Boot: one document, two worlds (index.html). Home (src/home/, imported from depthwizard-home) or the app
// (src/main.js), picked from the URL by worldForHash (src/routes.js). Only the chosen world's markup is inserted and
// only its code, styles and fonts load. Moving between worlds is a full page load, so Home's global styles, cursor
// and scroll lock never reach the app, and the app's never reach Home.
import { worldForHash } from "./routes.js";

const desktop = Boolean(window.__TAURI__);
const appMarkup = document.getElementById("world-app");
const appHasId = id => Boolean(appMarkup.content.getElementById(id));
const world = worldForHash(location.hash, appHasId, desktop);

document.getElementById(world === "home" ? "world-app" : "world-home").remove();
const markup = document.getElementById(`world-${world}`);
markup.replaceWith(markup.content.cloneNode(true)); // the templates stay intact for appHasId
document.documentElement.dataset.world = world;

// e.g. Home's ENTER STUDIO (/#/studio) or a #credits link typed in the app: a same-document hash change into the
// other world reloads the page there.
const followWorld = () => {
    if (worldForHash(location.hash, appHasId, desktop) !== world) {
        location.reload();
    }
};
window.addEventListener("hashchange", followWorld);
window.addEventListener("popstate", followWorld);

await (world === "home" ? import("./home/main.js") : import("./main.js"));
document.documentElement.classList.remove("dw-boot");
