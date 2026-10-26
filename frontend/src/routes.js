// URL routes for the app's pages (hash routing, so it works on any static host).
//
// PERMANENT: these route strings are submission-facing links (e.g. <site>/#/demo,
// <site>/#demo-video). They are keyed by page element id, never derived from the
// sidebar label or a page heading. Renaming a page's display text (e.g. "Docs" ->
// "Home") must NOT change its route here. Only ever add routes; never edit or remove one.
// (Owner decisions: 2026-10-25 the Studio moved from the bare URL to #/studio when Home took the bare URL;
// 2026-10-26 the Docs page was removed, see isRetiredDocsHash below.)
export const PAGE_ROUTES = Object.freeze({
    "page-workbench": "#/studio",
    "page-explore": "#/demo",
});

// Retired addresses that must keep working: the Docs page (#/docs, linked from the SIH deck, and its #docs-... anchors)
// was removed on 2026-10-26. They open Home, and src/boot.js drops the hash from the URL (replacing the history entry).
export function isRetiredDocsHash(hash) {
    const h = (hash || "").trim();
    return h === "#/docs" || h.startsWith("#docs-");
}

// Anchor ids that are linked from outside the app and must keep resolving.
// PERMANENT, same rule as above. #demo-video lives on Home (the "home" world).
export const DEMO_VIDEO_ANCHOR = "demo-video";

// Which world a URL opens (src/boot.js): "app" for the page routes and for anchors in the app's markup
// (appHasId(id)), "home" for everything else (bare URL, Home's anchors, the retired Docs addresses, unknown hashes).
// The desktop app has no Home: always "app".
export function worldForHash(hash, appHasId, desktop = false) {
    const h = (hash || "").trim();
    if (desktop || Object.values(PAGE_ROUTES).includes(h)) {
        return "app";
    }
    if (h.length > 1 && !h.startsWith("#/")) {
        let id = null;
        try {
            id = decodeURIComponent(h.slice(1));
        } catch {
            id = null; // malformed escape: unknown hash
        }
        if (id && appHasId(id)) {
            return "app";
        }
    }
    return "home";
}

const DEFAULT_PAGE = "page-workbench";

// hash -> { pageId, anchorId }.
// "#/demo" -> Demo page; an element id -> the page holding that
// element, scrolled to it; anything else -> the Studio (the app's default page; on the web, src/boot.js sends the bare
// URL and unknown hashes to Home before the app loads).
// findPageOfElement(id) returns the id of the .page containing element `id`, or null.
export function resolveHash(hash, findPageOfElement) {
    const h = (hash || "").trim();
    if (!h || h === "#" || h === "#/") {
        return { pageId: DEFAULT_PAGE, anchorId: null };
    }
    for (const [pageId, route] of Object.entries(PAGE_ROUTES)) {
        if (route && h === route) {
            return { pageId, anchorId: null };
        }
    }
    if (!h.startsWith("#/")) {
        const id = decodeURIComponent(h.slice(1));
        const pageId = findPageOfElement ? findPageOfElement(id) : null;
        if (pageId) {
            return { pageId, anchorId: id };
        }
    }
    return { pageId: DEFAULT_PAGE, anchorId: null };
}

// True when `hash` already points somewhere on `pageId` (its route or an anchor on it),
// so switching to that page does not need to rewrite the URL.
export function hashMatchesPage(hash, pageId, findPageOfElement) {
    return resolveHash(hash, findPageOfElement).pageId === pageId;
}
