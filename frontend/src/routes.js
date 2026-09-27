// URL routes for the three pages (hash routing, so it works on any static host).
//
// PERMANENT: these route strings are submission-facing links (e.g. <site>/#/docs,
// <site>/#demo-video). They are keyed by page element id, never derived from the
// sidebar label or a page heading. Renaming a page's display text (e.g. "Docs" ->
// "Home") must NOT change its route here. Only ever add routes; never edit or remove one.
export const PAGE_ROUTES = Object.freeze({
    "page-workbench": "",        // landing page: bare URL
    "page-explore": "#/demo",
    "page-docs": "#/docs",
});

// Anchor ids that are linked from outside the app and must keep resolving.
// PERMANENT, same rule as above.
export const DEMO_VIDEO_ANCHOR = "demo-video";

const DEFAULT_PAGE = "page-workbench";

// hash -> { pageId, anchorId }.
// "#/docs" -> Docs page; "#demo-video" (or any element id) -> the page holding that
// element, scrolled to it; anything else -> the landing page.
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
