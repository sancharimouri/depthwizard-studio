// Reusable magnifying-glass inspector for an <img> inside a positioned stage:
// a round 3× lens that follows the pointer, a pixel readout, and optional
// hover / pick callbacks with the pointer's position in image UV (0..1).
// Used by the 3D viewer's Image Inspection box and the input page's preview.
//
// Coordinates are measured on the image's CONTENT box: both images use
// object-fit: contain, so the element box can be letterboxed; positions in
// the letterbox margin count as "not on the image".

export function attachMagnifier({ stage, img, readout = null, zoom = 3, onHover, onLeave, onPick }) {
    let lens = stage.querySelector(".mag-lens");
    if (!lens) {
        lens = document.createElement("div");
        lens.className = "mag-lens";
        lens.hidden = true;
        stage.append(lens);
    }
    const marker = document.createElement("div");
    marker.className = "mag-marker";
    marker.hidden = true;
    stage.append(marker);
    let selected = null; // { u, v }

    const idleText = () => (img.naturalWidth
        ? `${img.naturalWidth} × ${img.naturalHeight} px preview · hover to magnify (${zoom}×)`
        : "");

    // the drawn image inside the element box (object-fit: contain)
    function contentRect() {
        const r = img.getBoundingClientRect();
        if (!img.naturalWidth || !r.width || !r.height) {
            return null;
        }
        const scale = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
        const w = img.naturalWidth * scale;
        const h = img.naturalHeight * scale;
        return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, width: w, height: h };
    }

    function uvAt(e) {
        if (img.hidden) {
            return null;
        }
        const c = contentRect();
        if (!c) {
            return null;
        }
        const u = (e.clientX - c.left) / c.width;
        const v = (e.clientY - c.top) / c.height;
        return u >= 0 && u <= 1 && v >= 0 && v <= 1 ? { u, v, c } : null;
    }

    function placeMarker() {
        const c = contentRect();
        if (!selected || !c || img.hidden) {
            marker.hidden = true;
            return;
        }
        const s = stage.getBoundingClientRect();
        marker.hidden = false;
        marker.style.left = `${c.left - s.left + selected.u * c.width}px`;
        marker.style.top = `${c.top - s.top + selected.v * c.height}px`;
    }

    function leave() {
        lens.hidden = true;
        if (readout) {
            readout.textContent = idleText();
        }
        onLeave?.();
    }

    stage.addEventListener("pointermove", e => {
        const hit = uvAt(e);
        if (!hit) {
            if (!lens.hidden) {
                leave();
            }
            return;
        }
        const { u, v, c } = hit;
        const s = stage.getBoundingClientRect();
        const size = lens.offsetWidth || 96;
        lens.hidden = false;
        lens.style.left = `${e.clientX - s.left - size / 2}px`;
        lens.style.top = `${e.clientY - s.top - size / 2}px`;
        lens.style.backgroundImage = `url("${img.src}")`;
        lens.style.backgroundSize = `${c.width * zoom}px ${c.height * zoom}px`;
        lens.style.backgroundPosition = `${size / 2 - u * c.width * zoom}px ${size / 2 - v * c.height * zoom}px`;
        const px = Math.floor(u * (img.naturalWidth - 1));
        const py = Math.floor(v * (img.naturalHeight - 1));
        if (readout) {
            readout.textContent = `px ${px}, ${py} of ${img.naturalWidth} × ${img.naturalHeight} (preview)`
                + (onPick ? " · click to select" : "");
        }
        onHover?.(u, v);
    });
    stage.addEventListener("pointerleave", leave);
    stage.addEventListener("click", e => {
        const hit = uvAt(e);
        if (hit && onPick) {
            onPick(hit.u, hit.v);
        }
    });
    img.addEventListener("load", () => {
        if (readout) {
            readout.textContent = idleText();
        }
        placeMarker();
    });
    new ResizeObserver(placeMarker).observe(stage);

    return {
        // show / hide the selected-pixel marker on the image (u, v in 0..1)
        setSelected(uv) {
            selected = uv;
            placeMarker();
        },
        refresh: placeMarker,
    };
}
