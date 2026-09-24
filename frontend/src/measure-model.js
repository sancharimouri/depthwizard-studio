// State machine + undo stack for the 3D measurement tool. Pure logic (no
// THREE/DOM) so it is unit-testable in Node; measure-tool.js does the
// picking/drawing and feeds it grid coordinates.
//
// Modes: "normal" (navigation), "two-point", "continuous".
//
// The selection is a list of chains; each chain is an ordered list of points
// ({ id, x, y } in fractional grid pixels) plus a `closed` flag (polygon).
// Two-point mode only ever holds one chain of ≤ 2 points. In continuous mode
// new clicks append to the ACTIVE chain; more than one chain only exists
// after a middle segment of an open chain is deleted (it splits in two).
//
// Every edit (place, move, numeric edit, delete, clear, close-loop) pushes a
// snapshot of the selection first, so undo restores any number of steps
// exactly. Mode switches and Save are not edits and are not undone.

export const MODES = Object.freeze({ NORMAL: "normal", TWO_POINT: "two-point", CONTINUOUS: "continuous" });

const UNDO_LIMIT = 200;

export function createMeasureModel({ onChange } = {}) {
    let mode = MODES.NORMAL;
    let chains = [];
    let active = null; // index of the chain continuous clicks append to
    let owner = null; // the measure sub-mode the current selection was made in
    let nextId = 1;
    const undoStack = [];
    const saved = []; // SESSION-ONLY placeholder: no backend persistence exists yet

    const clone = () => ({ chains: chains.map(c => ({ closed: c.closed, points: c.points.map(p => ({ ...p })) })), active, owner });
    const pushUndo = () => {
        undoStack.push(clone());
        if (undoStack.length > UNDO_LIMIT) {
            undoStack.shift();
        }
    };
    const changed = () => onChange?.();
    const pointCount = () => chains.reduce((n, c) => n + c.points.length, 0);
    const newPoint = (x, y) => ({ id: nextId++, x, y });

    // Two-point selection is "complete" once it has its line; a continuous
    // selection once its active chain has been closed into a polygon.
    function isComplete() {
        if (mode === MODES.TWO_POINT) {
            return chains.length === 1 && chains[0].points.length === 2;
        }
        return chains.length > 0 && active === null && chains.every(c => c.closed);
    }

    function clearSelection() {
        chains = [];
        active = null;
    }

    function outsideMessage() {
        if (mode === MODES.TWO_POINT) {
            return pointCount() === 1
                ? "The second point must be on the terrain. Click on the terrain surface to finish the measurement."
                : "Click on the terrain surface to place the first point.";
        }
        return pointCount() === 0
            ? "Click on the terrain surface to start the chain."
            : "Points can only be placed on the terrain. Click on the terrain surface to add the next point.";
    }

    const api = {
        MODES,

        get mode() {
            return mode;
        },
        get chains() {
            return chains;
        },
        get activeChain() {
            return active;
        },
        get saved() {
            return saved;
        },
        get canUndo() {
            return undoStack.length > 0;
        },
        get undoDepth() {
            return undoStack.length;
        },
        isComplete,
        pointCount,

        // A selection belongs to the measure sub-mode that made it. Switching
        // to the OTHER sub-mode (directly, or via normal mode) clears it —
        // undoably — since two-point and continuous selections differ.
        // Returning to normal keeps it on screen, just not editable; coming
        // back to the same sub-mode keeps it editable.
        setMode(next) {
            if (next === mode) {
                return { type: "unchanged" };
            }
            if (next !== MODES.NORMAL && owner !== null && owner !== next && pointCount() > 0) {
                pushUndo();
                clearSelection();
            }
            mode = next;
            if (mode !== MODES.NORMAL) {
                owner = mode;
            }
            if (mode === MODES.CONTINUOUS && active === null) {
                const open = chains.findIndex(c => !c.closed);
                active = open >= 0 ? open : null;
            }
            changed();
            return { type: "mode", mode };
        },

        // hit: { x, y } grid position under the click, or null if off-terrain.
        // pointHit: { chain, index } if the click landed on an existing point.
        click(hit, pointHit = null) {
            if (mode === MODES.NORMAL) {
                return { type: "ignored" };
            }
            if (isComplete()) {
                pushUndo();
                clearSelection();
                changed();
                return { type: "cleared" };
            }
            if (mode === MODES.CONTINUOUS && pointHit && active !== null &&
                pointHit.chain === active && pointHit.index === 0 && chains[active].points.length >= 3) {
                pushUndo();
                chains[active].closed = true;
                active = null;
                changed();
                return { type: "closed" };
            }
            if (pointHit) {
                return {
                    type: "duplicate",
                    message: "A point is already there. Drag it to move it, or click somewhere else on the terrain.",
                };
            }
            if (!hit) {
                return { type: "error", message: outsideMessage() };
            }
            pushUndo();
            if (mode === MODES.TWO_POINT) {
                if (chains.length === 0) {
                    chains.push({ closed: false, points: [] });
                }
                chains[0].points.push(newPoint(hit.x, hit.y));
                active = 0;
            } else {
                if (active === null) {
                    chains.push({ closed: false, points: [] });
                    active = chains.length - 1;
                }
                chains[active].points.push(newPoint(hit.x, hit.y));
            }
            changed();
            return { type: "placed", complete: isComplete() };
        },

        // Drag: beginDrag snapshots once, moveDrag updates live (no undo
        // entry per frame), endDrag keeps the snapshot only if it moved.
        beginDrag(ref) {
            const p = chains[ref.chain]?.points[ref.index];
            if (!p) {
                return false;
            }
            api._dragSnapshot = clone();
            api._dragStart = { x: p.x, y: p.y };
            return true;
        },
        moveDrag(ref, x, y) {
            const p = chains[ref.chain]?.points[ref.index];
            if (p) {
                p.x = x;
                p.y = y;
                changed();
            }
        },
        endDrag(ref) {
            const p = chains[ref.chain]?.points[ref.index];
            const moved = p && api._dragStart && (p.x !== api._dragStart.x || p.y !== api._dragStart.y);
            if (moved) {
                undoStack.push(api._dragSnapshot);
                if (undoStack.length > UNDO_LIMIT) {
                    undoStack.shift();
                }
            }
            api._dragSnapshot = null;
            api._dragStart = null;
            return { type: moved ? "moved" : "unchanged" };
        },

        setPointXY(ref, x, y) {
            const p = chains[ref.chain]?.points[ref.index];
            if (!p) {
                return { type: "error", message: "That point no longer exists." };
            }
            if (p.x === x && p.y === y) {
                return { type: "unchanged" };
            }
            pushUndo();
            p.x = x;
            p.y = y;
            changed();
            return { type: "moved" };
        },

        deletePoint(ref) {
            const chain = chains[ref.chain];
            if (!chain?.points[ref.index]) {
                return { type: "error", message: "That point no longer exists." };
            }
            pushUndo();
            chain.points.splice(ref.index, 1);
            if (chain.closed && chain.points.length < 3) {
                chain.closed = false; // a polygon needs 3 points; reopen it
                active = ref.chain;
            }
            if (chain.points.length === 0) {
                chains.splice(ref.chain, 1);
                if (active === ref.chain) {
                    active = null;
                } else if (active !== null && active > ref.chain) {
                    active -= 1;
                }
            }
            if (mode === MODES.TWO_POINT) {
                active = chains.length ? 0 : null;
            } else if (active === null) {
                const open = chains.findIndex(c => !c.closed);
                active = open >= 0 ? open : null;
            }
            changed();
            return { type: "deleted-point" };
        },

        // Segment `index` joins point index → index+1 (→ 0 when closed).
        deleteSegment(ref) {
            const chain = chains[ref.chain];
            const n = chain?.points.length ?? 0;
            const count = chain ? (chain.closed ? n : n - 1) : 0;
            if (!chain || ref.index < 0 || ref.index >= count) {
                return { type: "error", message: "That line no longer exists." };
            }
            pushUndo();
            if (mode === MODES.TWO_POINT) {
                // The only line of a two-point measurement IS the measurement.
                clearSelection();
            } else if (chain.closed) {
                // Opening a polygon at this edge: it becomes an open chain
                // running from the edge's far end round to its near end.
                chain.points = [...chain.points.slice(ref.index + 1), ...chain.points.slice(0, ref.index + 1)];
                chain.closed = false;
                active = ref.chain;
            } else {
                // Split: [0..index] stays, [index+1..] becomes its own chain
                // and keeps the growing end (the active one if this was).
                const tail = { closed: false, points: chain.points.slice(ref.index + 1) };
                chain.points = chain.points.slice(0, ref.index + 1);
                chains.splice(ref.chain + 1, 0, tail);
                if (active !== null && active > ref.chain) {
                    active += 1;
                } else if (active === ref.chain) {
                    active = ref.chain + 1;
                }
            }
            changed();
            return { type: "deleted-segment" };
        },

        clear() {
            if (pointCount() === 0) {
                return { type: "unchanged" };
            }
            pushUndo();
            clearSelection();
            changed();
            return { type: "cleared" };
        },

        undo() {
            const snap = undoStack.pop();
            if (!snap) {
                return { type: "empty" };
            }
            chains = snap.chains;
            active = snap.active;
            owner = snap.owner;
            // Undoing a sub-mode switch's clear brings the selection back in
            // the mode that made it (a polygon can't live in two-point mode).
            const restoredMode = mode !== MODES.NORMAL && owner !== null && owner !== mode ? owner : null;
            if (restoredMode) {
                mode = restoredMode;
            }
            changed();
            return { type: "undone", mode: restoredMode };
        },

        // Saves every chain with a line (≥ 2 points). SESSION-ONLY: kept in
        // memory, lost on reload / Close. Placeholder for future persistence.
        save(measureChain) {
            const items = chains.filter(c => c.points.length >= 2);
            if (!items.length) {
                return {
                    type: "error",
                    message: mode === MODES.TWO_POINT
                        ? "Place two points before saving."
                        : "Place at least two points before saving.",
                };
            }
            const out = items.map(c => {
                const pts = c.points.map(p => ({ x: p.x, y: p.y }));
                const item = {
                    id: `m${saved.length + 1}`,
                    savedAt: new Date().toISOString(),
                    closed: c.closed,
                    points: pts,
                    metrics: measureChain ? measureChain(pts, c.closed) : null,
                };
                saved.push(item);
                return item;
            });
            changed();
            return { type: "saved", items: out };
        },
    };
    return api;
}
