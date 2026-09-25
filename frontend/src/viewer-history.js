// Undo / redo for the expanded 3D viewer's actions: rotations (camera),
// measurements and selections, notes, and view changes (layer, vertical
// exaggeration, scenario overlay).
//
// Two kinds of entries share one timeline:
//   - state entries: before/after snapshots of the undoable viewer state
//     (capture() / apply()), committed after a user action if the state
//     actually changed. One entry per action, however many internal updates
//     it caused (e.g. a whole point drag).
//   - camera entries: the camera before/after one user gesture (drag, scroll,
//     pinch). Recorded separately so the idle auto-rotation, which moves the
//     camera constantly, never floods the history.
//
// Pure logic (no DOM, no THREE), so it is unit-testable in Node.

const LIMIT = 200;

export function createViewerHistory({ capture, apply, applyCamera, sameCamera = () => false, onChange = () => {} }) {
    let past = [];
    let future = [];
    let last = null; // serialized state as of the last commit / undo / redo
    let applying = false;

    function push(entry) {
        past.push(entry);
        if (past.length > LIMIT) {
            past.shift();
        }
        future = [];
        onChange();
    }

    return {
        // Start a fresh timeline from the current state (new terrain, new viewer).
        reset() {
            past = [];
            future = [];
            last = JSON.stringify(capture());
            onChange();
        },
        // Record the state change caused by the action that just happened, if any.
        commit() {
            if (applying) {
                return false;
            }
            const now = JSON.stringify(capture());
            if (last === null) {
                last = now;
                return false;
            }
            if (now === last) {
                return false;
            }
            push({ kind: "state", before: last, after: now });
            last = now;
            return true;
        },
        // Record one camera gesture.
        commitCamera(before, after) {
            if (applying || sameCamera(before, after)) {
                return false;
            }
            push({ kind: "camera", before, after });
            return true;
        },
        undo() {
            const entry = past.pop();
            if (!entry) {
                return null;
            }
            applying = true;
            try {
                if (entry.kind === "camera") {
                    applyCamera(entry.before);
                } else {
                    apply(JSON.parse(entry.before));
                    last = entry.before;
                }
            } finally {
                applying = false;
            }
            future.push(entry);
            onChange();
            return entry;
        },
        redo() {
            const entry = future.pop();
            if (!entry) {
                return null;
            }
            applying = true;
            try {
                if (entry.kind === "camera") {
                    applyCamera(entry.after);
                } else {
                    apply(JSON.parse(entry.after));
                    last = entry.after;
                }
            } finally {
                applying = false;
            }
            past.push(entry);
            onChange();
            return entry;
        },
        get canUndo() {
            return past.length > 0;
        },
        get canRedo() {
            return future.length > 0;
        },
        get isApplying() {
            return applying;
        },
    };
}
