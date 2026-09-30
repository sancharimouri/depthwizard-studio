"""Output parity between two container_bench flow runs (build/slim/runs/<label>/<flow>/).

  python scripts/flow_parity.py build/slim/runs/final_amd64 build/slim/runs/rc_amd64 [png geotiff cdse]

Per flow:
- depth.json: the depth payload (data_b64 + shape/min/max) must be identical; timing (infer_s) and host are ignored.
- PNGs: pixel-identical (max |diff|).
- terrain.json: heights max |diff|, and bounds equal.
- fabdem_meta.json: the stats equal.
- response.json meta: differences excluding the job id, input id and timing.
Prints one line per file and exits 1 on any difference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IGNORE_META = {"job", "input"}
IGNORE_DEPTH = {"infer_s", "host", "fallback_used", "device"}


def meta_diff(a: dict, b: dict, path="") -> dict:
    out = {}
    for k in sorted(set(a) | set(b)):
        if k in IGNORE_META or (path.endswith("depth") and k in IGNORE_DEPTH):
            continue
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.update(meta_diff(va, vb, f"{path}.{k}" if path else k))
        elif va != vb:
            out[f"{path}.{k}" if path else k] = (va, vb)
    return out


def compare(a: Path, b: Path) -> list[tuple[str, bool, str]]:
    rows = []
    da, db = json.loads((a / "depth.json").read_text()), json.loads((b / "depth.json").read_text())
    same = all(da.get(k) == db.get(k) for k in ("data_b64", "shape", "min", "max", "encoding", "model"))
    rows.append(("depth.json", same, f"payload identical={same}"))
    for png in ("satellite.png", "relative_depth.png", "elevation.png"):
        ia = np.asarray(Image.open(a / png)).astype(np.int32)
        ib = np.asarray(Image.open(b / png)).astype(np.int32)
        ok = ia.shape == ib.shape and int(np.abs(ia - ib).max()) == 0
        rows.append((png, ok, f"identical={ok} shape {ia.shape} vs {ib.shape}"
                     + (f" max|diff| {int(np.abs(ia - ib).max())}" if ia.shape == ib.shape else "")))
    ta, tb = json.loads((a / "terrain.json").read_text()), json.loads((b / "terrain.json").read_text())
    ha, hb = np.asarray(ta["heights"], float), np.asarray(tb["heights"], float)
    dh = float(np.abs(ha - hb).max()) if ha.shape == hb.shape else float("inf")
    tb_eq = {k: v for k, v in ta.items() if k not in ("heights", "display")} == {k: v for k, v in tb.items() if k not in ("heights", "display")}
    rows.append(("terrain.json", dh == 0 and tb_eq, f"heights max|diff| {dh:.2e}, header equal={tb_eq}"))
    if (a / "fabdem_meta.json").exists() or (b / "fabdem_meta.json").exists():
        fa, fb = (json.loads((p / "fabdem_meta.json").read_text()) for p in (a, b))
        ok = fa == fb
        rows.append(("fabdem_meta.json", ok, f"stats equal={ok}"))
    ma = json.loads((a / "response.json").read_text())["meta"]
    mb = json.loads((b / "response.json").read_text())["meta"]
    d = meta_diff(ma, mb)
    rows.append(("response.json", not d, f"meta differences (excluding job/input id and timing): {d}"))
    return rows


def main():
    base, new = Path(sys.argv[1]), Path(sys.argv[2])
    flows = sys.argv[3:] or ["png", "geotiff", "cdse"]
    bad = 0
    for f in flows:
        for name, ok, msg in compare(base / f, new / f):
            bad += not ok
            print(f"{f:8s} {name:20s} {'OK  ' if ok else 'DIFF'} {msg}")
    print("PARITY: identical" if not bad else f"PARITY: {bad} difference(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
