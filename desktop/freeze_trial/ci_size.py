"""Installed size of a one-folder build + its compressed archive size(s); one JSON line."""
import json
import shutil
import sys
from pathlib import Path

d = Path(sys.argv[1])
files = [p for p in d.rglob("*") if p.is_file() and not p.is_symlink()]
total = sum(p.stat().st_size for p in files)
top = {}
for p in files:
    rel = p.relative_to(d).parts
    key = rel[1] if len(rel) > 2 and rel[0] == "_internal" else rel[0]
    top[key] = top.get(key, 0) + p.stat().st_size
out = {"installed_mb": round(total / 1e6, 1), "files": len(files),
       "top": {k: round(v / 1e6, 1) for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:10]}}
for fmt in (["zip", "gztar"] if sys.platform != "win32" else ["zip"]):
    a = shutil.make_archive(str(d.parent / f"dw2-backend-{fmt}"), fmt, d.parent, d.name)
    out[f"{fmt}_mb"] = round(Path(a).stat().st_size / 1e6, 1)
print(json.dumps(out))
