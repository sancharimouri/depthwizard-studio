# RS3DAda — Stage 0 feasibility & contamination gate

**Question:** is RS3DAda even testable as a fair zero-shot comparison on
this project's 50-tile DFC2019 benchmark? Feasibility only — the full
50-tile benchmark was NOT run (per instruction).

**Bottom line up front: it loads and runs cleanly (MPS, no CUDA patching
needed), but it cannot be called zero-shot or independent. 49 of our 50
benchmark tiles are literally present in RS3DAda's own training split file,
and the paper's own dataset table confirms DFC19_JAX/DFC19_OMA were used as
named Target-Domain datasets during UDA training.** Any future comparison
against RS3DAda must be reported as "in-domain / partially seen," not
zero-shot.

## 1. Repo clone

`git clone --depth 1 https://github.com/JTRNEO/SynRS3D.git external/SynRS3D`
— same flat structure as `external/RDAH-Net/` (repo contents directly under
`external/SynRS3D/`, checkpoint alongside code once downloaded). Left
untracked in git, matching how `external/RDAH-Net/104best_model.pth` is
already handled (checked: it's untracked, not gitignored — no explicit rule
for `external/`, just an established local convention of not committing
large model files). Did the same for `external/SynRS3D/` — no `git add`.

## 2. Checkpoint download and hash verification

Downloaded `RS3DAda_vitl_DPT_height.pth` (1.47 GB) via
`huggingface_hub.hf_hub_download('JTRNEO/RS3DAda', ...)` to
`external/SynRS3D/pretrain/`.

**Verified, not trusted** — two independent sources agreed before and after
download:

| Source | SHA256 |
|---|---|
| `curl -I` on the resolve URL (`x-linked-etag` header) | `6c1c6843c3583604227896583acfee621dd841cbadd49c023d3d32bcc4b3565b` |
| `huggingface_hub.HfApi().model_info(..., files_metadata=True)` (`BlobLfsInfo.sha256`) | `6c1c6843c3583604227896583acfee621dd841cbadd49c023d3d32bcc4b3565b` |
| **Local file, `shasum -a 256` after download** | **`6c1c6843c3583604227896583acfee621dd841cbadd49c023d3d32bcc4b3565b`** |

All three match exactly. File size also matches exactly: 1,466,477,172
bytes reported by HF, identical on disk.

## 3. Contamination audit — exact overlap, not just paper-level claim

The paper's own dataset table (`README.md` lines 84–86) already states
DFC19_JAX and DFC19_OMA are **Target Domain 1** datasets — used during
RS3DAda's UDA training, confirming this is not zero-shot before even
checking file-level overlap. Did the file-level check anyway, since "used
as a named target domain" doesn't tell you *how much* of our specific
50-tile benchmark was actually touched.

**Method:** the repo ships `data/DFC19_JAX/*.txt` and `data/DFC19_OMA/*.txt`
split files directly (no dataset download needed — these are tracked in the
git clone). Entries are 512×512-patch identifiers in the form
`{TILE_ID}_{row_offset}_{col_offset}` (e.g. `JAX_204_004_512_512` = tile
`JAX_204_004`, patch at offset (512,512) within the 1024×1024 tile). Parsed
out the base tile ID from every line across every split file and diffed
against `dav2_baseline/manifest.csv`'s 50 tile IDs.

**Result: 50 of 50 of our benchmark tiles appear somewhere in RS3DAda's
split files** (their files collectively cover essentially the full
2,783-tile DFC2019 set). The more decision-relevant breakdown — membership
in `train.txt` specifically (the file RS3DAda's own training script points
`--pesudo_file` at, see §3b) — per city:

| | in `train.txt` | in `test.txt` | in `test_tgt.txt` |
|---|---:|---:|---:|
| JAX (26 of our tiles) | 25 / 26 | 23 / 26 | 6 / 26 |
| OMA (24 of our tiles) | 24 / 24 (100%) | 16 / 24 | 2 / 24 |
| **Total (50 tiles)** | **49 / 50** | 39 / 50 | 8 / 50 |

Only **one** of our 50 tiles (`JAX_004_016`) is absent from `train.txt` —
it appears only in `test.txt`/`test_tgt.txt`. Every other tile, **all 24
OMA tiles without exception**, is in RS3DAda's training list.

**3b. What "in train.txt" actually means for this model, checked in code,
not assumed:** `RS3DAda.sh` (the paper's own training launch script) passes
`--pesudo_datasets ${test_set[*]}` (includes `DFC19_JAX`, `DFC19_OMA`) and
`--pesudo_file 'train.txt'`. Cross-referenced against
`train_dpt_RS3DAda.py`'s argparse help text: `--pesudo_datasets` = "target
domain datasets list used for generate pesudo labels." This is a
self-training/pseudo-label UDA setup (also uses `ClassMix`, an EMA teacher
`--ema_alpha`, and a pseudo-label confidence threshold) — the model does
**not** train on DFC2019's real ground-truth height labels, but it **does**
train directly on the real optical imagery of `train.txt`'s tiles, using
self-generated pseudo-height-labels on those same images as a training
signal. So: no ground-truth label leakage, but real, direct pixel-level
exposure to 49 of our 50 tiles' actual imagery during adaptation, with a
model-in-the-loop feedback signal specific to those tiles. This is
meaningfully different from (and weaker cover than) a truly withheld
zero-shot tile, even without the label leakage.

**Conclusion, stated plainly as instructed:** RS3DAda cannot be reported as
a zero-shot or independent test on this benchmark. 49 of 50 tiles were
directly used (imagery + self-generated pseudo-labels) during its own UDA
training; the 1 exception (`JAX_004_016`) is the only tile that could even
be framed as held-out by RS3DAda's own split, and even that framing
requires ignoring that the *model itself* (not just that one tile) was
shaped by training on the other 49.

## 4. Load and inference attempt

**No CUDA patching needed to get a result — but only because a minimal
standalone script was written instead of running the official one
in-place.** Read `infer_height.py` first: it hardcodes `.cuda()` in 3 lines
(`model.cuda()` at line 83; `.cuda()` called twice more inside the
TTA/non-TTA input branches at lines 132/134) plus an unconditional
`CUDA_VISIBLE_DEVICES` env-var set — this would fail outright on a non-CUDA
machine (`AssertionError`/`RuntimeError` from `.cuda()` with no CUDA
backend). Also grepped `models/*.py`, `utils/`, `dataset/` for `.cuda()` /
`xformers` / `flash_attn` — **zero hits**; the CUDA-hardcoding is confined
to the inference script's own code, not the model architecture.

Given that, wrote `scripts/rs3dada_mps_smoketest.py` — a ~60-line script
that imports the same `DPT_DINOv2` model class and the same
`Normalize`+`ToTensorV2` preprocessing pipeline, but uses
`device = mps if available else cpu` and reads the input tile via
`rasterio` (avoiding a `gdal`/`osgeo` dependency that isn't installed and
wasn't needed for a single-tile shape check). This is not "patching"
`infer_height.py` — it's a fresh, minimal script reusing their exact model
+ checkpoint + preprocessing, which is a smaller, faster path to the same
feasibility answer.

**Other environment notes surfaced along the way:**
- `albumentations` wasn't installed; installed cleanly via `uv pip install`
  (the project's venv has no `pip` module — it's `uv`-managed; a plain
  `pip install` silently installed into the system Python instead of the
  venv, which is why the first attempt appeared to succeed but the import
  still failed — resolved by using `uv pip install --python .venv/bin/python3`).
- `DPT_DINOv2.__init__` calls `torch.hub.load('facebookresearch/dinov2', ...)`
  even with `pretrained=False` — this fetches the DINOv2 *architecture code*
  from GitHub (not weights), requiring one-time internet access; cached
  afterward at `~/.cache/torch/hub/`.
- DINOv2's own code emits `UserWarning: xFormers is not available` (for
  SwiGLU/Attention/Block) since `xformers` isn't installed — harmless,
  confirmed: it falls back to a standard (non-`xformers`) attention path
  automatically, no crash.

**Ran on `JAX_004_006`** (1022×1022 crop, matching `infer_height.py`'s own
default `--patch_size`), MPS device, forward pass took 0.9s:

```
load_state_dict: missing=0 unexpected=0   (checkpoint matches architecture exactly)
Output keys: ['regression', 'segmentation']
Pred shape: (1022, 1022)  dtype: float32
Finite: True   NaN: 0   Inf: 0
min=-1.03  max=15.16  mean=2.75  std=3.11
p1=-0.19  p50=1.25  p99=11.42
```

## 5. Answering the checklist directly

| Check | Result |
|---|---|
| Does it load? | **Yes.** `missing=0, unexpected=0` — the checkpoint's state dict matches the `DPT_DINOv2` architecture exactly, no shape mismatches, no partial load. |
| Correct output shape? | **Yes.** (1022, 1022) — matches the (H, W) of the input crop exactly, single-channel regression head as expected. |
| Finite values? | **Yes.** Zero NaN, zero Inf, across all 1,044,484 output pixels. |
| Plausible nDSM-like range? | **Yes, plausible.** Output is in meters, not a normalized [0,1]/[-1,1] range: mean 2.75 m, p50 1.25 m, p99 11.42 m, small negative values near zero (min -1.03 m) — the small negatives are normal regression noise around the ground plane, not a units/scaling bug. For context, this project's own DFC2019 AGL ground truth (same tile family) has mean ≈3.4 m, p50 ≈0.05 m, p95 ≈15 m across the 50-tile benchmark — RS3DAda's output distribution is in the same general order of magnitude and shape (heavy right skew, most mass near zero, a long tail up to double-digit meters). This is *not* evidence RS3DAda is accurate on our data — contamination (§3) means any accuracy number would be tainted anyway — it's only evidence the checkpoint is a genuine, correctly-wired height regressor, not a broken/mismatched load producing garbage.

## Recommendation

RS3DAda is technically usable (loads cleanly, runs on MPS without patching,
plausible output) but **not usable as a zero-shot benchmark comparison** —
49 of 50 tiles are training-set contamination for this specific model. If
RS3DAda is still worth including in this audit for any reason, it would
need to be reported explicitly as "trained with exposure to this benchmark
via UDA self-training," alongside whatever real zero-shot methods this
project is comparing against — never presented as a peer comparison to a
genuinely held-out method. Running the full 50-tile benchmark now, as
originally asked, would produce a number that looks like an apples-to-apples
comparison but isn't — recommend against running it without that caveat
attached everywhere the number is later used.

## 6. Follow-up — genuine zero-shot smoke test on Sentinel-2 India

**This section IS a genuine zero-shot test, unlike §§1-5 above.** Sentinel-2
India was never in SynRS3D's training domains (its own README lists real
and synthetic source datasets; India Sentinel-2 at 10m GSD is not among
them) — no contamination caveat needed here.

**Method:** `scripts/rs3dada_sentinel2_smoketest.py`. Same model class,
checkpoint, and preprocessing as §4 (unchanged) — only the input changed,
from JAX_004_006 to the 4 production Sentinel-2 tiles
(`data/sentinel2/{darjeeling,kolkata,bardhaman,sundarbans}/*.tif`, real 10m/
3-band/EPSG:32645 GeoTIFFs). One forward pass per region, center-cropped to
the largest multiple-of-14 patch that fits (1022px for the three larger
tiles, 994px for Darjeeling, whose tile is 1004×1118). A few-tiles smoke
test only, not a benchmark — no ground truth exists for these regions'
absolute elevation to score against (that capability was never
established, per this project's own CLAUDE.md).

**The specific thing being checked:** SynRS3D's training GSD is 0.05-1
m/pixel; Sentinel-2 is 10 m/pixel — 10-200x coarser than anything in the
training distribution. At 10m GSD, individual buildings (typically 5-30m
footprints) collapse to 1-3 pixels and the fine texture the height head
relies on is largely gone. The predicted failure mode was degenerate,
near-flat output — not a crash, since the model doesn't know its input is
out-of-domain, but a loss of any real signal.

**Results — all 4 regions ran cleanly, no NaN/Inf, output in a plausible
raw range:**

| Region | Patch | Finite | min | max | mean | std | p50 |
|---|---:|---|---:|---:|---:|---:|---:|
| Darjeeling | 994 | Yes | -1.154 | 2.739 | 0.087 | **0.200** | 0.075 |
| Kolkata | 1022 | Yes | -1.106 | 8.804 | 1.514 | 1.235 | 1.263 |
| Bardhaman | 1022 | Yes | -0.886 | 19.390 | 0.415 | 1.443 | 0.191 |
| Sundarbans | 1022 | Yes | -0.976 | 5.180 | 0.185 | **0.286** | 0.161 |

(For reference, RS3DAda-on-DFC2019 from §4, contaminated, ~0.3m GSD: mean
2.75, p50 1.25, p99 11.42, min -1.03 — shown only as a second data point on
what this checkpoint's output looks like at a GSD it was actually trained
near, not a target these Sentinel-2 numbers should match.)

**This IS the predicted degenerate-output failure mode, and it shows up in
exactly the place you'd want to check first: Darjeeling.** Darjeeling is
the one region with substantial real elevation (OpenTopography DSM range
~557-2478m, ~1900m of real relief across the tile per this project's own
CLAUDE.md) — and it produced the **flattest, lowest-magnitude output of all
four** (std 0.200, max 2.74m). Meanwhile Bardhaman — described in this
project's own CLAUDE.md as flat agricultural terrain — produced the
**highest max (19.39m) and highest std (1.443)** of any region. That's
backwards from what a working elevation signal would look like: the tile
with the most real vertical relief gave the flattest prediction, and a flat
tile gave the most exaggerated one. This is consistent with the model
responding to input texture/contrast (field boundaries, urban block
patterns) that happens to survive at 10m GSD, rather than to any real
elevation structure the input no longer contains at that resolution — not
proof of the exact mechanism, since there's no ground truth here to check
predictions against directly, but a clear, direction-consistent signature
of the GSD-mismatch failure mode predicted going in, not an ambiguous
result.

**Recommendation, extending §"Recommendation" above:** RS3DAda is not a
viable zero-shot source for this project's actual domain (Sentinel-2, 10m
GSD) independent of the DFC2019 contamination problem — even with a clean,
genuinely held-out test, the GSD gap alone produces output that inverts the
relationship it should have with real terrain relief. This closes the loop
on RS3DAda for this project: contaminated on the one domain where it's
plausible (§§1-5), degenerate on the one domain where it's genuinely
zero-shot (§6).
