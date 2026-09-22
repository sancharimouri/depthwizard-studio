# RDAH-Net DFC2019 zero-shot artifacts (rescued 2026-09-23)

Rescued from a session scratchpad under /private/tmp (cleared on reboot). Scripts are in
`scripts/diag/`. All `*_stdout.txt` files and `swiss_zeroshot_fold_results.csv` are
**transcribed** from the session transcript, because those scripts printed their results
instead of saving them.

| File | Source script | Checkpoint |
|---|---|---|
| `scale_corr_stdout.txt` | `diag_rdah_scale_corr.py`: ×1–×1000 sweep on 4 probe tiles, where ×255 was picked | Track1 `104best_model.pth` (imports `CHECKPOINT` from `run_rdah_probe.py`) |
| `x255_calib_stdout.txt` | `diag_rdah_x255_calib.py`: ×255 + held-out affine on the same 4 tiles | Track1 |
| `x255_fullcv_stdout.txt`, `rdah_x255_zeroshot_fold_results.csv` (saved by the script) | `diag_rdah_x255_fullcv.py`: 50-tile tile-level 4-fold, the source of 2.231/4.566/0.716/0.655 | Track1 |
| `swiss_zeroshot_fold_results.csv`, `swiss_zeroshot_per_fold_stdout.txt` | `diag_swiss_zeroshot_per_fold.py`: quadrant protocol, FT-2's true-test halves | Swiss `swiss_best_model.pth` |
| `rdah_nested_variance_out.log` (saved by the script) | `diag_rdah_nested_variance.py`: FT-1 nested selection + variance ratio (05 summary §7–8) | FT-1 fold checkpoints |

Probe tiles: JAX_004_006, JAX_264_013 and OMA_248_029 are in `Track1-train.txt`; JAX_149_006 is
in `Track1-test.txt`.

Swiss per-fold scale re-derivation (from FT-2's training stdout in the same transcript; only
fold 0's full sweep line survives):
- fold 0: x1 +0.011, x30 +0.009, x65 +0.146, x100 +0.314, x150 +0.387, x200 +0.437,
  **x255 +0.469**, x300 +0.456, x500 +0.469, x1000 +0.436. Selected ×255.
- fold 1: ×255 (Pearson +0.4265)
- fold 2: ×300 (+0.4304)
- fold 3: ×300 (+0.4212). This fold is assigned by elimination: +0.4212 is the one selected
  value not labelled with a fold in the recovered text.
