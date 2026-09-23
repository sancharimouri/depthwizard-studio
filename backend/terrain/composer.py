# DEM terrain + nDSM -> absolute DSM.
"""
Compose an absolute digital surface model from a bare-earth terrain model and a
predicted height-above-ground (nDSM / AGL) raster on the same grid:

    DSM = DTM + max(AGL, 0)

The terrain input must be a BARE-EARTH model (e.g. FABDEM). Adding AGL to a surface
model such as Copernicus GLO-30 double-counts canopy and buildings (see
docs/method-audit/final-comparison.md §3.1, A3). Both inputs are expected in the same
vertical datum (FABDEM and GLO-30 are EGM2008 orthometric); the output inherits it.

Negative AGL predictions are clipped to 0 for the DSM (a surface cannot lie below
the ground), but the fraction clipped is reported so it is never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ComposedDSM:
    dsm: np.ndarray            # absolute surface elevation, m (datum of `dtm`)
    agl_used: np.ndarray       # AGL actually added (clipped at 0), m
    frac_negative_agl: float   # fraction of valid pixels whose raw AGL < 0
    frac_valid: float          # fraction of pixels with finite DTM and AGL


def compose_dsm(dtm: np.ndarray, agl: np.ndarray, valid: np.ndarray | None = None) -> ComposedDSM:
    """DSM = DTM + max(AGL, 0) on a shared grid. Pixels outside `valid` (or with a
    non-finite input) are NaN in the output."""
    if dtm.shape != agl.shape:
        raise ValueError(f"DTM {dtm.shape} and AGL {agl.shape} must share a grid")
    ok = np.isfinite(dtm) & np.isfinite(agl)
    if valid is not None:
        ok &= valid
    agl_used = np.where(ok, np.maximum(agl, 0.0), np.nan).astype(np.float32)
    dsm = np.where(ok, dtm + agl_used, np.nan).astype(np.float32)
    n = int(ok.sum())
    return ComposedDSM(dsm=dsm, agl_used=agl_used,
                       frac_negative_agl=float((agl[ok] < 0).sum() / n) if n else float("nan"),
                       frac_valid=float(ok.mean()))
