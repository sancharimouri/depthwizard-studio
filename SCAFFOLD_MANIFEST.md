# Scaffold manifest

Included: complete source/config/test/docs skeleton for DAv2, RDAH-Net, training, LoRA/anchor adaptation, ICESat-2, CartoDEM, OpenTopography DEMs, calibration, validation, terrain composition, FastAPI and Three.js.

Not included:
- real satellite imagery/GeoTIFFs
- CartoDEM/COP30/AW3D30/SRTM tiles
- ICESat-2 ATL03/ATL08
- DFC2019/GAMUS/US3D training data
- DAv2 weights
- RDAH-Net checkpoints
- LoRA checkpoints
- generated .npy/.npz/.tif/.png/.jpg/mesh artifacts
- API keys/secrets
- .git metadata
- uv.lock (dependencies not resolved yet)
- Excel workbook: none is required by the current architecture; CSV/JSON templates are provided instead.

Binary outputs are represented with `.placeholder` markers rather than invalid fake binary files.
