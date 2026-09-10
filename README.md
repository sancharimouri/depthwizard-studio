# DepthWizard2
SIH 2026 — Single-View Height Estimation and 3D Flythrough.

Clean rebuild scaffold. Implementation is intentionally deferred.
Planned architecture: Satellite RGB → frozen DAv2 relative-depth prior → RDAH-Net-style nDSM fusion → sparse-anchor/LoRA correction → CartoDEM terrain → absolute DSM → validation/failure detector → Three.js visualization.

Terrain and nDSM remain separate products. Demo mode is replaceable and must not masquerade as validated ML.
