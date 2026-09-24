#!/bin/bash
# Sentinel-2 token-grid test, Phase C (2026-09-24): waits for Phase B's chain, then P arm, R arm, analysis.
cd /Users/anweshasaha/projects/DepthWizard2
until grep -q "CHAINB DONE" data/dfc2019/experiments/resolution_transfer/chainB.log; do sleep 30; done
PY=.venv/bin/python; S=scripts/s2_token_grid_phase_c.py
for a in P R; do echo "=== train $a start $(date)"; $PY -u $S train --arm $a --steps 600 --batch 4 || echo "!!! train $a FAILED"; done
echo "=== analyze start $(date)"; $PY -u $S analyze || echo "!!! analyze FAILED"
echo "=== PHASE C DONE $(date)"
