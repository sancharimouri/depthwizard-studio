#!/bin/bash
# Sentinel-2 token-grid test, Phase B (2026-09-24): R at 10 m and 12 m, extend the eval matrix, re-run the analysis.
cd /Users/anweshasaha/projects/DepthWizard2
PY=.venv/bin/python; S=scripts/method6_resolution_transfer.py
for g in 10 12; do echo "=== train R $g m start $(date)"; $PY -u $S train --gsd $g --proto R || echo "!!! train R $g FAILED"; done
echo "=== eval start $(date)"; $PY -u $S eval --train-gsds 10 12 || echo "!!! eval FAILED"
echo "=== analyze start $(date)"; $PY -u $S analyze || echo "!!! analyze FAILED"
echo "=== CHAINB DONE $(date)"
