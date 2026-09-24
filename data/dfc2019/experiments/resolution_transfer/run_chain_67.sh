#!/bin/bash
# Boundary sweep (2026-09-24): R at 6 m and 7 m, extend the eval matrix, re-run the analysis.
cd /Users/anweshasaha/projects/DepthWizard2
PY=.venv/bin/python; S=scripts/method6_resolution_transfer.py
for g in 6 7; do echo "=== train R $g m start $(date)"; $PY -u $S train --gsd $g --proto R || echo "!!! train R $g FAILED"; done
echo "=== eval start $(date)"; $PY -u $S eval --train-gsds 6 7 || echo "!!! eval FAILED"
echo "=== analyze start $(date)"; $PY -u $S analyze || echo "!!! analyze FAILED"
echo "=== CHAIN67 DONE $(date)"
