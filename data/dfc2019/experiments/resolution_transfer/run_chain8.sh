#!/bin/bash
# Phase 5 (optional, 2026-09-24): 8 m training (P, R), then extend the eval matrix and re-run the analysis.
cd /Users/anweshasaha/projects/DepthWizard2
PY=.venv/bin/python; S=scripts/method6_resolution_transfer.py
for proto in P R; do echo "=== train $proto 8 m start $(date)"; $PY -u $S train --gsd 8 --proto $proto || echo "!!! train $proto 8 FAILED"; done
echo "=== eval start $(date)"; $PY -u $S eval --train-gsds 2 3 5 8 || echo "!!! eval FAILED"
echo "=== analyze start $(date)"; $PY -u $S analyze || echo "!!! analyze FAILED"
echo "=== CHAIN8 DONE $(date)"
