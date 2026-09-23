#!/bin/bash
# Resolution-transfer chain (2026-09-24): Phase 2 training grid -> Phase 3 eval -> Phase 4 analysis.
cd /Users/anweshasaha/projects/DepthWizard2
PY=.venv/bin/python; S=scripts/method6_resolution_transfer.py
for proto in P R; do for g in 2 3 5; do
  echo "=== train $proto $g m start $(date)"
  $PY -u $S train --gsd $g --proto $proto || echo "!!! train $proto $g FAILED"
done; done
echo "=== eval start $(date)"; $PY -u $S eval || echo "!!! eval FAILED"
echo "=== analyze start $(date)"; $PY -u $S analyze || echo "!!! analyze FAILED"
echo "=== CHAIN DONE $(date)"
