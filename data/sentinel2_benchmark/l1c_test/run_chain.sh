#!/bin/bash
# L1C vs L2A test (2026-09-24, 16 tiles): fetch the remaining tiles, then 4 arms, then the pre-registered comparison.
cd /Users/anweshasaha/projects/DepthWizard2
PY=.venv/bin/python; S=scripts/s2_l1c_test.py
echo "=== fetch start $(date)"; $PY -u $S fetch || echo "!!! fetch FAILED"
for a in l2a_gain l1c_gain l2a_stretch l1c_stretch; do echo "=== $a start $(date)"; $PY -u $S run --arm $a || echo "!!! $a FAILED"; done
echo "=== compare $(date)"; $PY -u $S compare || echo "!!! compare FAILED"
echo "=== L1C CHAIN DONE $(date)"
