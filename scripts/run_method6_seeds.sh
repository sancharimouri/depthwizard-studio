#!/bin/bash
# C3 (2026-09-23): 2 extra seeds of the adopted Method 6 recipe, sequential. Training only here.
cd /Users/anweshasaha/projects/DepthWizard2
for SEED in 43 44; do
  OUT=data/dfc2019/experiments/method6_height_balanced_seed${SEED}
  mkdir -p "$OUT"
  .venv/bin/python -u scripts/evaluate_method6_gsd_film_height_balanced.py \
    --outdir "$OUT" --tag m6_heightbal_seed${SEED} --enable-height-balanced --epochs 12 \
    --seed ${SEED} --save-checkpoints > "$OUT/train.log" 2>&1
  echo "seed ${SEED} exit $? at $(date)" >> data/dfc2019/experiments/method6_seeds_status.txt
done
