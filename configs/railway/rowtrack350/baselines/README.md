# rowtrack350 baselines

This directory stores frozen cross-scene regression baselines for the railway rowtrack350 mainline.

- `rowtrack350_mainline_20260526.csv`: current 6-scene mainline baseline. It uses the rowtrack350 metric-width setup and records the exact result folder used for each scene.

Use `scripts/compare_trajectory_regression.py` to compare future experiments against this baseline. Do not update the baseline just because a single sequence improves; a new baseline should only be promoted after a full cross-scene comparison and a documented method change.
