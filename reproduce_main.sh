#!/usr/bin/env bash
set -e
python3 plot_from_csv.py --csv S1_data.csv --out figures/diagnostic_pattern_trajectory.png
echo "Figure -> figures/diagnostic_pattern_trajectory.png"
