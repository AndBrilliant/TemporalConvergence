# Temporal Convergence — minimal reproduction

This repo includes the minimal artifact to reproduce the diagnostic figure:
- `S1_data.csv` — exact values used (FLAG/PDG-derived)
- `plot_from_csv.py` — reads the CSV and writes a PNG to `figures/`
- `diagnostic_pattern_trajectory.py` — original analysis script

## Reproduce
```bash
python3 plot_from_csv.py --csv S1_data.csv --out figures/diagnostic_pattern_trajectory.png
```

Or use the wrapper:
```bash
./reproduce_main.sh
```

## Requirements
```bash
pip install -r requirements.txt
```

## Citation
See CITATION.cff for citation information.
