#!/usr/bin/env python3
import argparse, csv, os
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description="Plot Temporal Convergence diagnostic from CSV.")
p.add_argument("--csv", default="S1_data.csv")
p.add_argument("--out", default="figures/diagnostic_pattern_trajectory.png")
args = p.parse_args()

rows = []
with open(args.csv, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        year = int(row["year"])
        md_mu = float(row["md_over_mu"])
        sig   = float(row["md_over_mu_sigma"])
        pred  = float(row.get("md_over_mu_pred", 2.154))
        delta = abs(md_mu - pred) / sig if sig else float("nan")
        rows.append((year, row["release"], delta))

rows.sort(key=lambda t: t[0])
years  = [y for y,_,_ in rows]
labs   = [l for _,l,_ in rows]
deltas = [d for *_,d in rows]

os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.figure(figsize=(6,4))
plt.plot(years, deltas, marker="o")
for x,y,lab in zip(years, deltas, labs):
    plt.annotate(lab, (x,y), textcoords="offset points", xytext=(4,4), fontsize=8)
plt.xlabel("Year")
plt.ylabel(r"$\Delta = |x - x_p| / \sigma$")
plt.title("Temporal Convergence Diagnostic (md/mu)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(args.out, dpi=200)
print(f"Saved {args.out}")
