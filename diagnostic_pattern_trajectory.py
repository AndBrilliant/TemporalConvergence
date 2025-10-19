import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
Criterion 4 Temporal Validation: Diagnostic Pattern Trajectory Analysis

Quantitative assessment of directional convergence using FLAG review data.
Demonstrates systematic offset revealed through precision improvement.
"""

# ============================================================================
# FLAG REVIEW DATA
# ============================================================================

years = np.array([2019, 2021, 2024])
measurements = np.array([2.16, 2.17, 2.162])
uncertainties = np.array([0.08, 0.07, 0.05])
prediction = 10**(1/3)  # 2.154

sigma_deviations = np.abs(measurements - prediction) / uncertainties

print("=" * 70)
print("DIAGNOSTIC PATTERN: FLAG REVIEW DATA ANALYSIS")
print("=" * 70)
print(f"Predicted value: m_d/m_u = {prediction:.4f}")
print()
for i, year in enumerate(years):
    print(f"FLAG {year}: {measurements[i]:.3f} ± {uncertainties[i]:.3f} "
          f"(deviation: {sigma_deviations[i]:.3f}σ)")
print()

# ============================================================================
# TEMPORAL TREND ANALYSIS
# ============================================================================

weights = 1/uncertainties**2
slope, intercept, r_value, p_value, std_err = stats.linregress(years, measurements)
projected_value_2050 = slope * 2050 + intercept

print("=" * 70)
print("LINEAR TREND ANALYSIS")
print("=" * 70)
print(f"Slope: {slope:.6f} year^-1")
print(f"Extrapolated value (2050): {projected_value_2050:.4f}")
print(f"Deviation from prediction: {abs(projected_value_2050 - prediction):.4f}")
print()

# ============================================================================
# PRECISION EVOLUTION ANALYSIS
# ============================================================================

uncertainty_reduction = (uncertainties[0] - uncertainties[-1]) / uncertainties[0]
sigma_ratio = sigma_deviations[-1] / sigma_deviations[0]

print("=" * 70)
print("PRECISION IMPROVEMENT ANALYSIS")
print("=" * 70)
print(f"Relative uncertainty reduction: {100*uncertainty_reduction:.1f}%")
print(f"Statistical significance evolution: {sigma_deviations[0]:.3f}σ → {sigma_deviations[-1]:.3f}σ")
print(f"Ratio: {sigma_ratio:.2f}")
print()

if sigma_ratio > 1:
    print("Note: Statistical significance increased despite uncertainty reduction.")
    print("Interpretation: Measurements converging toward offset value.")
print()

# ============================================================================
# FUTURE PROJECTIONS
# ============================================================================

future_years = np.array([2027, 2030, 2033, 2036])
future_measurements = slope * future_years + intercept

# Project uncertainties (simple linear decay, more conservative)
# Lattice QCD uncertainties don't drop exponentially, they reduce gradually
uncertainty_rate = (uncertainties[-1] - uncertainties[0]) / (years[-1] - years[0])
future_uncertainties = uncertainties[-1] + uncertainty_rate * (future_years - years[-1])
# Floor at reasonable minimum (lattice won't reach arbitrary precision)
future_uncertainties = np.maximum(future_uncertainties, 0.02)

future_sigma = np.abs(future_measurements - prediction) / future_uncertainties

print("=" * 70)
print("PROJECTED TEMPORAL EVOLUTION")
print("=" * 70)
for i, year in enumerate(future_years):
    print(f"FLAG {year} (projected): {future_measurements[i]:.3f} ± {future_uncertainties[i]:.3f} "
          f"(deviation: {future_sigma[i]:.2f}σ)")
print()

# ============================================================================
# CRITERION 4 ASSESSMENT (Two-Part Test)
# ============================================================================

# Part A: 2/3 Rule (Transition Counting)
def two_thirds_rule(sigma_values):
    N = len(sigma_values)
    if N < 3:
        return False, 0, 0
    transitions = N - 1
    converging = sum(1 for i in range(1, N) if sigma_values[i] < sigma_values[i-1])
    required = int(np.ceil(2 * transitions / 3))
    passes = converging >= required
    return passes, converging, transitions

passes_23, conv, total = two_thirds_rule(sigma_deviations)

# Part B: Directional Slope Test
slope_sigma, intercept_sigma, _, _, _ = stats.linregress(years, sigma_deviations)
passes_slope = slope_sigma < 0  # Must be decreasing

print("=" * 70)
print("CRITERION 4 ASSESSMENT (Two-Part Test)")
print("=" * 70)
print(f"Part A - 2/3 Rule:")
print(f"  Converging transitions: {conv}/{total}")
print(f"  Required threshold: {int(np.ceil(2*total/3))}/{total}")
print(f"  Status: {'Pass' if passes_23 else 'Fail'}")
print()
print(f"Part B - Directional Slope Test:")
print(f"  Linear regression slope: {slope_sigma:.6f} σ/year")
print(f"  Trend: {'Converging (σ decreasing)' if slope_sigma < 0 else 'Diverging (σ increasing)'}")
print(f"  Status: {'Pass' if passes_slope else 'Fail'}")
print()
print(f"OVERALL CRITERION 4: {'Pass' if (passes_23 and passes_slope) else 'Fail'}")
print("=" * 70)
print()

# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30, 
                      left=0.08, right=0.96, top=0.94, bottom=0.08)

# ============================================================================
# Panel A: Temporal Measurements
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

ax1.axhline(y=prediction, color='#CC0000', linestyle='--', linewidth=2, 
            label=f'Predicted: {prediction:.4f}', zorder=1)

ax1.errorbar(years, measurements, yerr=uncertainties, fmt='o', 
            markersize=9, capsize=5, capthick=2, linewidth=2,
            color='#003366', label='FLAG data', zorder=3, ecolor='#003366')

year_range = np.linspace(2018, 2025, 100)
trend_line = slope * year_range + intercept
ax1.plot(year_range, trend_line, '-', alpha=0.4, linewidth=1.5,
         color='#003366', zorder=2)

ax1.fill_between([2023, 2025], [2.155, 2.155], [2.169, 2.169],
                 alpha=0.15, color='#003366', zorder=0)
ax1.text(2024, 2.169, 'Convergence\nregion', fontsize=9, ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.8))

ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('$m_d/m_u$', fontsize=11)
ax1.set_title('(a) Temporal evolution of FLAG determinations', 
              fontsize=11, loc='left', pad=10)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.95)
ax1.grid(True, alpha=0.2, linestyle=':')
ax1.set_xlim(2018, 2025)
ax1.set_ylim(2.14, 2.20)

# ============================================================================
# Panel B: Statistical Significance Evolution
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

ax2.plot(years, sigma_deviations, 'o-', markersize=10, linewidth=2.5,
         color='#8B0000', label='Observed deviation')

for i, (year, sigma) in enumerate(zip(years, sigma_deviations)):
    ax2.annotate(f'{sigma:.3f}σ', xy=(year, sigma), 
                xytext=(0, 8 if i != 1 else -12), textcoords='offset points',
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8))

ax2.axhline(y=1.0, color='#FF8C00', linestyle=':', linewidth=1.5, alpha=0.6,
            label='1σ reference')

ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Statistical significance (σ)', fontsize=11)
ax2.set_title('(b) Precision-normalized deviation magnitude', 
              fontsize=11, loc='left', pad=10)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.95)
ax2.grid(True, alpha=0.2, linestyle=':')
ax2.set_ylim(0, 0.28)

textstr = (f'Uncertainty reduction: {100*uncertainty_reduction:.0f}%\n'
           f'Deviation increase: {sigma_ratio:.1f}×\n'
           'Interpretation: Systematic\noffset from prediction')
props = dict(boxstyle='round,pad=0.5', facecolor='#FFF8DC', alpha=0.9, edgecolor='gray')
ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right', bbox=props)

# ============================================================================
# Panel C: Forward Projection
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])

all_years = np.concatenate([years, future_years])
all_measurements = np.concatenate([measurements, future_measurements])
all_uncertainties = np.concatenate([uncertainties, future_uncertainties])

ax3.axhline(y=prediction, color='#CC0000', linestyle='--', linewidth=2,
            label=f'Predicted: {prediction:.4f}', zorder=1)

ax3.errorbar(years, measurements, yerr=uncertainties, fmt='o',
            markersize=9, capsize=5, capthick=2, linewidth=2,
            color='#003366', label='Historical', zorder=4, ecolor='#003366')

ax3.errorbar(future_years, future_measurements, yerr=future_uncertainties, 
            fmt='s', markersize=8, capsize=4, capthick=2, linewidth=1.5,
            color='#4682B4', linestyle='--', alpha=0.6,
            label='Projected', zorder=3, ecolor='#4682B4')

for i, year in enumerate(all_years):
    upper = all_measurements[i] + all_uncertainties[i]
    lower = all_measurements[i] - all_uncertainties[i]
    alpha_val = 0.2 if year in years else 0.1
    color = '#003366' if year in years else '#4682B4'
    ax3.fill_between([year-0.8, year+0.8], [upper, upper], [lower, lower],
                     color=color, alpha=alpha_val, zorder=0)

ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('$m_d/m_u$', fontsize=11)
ax3.set_title('(c) Projected trajectory assuming trend continuation', 
              fontsize=11, loc='left', pad=10)
ax3.legend(fontsize=9, loc='upper left', framealpha=0.95)
ax3.grid(True, alpha=0.2, linestyle=':')
ax3.set_xlim(2017, 2038)
ax3.set_ylim(2.10, 2.24)

# ============================================================================
# Panel D: Quantitative Summary
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

summary_text = f"""QUANTITATIVE TEMPORAL ANALYSIS

Pattern: 2(m_d/m_u)³ = m_s/m_d
Predicted: m_d/m_u = {prediction:.4f}

HISTORICAL MEASUREMENTS
  FLAG 2019: {measurements[0]:.3f}±{uncertainties[0]:.3f} ({sigma_deviations[0]:.3f}σ)
  FLAG 2021: {measurements[1]:.3f}±{uncertainties[1]:.3f} ({sigma_deviations[1]:.3f}σ)
  FLAG 2024: {measurements[2]:.3f}±{uncertainties[2]:.3f} ({sigma_deviations[2]:.3f}σ)

TEMPORAL CHARACTERISTICS
  Uncertainty reduction: {100*uncertainty_reduction:.1f}%
  Deviation evolution: {sigma_ratio:.2f}× increase
  Central convergence: ~{measurements[-1]:.3f}
  Offset from prediction: {abs(measurements[-1]-prediction):.4f}

CRITERION 4 ASSESSMENT
  Part A (2/3 rule): {conv}/{total} → {'Pass' if passes_23 else 'Fail'}
  Part B (Slope test): {slope_sigma:+.4f} σ/yr → {'Pass' if passes_slope else 'Fail'}
  Overall: {'Pass' if (passes_23 and passes_slope) else 'Fail'}

PROJECTED EVOLUTION
  2027: {future_measurements[0]:.3f}±{future_uncertainties[0]:.3f} ({future_sigma[0]:.2f}σ)
  2030: {future_measurements[1]:.3f}±{future_uncertainties[1]:.3f} ({future_sigma[1]:.2f}σ)
  2033: {future_measurements[2]:.3f}±{future_uncertainties[2]:.3f} ({future_sigma[2]:.2f}σ)
  2036: {future_measurements[3]:.3f}±{future_uncertainties[3]:.3f} ({future_sigma[3]:.2f}σ)

INTERPRETATION
Pattern exhibits directional divergence under
precision improvement. Measurements converge
toward systematically offset value rather than
predicted target. Both transition counting and
directional slope tests fail. Temporal behavior
consistent with numerical near-coincidence.
"""

ax4.text(0.05, 0.97, summary_text, transform=ax4.transAxes,
         fontsize=9.5, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#F5F5F5', 
                  alpha=0.95, edgecolor='gray', linewidth=1))

plt.savefig('diagnostic_pattern_trajectory.png', dpi=150, bbox_inches='tight')
print("Figure generated: diagnostic_pattern_trajectory.png")
print()
