"""
Deep dive into the spatial proximity failure.
Questions:
1. Is "Brain - Cortex" (catch-all region) corrupting distances?
2. Subcortical-only: does finding hold where mapping is cleanest?
3. Multiple regression: does WJ add variance BEYOND distance?
4. Propofol: does WJ predict unconscious FC after distance control?
5. Propofol CHANGE: does WJ predict WHICH connections survive anesthesia?
6. Residual analysis: which pairs deviate from distance in WJ's direction?
7. Distance doesn't change under propofol — so WHY does WJ-FC get stronger?
"""
import os, numpy as np, pandas as pd, json
from scipy import stats
from scipy.stats import rankdata, pearsonr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = r'G:\My Drive\inner_architecture_research\gene_to_brain_wj'
RESULTS = os.path.join(BASE, 'results')
FIGURES = os.path.join(BASE, 'figures')

# Load bulletproof pairs
df = pd.read_csv(os.path.join(RESULTS, 'bulletproof_pairs.csv'))
print(f"Total pairs: {len(df)}")
print(f"Regions: {sorted(set(df['Region1'].tolist() + df['Region2'].tolist()))}")

def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    sx = np.polyfit(rz, rx, 1); sy = np.polyfit(rz, ry, 1)
    return pearsonr(rx - np.polyval(sx, rz), ry - np.polyval(sy, rz))

# ================================================================
print(f"\n{'='*70}")
print("1. REMOVE 'Brain - Cortex' CATCH-ALL")
print(f"{'='*70}")
# "Brain - Cortex" is a catch-all that overlaps with Frontal and ACC.
# Its centroid (0,0,40) is arbitrary. Its FC mapping covers ALL 400
# cortical parcels which includes Frontal and ACC parcels.

df_no_ctx = df[(df['Region1'] != 'Brain - Cortex') &
               (df['Region2'] != 'Brain - Cortex')]
print(f"Pairs without 'Brain - Cortex': {len(df_no_ctx)}")

r1, p1 = stats.spearmanr(df_no_ctx['WJ'], df_no_ctx['FC_Awake'])
print(f"WJ vs FC (no Cortex): rho={r1:.4f}, p={p1:.6f}")

r1d, p1d = partial_spearman(df_no_ctx['WJ'].values,
                             df_no_ctx['FC_Awake'].values,
                             df_no_ctx['Distance_mm'].values)
print(f"WJ vs FC (no Cortex, dist controlled): rho={r1d:.4f}, p={p1d:.6f}")

# ================================================================
print(f"\n{'='*70}")
print("2. SUBCORTICAL-ONLY ANALYSIS")
print(f"{'='*70}")
# Subcortical structures have: exact parcel mapping, well-defined centroids,
# less experience-dependent plasticity.

subcortical = ['Brain - Amygdala', 'Brain - Caudate (basal ganglia)',
               'Brain - Hippocampus', 'Brain - Hypothalamus',
               'Brain - Nucleus accumbens (basal ganglia)',
               'Brain - Putamen (basal ganglia)', 'Brain - Substantia nigra']

df_sub = df[df['Region1'].isin(subcortical) & df['Region2'].isin(subcortical)]
print(f"Subcortical pairs: {len(df_sub)}")

r2, p2 = stats.spearmanr(df_sub['WJ'], df_sub['FC_Awake'])
print(f"WJ vs FC (subcortical only): rho={r2:.4f}, p={p2:.6f}")

r2d, p2d = partial_spearman(df_sub['WJ'].values,
                             df_sub['FC_Awake'].values,
                             df_sub['Distance_mm'].values)
print(f"WJ vs FC (subcortical, dist controlled): rho={r2d:.4f}, p={p2d:.6f}")

# Also expression
r2e, p2e = stats.spearmanr(df_sub['Expr_Sim'], df_sub['FC_Awake'])
print(f"Expr vs FC (subcortical only): rho={r2e:.4f}, p={p2e:.6f}")

# ================================================================
print(f"\n{'='*70}")
print("3. MULTIPLE REGRESSION: FC ~ Distance + WJ")
print(f"{'='*70}")
# Does WJ add variance BEYOND distance?

from numpy.linalg import lstsq

# Standardize
X = np.column_stack([
    (df['Distance_mm'].values - df['Distance_mm'].mean()) / df['Distance_mm'].std(),
    (df['WJ'].values - df['WJ'].mean()) / df['WJ'].std(),
    np.ones(len(df))
])
y = (df['FC_Awake'].values - df['FC_Awake'].mean()) / df['FC_Awake'].std()

# Distance only model
X_dist = np.column_stack([X[:, 0], X[:, 2]])
beta_dist, resid_dist, _, _ = lstsq(X_dist, y, rcond=None)
r2_dist = 1 - np.sum((y - X_dist @ beta_dist)**2) / np.sum(y**2)

# Distance + WJ model
beta_full, resid_full, _, _ = lstsq(X, y, rcond=None)
r2_full = 1 - np.sum((y - X @ beta_full)**2) / np.sum(y**2)

# F-test for WJ contribution
n = len(y); k1 = 1; k2 = 2
ss_dist = np.sum((y - X_dist @ beta_dist)**2)
ss_full = np.sum((y - X @ beta_full)**2)
f_stat = ((ss_dist - ss_full) / (k2 - k1)) / (ss_full / (n - k2 - 1))
from scipy.stats import f as f_dist_scipy
p_f = 1 - f_dist_scipy.cdf(f_stat, k2 - k1, n - k2 - 1)

print(f"R-squared (distance only): {r2_dist:.4f}")
print(f"R-squared (distance + WJ): {r2_full:.4f}")
print(f"Delta R-squared: {r2_full - r2_dist:.4f}")
print(f"F-test for WJ contribution: F={f_stat:.3f}, p={p_f:.4f}")
print(f"Standardized beta (distance): {beta_full[0]:.3f}")
print(f"Standardized beta (WJ):       {beta_full[1]:.3f}")
if p_f < 0.05:
    print(">>> WJ adds SIGNIFICANT variance beyond distance")
else:
    print(">>> WJ does NOT add significant variance beyond distance")

# ================================================================
print(f"\n{'='*70}")
print("4. PROPOFOL: WJ vs UNCONSCIOUS FC (distance controlled)")
print(f"{'='*70}")

r4, p4 = stats.spearmanr(df['WJ'], df['FC_Unconscious'])
print(f"WJ vs FC_unconscious (raw): rho={r4:.4f}, p={p4:.6f}")

r4d, p4d = partial_spearman(df['WJ'].values,
                             df['FC_Unconscious'].values,
                             df['Distance_mm'].values)
print(f"WJ vs FC_unconscious (dist controlled): rho={r4d:.4f}, p={p4d:.6f}")

if p4d < 0.05:
    print(">>> WJ predicts UNCONSCIOUS FC even after distance control!")
else:
    print(">>> WJ does not predict unconscious FC after distance control")

# ================================================================
print(f"\n{'='*70}")
print("5. PROPOFOL CHANGE: What predicts WHICH connections survive?")
print(f"{'='*70}")
# FC_change = FC_unconscious - FC_awake
# If WJ predicts FC_change after distance control, WJ captures something
# about propofol vulnerability that distance alone doesn't explain.

df['FC_Change'] = df['FC_Unconscious'] - df['FC_Awake']
df['FC_Preservation'] = df['FC_Unconscious'] / df['FC_Awake']

r5a, p5a = stats.spearmanr(df['WJ'], df['FC_Change'])
print(f"WJ vs FC_change (raw): rho={r5a:.4f}, p={p5a:.6f}")

r5b, p5b = stats.spearmanr(df['Distance_mm'], df['FC_Change'])
print(f"Distance vs FC_change: rho={r5b:.4f}, p={p5b:.6f}")

r5c, p5c = partial_spearman(df['WJ'].values,
                             df['FC_Change'].values,
                             df['Distance_mm'].values)
print(f"WJ vs FC_change (dist controlled): rho={r5c:.4f}, p={p5c:.6f}")

r5d, p5d = stats.spearmanr(df['Expr_Sim'], df['FC_Change'])
print(f"Expr vs FC_change: rho={r5d:.4f}, p={p5d:.6f}")

if p5c < 0.05:
    print(">>> WJ predicts propofol-induced FC changes BEYOND distance!")
    print(">>> This is a DISTANCE-INDEPENDENT finding about consciousness.")
else:
    print(">>> WJ does not predict FC changes beyond distance")

# ================================================================
print(f"\n{'='*70}")
print("6. RESIDUAL ANALYSIS: Which pairs deviate from distance prediction?")
print(f"{'='*70}")
# Fit FC ~ distance, look at residuals, see if WJ explains them

slope, intercept = np.polyfit(df['Distance_mm'], df['FC_Awake'], 1)
df['FC_predicted_by_dist'] = slope * df['Distance_mm'] + intercept
df['FC_residual'] = df['FC_Awake'] - df['FC_predicted_by_dist']

r6, p6 = stats.spearmanr(df['WJ'], df['FC_residual'])
print(f"WJ vs FC_residual (after distance): rho={r6:.4f}, p={p6:.6f}")

# Which pairs have the largest positive residuals (stronger FC than distance predicts)?
df_sorted = df.sort_values('FC_residual', ascending=False)
print(f"\nTop 10 pairs with STRONGER FC than distance predicts:")
for _, row in df_sorted.head(10).iterrows():
    print(f"  {row['Label']:<45} residual={row['FC_residual']:+.3f} "
          f"WJ={row['WJ']:.4f} dist={row['Distance_mm']:.0f}mm")

print(f"\nTop 10 pairs with WEAKER FC than distance predicts:")
for _, row in df_sorted.tail(10).iterrows():
    print(f"  {row['Label']:<45} residual={row['FC_residual']:+.3f} "
          f"WJ={row['WJ']:.4f} dist={row['Distance_mm']:.0f}mm")

# ================================================================
print(f"\n{'='*70}")
print("7. WHY DOES WJ-FC GET STRONGER UNDER PROPOFOL?")
print(f"{'='*70}")
# Distance is constant. WJ is constant. Only FC changes.
# If WJ-FC increases under propofol, the FC that SURVIVES propofol
# is more aligned with molecular architecture than awake FC.
# This means experience-dependent modulation REDUCES the gene-brain
# correspondence. Removing it (via propofol) reveals the molecular
# architecture constraint more clearly.

print(f"WJ-FC awake:       rho={df[['WJ','FC_Awake']].corr('spearman').iloc[0,1]:.4f}")
print(f"WJ-FC unconscious: rho={df[['WJ','FC_Unconscious']].corr('spearman').iloc[0,1]:.4f}")

# Partial correlations controlling distance
r7a, p7a = partial_spearman(df['WJ'].values, df['FC_Awake'].values, df['Distance_mm'].values)
r7b, p7b = partial_spearman(df['WJ'].values, df['FC_Unconscious'].values, df['Distance_mm'].values)

print(f"WJ-FC awake (dist ctrl):       rho={r7a:.4f}, p={p7a:.6f}")
print(f"WJ-FC unconscious (dist ctrl): rho={r7b:.4f}, p={p7b:.6f}")

# Distance-FC correlations
r7c, p7c = stats.spearmanr(df['Distance_mm'], df['FC_Awake'])
r7d, p7d = stats.spearmanr(df['Distance_mm'], df['FC_Unconscious'])
print(f"\nDistance-FC awake:       rho={r7c:.4f}")
print(f"Distance-FC unconscious: rho={r7d:.4f}")
print(f"Distance explains {'MORE' if abs(r7d) > abs(r7c) else 'LESS'} "
      f"FC variance under propofol")

# ================================================================
print(f"\n{'='*70}")
print("8. CORTICAL-SUBCORTICAL CROSS-TYPE ANALYSIS")
print(f"{'='*70}")
# Cortical-subcortical pairs: these are spatially distant but
# functionally connected (thalamocortical, limbic-cortical).
# Do these follow the WJ prediction despite large distance?

cortical = ['Brain - Cortex', 'Brain - Frontal Cortex (BA9)',
            'Brain - Anterior cingulate cortex (BA24)']

df_cross = df[
    (df['Region1'].isin(cortical) & df['Region2'].isin(subcortical)) |
    (df['Region1'].isin(subcortical) & df['Region2'].isin(cortical))
]
print(f"Cortical-subcortical pairs: {len(df_cross)}")

r8, p8 = stats.spearmanr(df_cross['WJ'], df_cross['FC_Awake'])
print(f"WJ vs FC (cross-type): rho={r8:.4f}, p={p8:.6f}")

r8d, p8d = partial_spearman(df_cross['WJ'].values,
                             df_cross['FC_Awake'].values,
                             df_cross['Distance_mm'].values)
print(f"WJ vs FC (cross-type, dist ctrl): rho={r8d:.4f}, p={p8d:.6f}")

# ================================================================
print(f"\n{'='*70}")
print("SUMMARY: WHAT'S REAL, WHAT'S CONFOUNDED")
print(f"{'='*70}")

results = {
    'All pairs (raw)': (stats.spearmanr(df['WJ'], df['FC_Awake'])[0],
                        stats.spearmanr(df['WJ'], df['FC_Awake'])[1]),
    'All pairs (dist ctrl)': (r7a, p7a),
    'No Cortex catch-all': (r1, p1),
    'No Cortex (dist ctrl)': (r1d, p1d),
    'Subcortical only': (r2, p2),
    'Subcortical (dist ctrl)': (r2d, p2d),
    'Unconscious FC': (r4, p4),
    'Unconscious (dist ctrl)': (r4d, p4d),
    'FC change (propofol)': (r5a, p5a),
    'FC change (dist ctrl)': (r5c, p5c),
    'FC residual (after dist)': (r6, p6),
    'Cross-type pairs': (r8, p8),
    'Cross-type (dist ctrl)': (r8d, p8d),
}

for name, (rho, p) in results.items():
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f"  {name:<30} rho={rho:+.4f}  p={p:.6f}  {sig}")

# Save
df.to_csv(os.path.join(RESULTS, 'spatial_deep_dive.csv'), index=False, float_format='%.6f')

# Figure: key comparisons
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
colors = sns.color_palette('colorblind', 4)

# Panel 1: Raw WJ vs FC
ax = axes[0, 0]
ax.scatter(df['WJ'], df['FC_Awake'], s=50, alpha=0.7, c=[colors[0]], edgecolors='k', linewidth=0.5)
z = np.polyfit(df['WJ'], df['FC_Awake'], 1)
xl = np.linspace(df['WJ'].min(), df['WJ'].max(), 100)
ax.plot(xl, np.polyval(z, xl), 'r--', lw=2)
rho, p = stats.spearmanr(df['WJ'], df['FC_Awake'])
ax.set_title(f'(A) WJ vs FC (raw)\nrho={rho:.3f}, p={p:.4f}', fontweight='bold')
ax.set_xlabel('WJ'); ax.set_ylabel('FC')

# Panel 2: Distance vs FC
ax = axes[0, 1]
ax.scatter(df['Distance_mm'], df['FC_Awake'], s=50, alpha=0.7, c=[colors[1]], edgecolors='k', linewidth=0.5)
z = np.polyfit(df['Distance_mm'], df['FC_Awake'], 1)
xl = np.linspace(df['Distance_mm'].min(), df['Distance_mm'].max(), 100)
ax.plot(xl, np.polyval(z, xl), 'r--', lw=2)
rho, p = stats.spearmanr(df['Distance_mm'], df['FC_Awake'])
ax.set_title(f'(B) Distance vs FC\nrho={rho:.3f}, p={p:.4f}', fontweight='bold')
ax.set_xlabel('Distance (mm)'); ax.set_ylabel('FC')

# Panel 3: Distance vs WJ
ax = axes[0, 2]
ax.scatter(df['Distance_mm'], df['WJ'], s=50, alpha=0.7, c=[colors[2]], edgecolors='k', linewidth=0.5)
z = np.polyfit(df['Distance_mm'], df['WJ'], 1)
xl = np.linspace(df['Distance_mm'].min(), df['Distance_mm'].max(), 100)
ax.plot(xl, np.polyval(z, xl), 'r--', lw=2)
rho, p = stats.spearmanr(df['Distance_mm'], df['WJ'])
ax.set_title(f'(C) Distance vs WJ\nrho={rho:.3f}, p={p:.4f}', fontweight='bold')
ax.set_xlabel('Distance (mm)'); ax.set_ylabel('WJ')

# Panel 4: Awake vs Unconscious WJ-FC
ax = axes[1, 0]
ax.scatter(df['WJ'], df['FC_Awake'], s=50, alpha=0.5, c=[colors[0]], label='Awake')
ax.scatter(df['WJ'], df['FC_Unconscious'], s=50, alpha=0.5, c=[colors[3]], marker='s', label='Unconscious')
rho_a = stats.spearmanr(df['WJ'], df['FC_Awake'])[0]
rho_u = stats.spearmanr(df['WJ'], df['FC_Unconscious'])[0]
ax.set_title(f'(D) Awake rho={rho_a:.3f} vs Unconscious rho={rho_u:.3f}', fontweight='bold')
ax.set_xlabel('WJ'); ax.set_ylabel('FC'); ax.legend()

# Panel 5: WJ vs FC residual (after distance)
ax = axes[1, 1]
ax.scatter(df['WJ'], df['FC_residual'], s=50, alpha=0.7, c=[colors[0]], edgecolors='k', linewidth=0.5)
z = np.polyfit(df['WJ'], df['FC_residual'], 1)
xl = np.linspace(df['WJ'].min(), df['WJ'].max(), 100)
ax.plot(xl, np.polyval(z, xl), 'r--', lw=2)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_title(f'(E) WJ vs FC Residual (after distance)\nrho={r6:.3f}, p={p6:.4f}', fontweight='bold')
ax.set_xlabel('WJ'); ax.set_ylabel('FC Residual')

# Panel 6: WJ vs FC change (propofol effect)
ax = axes[1, 2]
ax.scatter(df['WJ'], df['FC_Change'], s=50, alpha=0.7, c=[colors[3]], edgecolors='k', linewidth=0.5)
z = np.polyfit(df['WJ'], df['FC_Change'], 1)
xl = np.linspace(df['WJ'].min(), df['WJ'].max(), 100)
ax.plot(xl, np.polyval(z, xl), 'r--', lw=2)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_title(f'(F) WJ vs Propofol FC Change\nrho={r5a:.3f}, p={p5a:.4f}', fontweight='bold')
ax.set_xlabel('WJ'); ax.set_ylabel('FC Change (Uncon - Awake)')

plt.suptitle('Spatial Deep Dive: Gene-to-Brain Cross-Scale Analysis',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'figure6_spatial_deep_dive.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved figure6_spatial_deep_dive.png")
