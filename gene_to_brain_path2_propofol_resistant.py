"""
Pipeline: Path 2 — Propofol-resistant FC component analysis
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-03-27
Description: Computes propofol-resistant FC (the structurally constrained component
    of functional connectivity) and tests whether WJ predicts it better than raw
    awake FC. Three approaches:
    1. FC_resistant = min(FC_awake, FC_unconscious) per pair
    2. FC_resistant = mean(FC_awake, FC_unconscious) per pair
    3. FC_activity = FC_awake - FC_unconscious (the experience-dependent component)
    If WJ predicts FC_resistant but not FC_activity, the molecular architecture
    constrains the structural backbone, not the activity-dependent layer.
Dependencies: scipy, numpy, pandas
Input: bulletproof_pairs.csv, variance_weighted_pairs.csv
Output: results/strengthening_v2/propofol_resistant_results.json
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform

# ============================================================================
# CONFIG
# ============================================================================
BASE = r'G:\My Drive\inner_architecture_research\gene_to_brain_wj'
RESULTS = os.path.join(BASE, 'results')
OUT = os.path.join(RESULTS, 'strengthening_v2')
os.makedirs(OUT, exist_ok=True)

FORCE_RECOMPUTE = True
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_PERM = 10000

# ============================================================================
# LOAD
# ============================================================================
print("Loading data...")
pairs = pd.read_csv(os.path.join(RESULTS, 'bulletproof_pairs.csv'))
vw = pd.read_csv(os.path.join(RESULTS, 'variance_weighting', 'variance_weighted_pairs.csv'))
pairs['Pair_type'] = vw['Pair_type']

# Compute derived FC measures
pairs['FC_resistant_min'] = pairs[['FC_Awake', 'FC_Unconscious']].min(axis=1)
pairs['FC_resistant_mean'] = pairs[['FC_Awake', 'FC_Unconscious']].mean(axis=1)
pairs['FC_activity'] = pairs['FC_Awake'] - pairs['FC_Unconscious']

print(f"  Pairs: {len(pairs)}")
print(f"  FC_Awake range: [{pairs['FC_Awake'].min():.3f}, {pairs['FC_Awake'].max():.3f}]")
print(f"  FC_Unconscious range: [{pairs['FC_Unconscious'].min():.3f}, {pairs['FC_Unconscious'].max():.3f}]")
print(f"  FC_resistant_min range: [{pairs['FC_resistant_min'].min():.3f}, {pairs['FC_resistant_min'].max():.3f}]")
print(f"  FC_activity range: [{pairs['FC_activity'].min():.3f}, {pairs['FC_activity'].max():.3f}]")

# ============================================================================
# HELPERS
# ============================================================================
wj_matrix = pd.read_csv(os.path.join(RESULTS, 'region_coexpression_wj_matrix.csv'), index_col=0)
regions = sorted(wj_matrix.index.tolist())
n_reg = len(regions)
region_idx = {r: i for i, r in enumerate(regions)}

def build_matrix(pairs_df, col):
    mat = np.zeros((n_reg, n_reg))
    for _, row in pairs_df.iterrows():
        i = region_idx.get(row['Region1'])
        j = region_idx.get(row['Region2'])
        if i is not None and j is not None:
            mat[i, j] = row[col]
            mat[j, i] = row[col]
    return mat

def mantel_test(vec_a, vec_b, n_perm, seed=42):
    rng = np.random.RandomState(seed)
    n = int(0.5 * (1 + np.sqrt(1 + 8 * len(vec_a))))
    observed_r, _ = stats.pearsonr(vec_a, vec_b)
    observed_rho, _ = stats.spearmanr(vec_a, vec_b)
    mat_a = squareform(vec_a)
    np.fill_diagonal(mat_a, 0)
    null_r = np.zeros(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n)
        shuffled = mat_a[np.ix_(perm, perm)]
        shuffled_vec = shuffled[np.triu_indices(n, k=1)]
        null_r[p], _ = stats.pearsonr(shuffled_vec, vec_b)
    return {
        'pearson_r': float(observed_r),
        'spearman_rho': float(observed_rho),
        'p_perm': float(np.mean(null_r >= observed_r)),
        'null_mean': float(np.mean(null_r)),
        'null_std': float(np.std(null_r)),
    }

tri = np.triu_indices(n_reg, k=1)
wj_vec = wj_matrix.values[tri]

# ============================================================================
# 1. MANTEL TESTS — WJ vs each FC variant
# ============================================================================
print("\n" + "="*60)
print("1. MANTEL TESTS — WJ vs FC variants (all pairs)")
print("="*60)

fc_variants = {
    'FC_Awake': 'FC_Awake',
    'FC_Unconscious': 'FC_Unconscious',
    'FC_resistant_min': 'FC_resistant_min',
    'FC_resistant_mean': 'FC_resistant_mean',
    'FC_activity': 'FC_activity',
}

mantel_results = {}
for label, col in fc_variants.items():
    mat = build_matrix(pairs, col)
    vec = mat[tri]
    result = mantel_test(wj_vec, vec, N_PERM)
    mantel_results[label] = result
    sig = "***" if result['p_perm'] < 0.001 else "**" if result['p_perm'] < 0.01 else "*" if result['p_perm'] < 0.05 else "ns"
    print(f"\n  WJ vs {label}:")
    print(f"    Pearson r = {result['pearson_r']:.4f}")
    print(f"    Spearman rho = {result['spearman_rho']:.4f}")
    print(f"    Mantel p = {result['p_perm']:.4f} {sig}")

# ============================================================================
# 2. SPEARMAN CORRELATIONS — by pair type
# ============================================================================
print("\n" + "="*60)
print("2. SPEARMAN — WJ vs FC variants by pair type")
print("="*60)

spearman_results = {}
for pair_type in ['all', 'cross-type', 'within-subcortical', 'within-cortical']:
    subset = pairs if pair_type == 'all' else pairs[pairs['Pair_type'] == pair_type]
    n = len(subset)
    if n < 5:
        continue

    spearman_results[pair_type] = {'n': n}
    print(f"\n  --- {pair_type} (n={n}) ---")

    for label, col in fc_variants.items():
        rho, p = stats.spearmanr(subset['WJ'], subset[col])
        spearman_results[pair_type][label] = {'rho': float(rho), 'p': float(p)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    WJ vs {label}: rho={rho:.4f}, p={p:.4f} {sig}")

    # Partial correlations controlling distance
    for label, col in [('FC_resistant_min', 'FC_resistant_min'),
                        ('FC_resistant_mean', 'FC_resistant_mean'),
                        ('FC_Unconscious', 'FC_Unconscious')]:
        # Residualize on distance
        if len(subset) > 5:
            slope_x, int_x, _, _, _ = stats.linregress(subset['Distance_mm'], subset['WJ'])
            slope_y, int_y, _, _, _ = stats.linregress(subset['Distance_mm'], subset[col])
            resid_x = subset['WJ'] - (slope_x * subset['Distance_mm'] + int_x)
            resid_y = subset[col] - (slope_y * subset['Distance_mm'] + int_y)
            rho_partial, p_partial = stats.spearmanr(resid_x, resid_y)
            spearman_results[pair_type][f'{label}_partial_dist'] = {
                'rho': float(rho_partial), 'p': float(p_partial)
            }
            sig = "***" if p_partial < 0.001 else "**" if p_partial < 0.01 else "*" if p_partial < 0.05 else "ns"
            print(f"    WJ vs {label} (partial dist): rho={rho_partial:.4f}, p={p_partial:.4f} {sig}")

# ============================================================================
# 3. EXPRESSION vs FC variants (null comparison)
# ============================================================================
print("\n" + "="*60)
print("3. EXPRESSION SIMILARITY vs FC variants (null baseline)")
print("="*60)

for label, col in fc_variants.items():
    rho, p = stats.spearmanr(pairs['Expr_Sim'], pairs[col])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  Expr vs {label}: rho={rho:.4f}, p={p:.4f} {sig}")

# ============================================================================
# 4. KEY DISSOCIATION TEST
# ============================================================================
print("\n" + "="*60)
print("4. KEY DISSOCIATION: WJ predicts resistant but not activity?")
print("="*60)

rho_resistant, p_resistant = stats.spearmanr(pairs['WJ'], pairs['FC_resistant_min'])
rho_activity, p_activity = stats.spearmanr(pairs['WJ'], pairs['FC_activity'])

print(f"  WJ vs FC_resistant (min): rho={rho_resistant:.4f}, p={p_resistant:.6f}")
print(f"  WJ vs FC_activity (awake-unconscious): rho={rho_activity:.4f}, p={p_activity:.6f}")

# Steiger test comparing the two correlations
from math import atanh, sqrt
def steiger_test(r1, r2, r12, n):
    """Test whether two dependent correlations differ (Steiger 1980)."""
    z1 = atanh(r1)
    z2 = atanh(r2)
    z12 = atanh(r12)
    denom = sqrt((2 * (1 - r12)) / ((1 - r1**2)**2 + (1 - r2**2)**2 - 2 * r12 * (1 - r1**2) * (1 - r2**2)))
    # Simplified approximation
    diff = z1 - z2
    se = sqrt(2 * (1 - r12) / (n - 3))
    z_stat = diff / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_val

# Correlation between FC_resistant and FC_activity
r_between, _ = stats.pearsonr(pairs['FC_resistant_min'], pairs['FC_activity'])
r_wj_res = stats.pearsonr(pairs['WJ'], pairs['FC_resistant_min'])[0]
r_wj_act = stats.pearsonr(pairs['WJ'], pairs['FC_activity'])[0]

z_steiger, p_steiger = steiger_test(r_wj_res, r_wj_act, r_between, len(pairs))
print(f"\n  Steiger test (WJ-FC_resistant vs WJ-FC_activity):")
print(f"    Z = {z_steiger:.4f}, p = {p_steiger:.6f}")
print(f"    WJ predicts resistant {'BETTER' if r_wj_res > r_wj_act else 'WORSE'} than activity")

# ============================================================================
# SAVE ALL RESULTS
# ============================================================================
all_results = {
    'mantel_tests': mantel_results,
    'spearman_by_pair_type': spearman_results,
    'dissociation': {
        'wj_vs_fc_resistant_min': {'rho': float(rho_resistant), 'p': float(p_resistant)},
        'wj_vs_fc_activity': {'rho': float(rho_activity), 'p': float(p_activity)},
        'steiger_z': float(z_steiger),
        'steiger_p': float(p_steiger),
    },
}

with open(os.path.join(OUT, 'propofol_resistant_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=float)

print(f"\nAll results saved to {os.path.join(OUT, 'propofol_resistant_results.json')}")
print("\n=== PATH 2 COMPLETE ===")
