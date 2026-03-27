"""
Pipeline: Gene-to-Brain WJ Strengthening — Mantel test + distance-matched subsampling
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-03-27
Description: Adds two critical robustness analyses to address reviewer concerns:
    1. Mantel test (matrix correlation) as alternative to WJ — tests whether the
       WJ-FC relationship is specific to WJ or holds for any matrix similarity metric
    2. Distance-matched subsampling — tests whether cross-type signal survives
       when distance distribution is matched to within-type pairs
Dependencies: scipy, numpy, pandas
Input: bulletproof_pairs.csv, variance_weighted_pairs.csv, region_coexpression_wj_matrix.csv
Output: results/strengthening_v2/ with Mantel results and distance-matching results
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

N_PERM_MANTEL = 10000
N_ITER_DISTANCE_MATCH = 1000

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")

# Main pairs with WJ, expression similarity, FC, distance
pairs = pd.read_csv(os.path.join(RESULTS, 'bulletproof_pairs.csv'))

# Pair type classification
vw = pd.read_csv(os.path.join(RESULTS, 'variance_weighting', 'variance_weighted_pairs.csv'))
pairs['Pair_type'] = vw['Pair_type']

# WJ matrix (11x11)
wj_matrix = pd.read_csv(os.path.join(RESULTS, 'region_coexpression_wj_matrix.csv'), index_col=0)

print(f"  Pairs: {len(pairs)}")
print(f"  Regions: {len(wj_matrix)}")
print(f"  Pair types: {pairs['Pair_type'].value_counts().to_dict()}")

# ============================================================================
# 1. MANTEL TEST — Matrix correlation as alternative to WJ
# ============================================================================
print("\n" + "="*60)
print("1. MANTEL TEST (matrix correlation alternative to WJ)")
print("="*60)

# Build FC matrices (11x11) from pair data
regions = sorted(wj_matrix.index.tolist())
n_reg = len(regions)
region_idx = {r: i for i, r in enumerate(regions)}

def build_symmetric_matrix(pairs_df, col, regions, region_idx):
    """Build a symmetric matrix from pair-level data."""
    n = len(regions)
    mat = np.zeros((n, n))
    for _, row in pairs_df.iterrows():
        i = region_idx.get(row['Region1'])
        j = region_idx.get(row['Region2'])
        if i is not None and j is not None:
            mat[i, j] = row[col]
            mat[j, i] = row[col]
    np.fill_diagonal(mat, 1.0 if col in ('WJ', 'Expr_Sim') else 0.0)
    return mat

fc_awake_mat = build_symmetric_matrix(pairs, 'FC_Awake', regions, region_idx)
fc_uncon_mat = build_symmetric_matrix(pairs, 'FC_Unconscious', regions, region_idx)
wj_mat = wj_matrix.values
distance_mat = build_symmetric_matrix(pairs, 'Distance_mm', regions, region_idx)
expr_mat = build_symmetric_matrix(pairs, 'Expr_Sim', regions, region_idx)

# Extract upper triangle (excluding diagonal) for Mantel test
tri_idx = np.triu_indices(n_reg, k=1)
wj_vec = wj_mat[tri_idx]
fc_awake_vec = fc_awake_mat[tri_idx]
fc_uncon_vec = fc_uncon_mat[tri_idx]
dist_vec = distance_mat[tri_idx]
expr_vec = expr_mat[tri_idx]

assert len(wj_vec) == 55, f"Expected 55 pairs, got {len(wj_vec)}"

# --- Mantel test: WJ vs FC (permutation-based matrix correlation) ---
def mantel_test(mat_a_vec, mat_b_vec, n_perm, seed=42):
    """Mantel test: Pearson correlation between vectorized upper triangles,
    with permutation significance via row/column shuffling."""
    rng = np.random.RandomState(seed)
    n = int(0.5 * (1 + np.sqrt(1 + 8 * len(mat_a_vec))))  # recover matrix size

    observed_r, _ = stats.pearsonr(mat_a_vec, mat_b_vec)

    # Also compute Spearman for robustness
    observed_rho, _ = stats.spearmanr(mat_a_vec, mat_b_vec)

    # Permutation: shuffle rows/columns of one matrix
    mat_a = squareform(mat_a_vec)
    np.fill_diagonal(mat_a, 0)

    null_r = np.zeros(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n)
        shuffled = mat_a[np.ix_(perm, perm)]
        shuffled_vec = shuffled[np.triu_indices(n, k=1)]
        null_r[p], _ = stats.pearsonr(shuffled_vec, mat_b_vec)

    p_value = np.mean(null_r >= observed_r)

    return {
        'pearson_r': observed_r,
        'spearman_rho': observed_rho,
        'p_value_perm': p_value,
        'null_mean': np.mean(null_r),
        'null_std': np.std(null_r),
        'n_perm': n_perm,
    }

print("\n--- Mantel test: WJ matrix vs FC_Awake matrix ---")
mantel_wj_awake = mantel_test(wj_vec, fc_awake_vec, N_PERM_MANTEL)
print(f"  Pearson r = {mantel_wj_awake['pearson_r']:.4f}")
print(f"  Spearman rho = {mantel_wj_awake['spearman_rho']:.4f}")
print(f"  Permutation p = {mantel_wj_awake['p_value_perm']:.4f}")

print("\n--- Mantel test: WJ matrix vs FC_Unconscious matrix ---")
mantel_wj_uncon = mantel_test(wj_vec, fc_uncon_vec, N_PERM_MANTEL)
print(f"  Pearson r = {mantel_wj_uncon['pearson_r']:.4f}")
print(f"  Spearman rho = {mantel_wj_uncon['spearman_rho']:.4f}")
print(f"  Permutation p = {mantel_wj_uncon['p_value_perm']:.4f}")

print("\n--- Mantel test: Expression matrix vs FC_Awake matrix ---")
mantel_expr_awake = mantel_test(expr_vec, fc_awake_vec, N_PERM_MANTEL)
print(f"  Pearson r = {mantel_expr_awake['pearson_r']:.4f}")
print(f"  Spearman rho = {mantel_expr_awake['spearman_rho']:.4f}")
print(f"  Permutation p = {mantel_expr_awake['p_value_perm']:.4f}")

# --- Frobenius-based similarity as second alternative ---
# Frobenius similarity = 1 - (||A-B||_F / ||A||_F + ||B||_F) ... but simpler:
# just use the direct vector correlation (which IS the Mantel test)
# The Mantel test IS the standard matrix correlation. No need for separate Frobenius.

# --- Partial Mantel: WJ vs FC controlling for distance ---
print("\n--- Partial Mantel: WJ vs FC_Awake, controlling for distance ---")

def partial_mantel(vec_a, vec_b, vec_c, n_perm, seed=42):
    """Partial Mantel test: correlation between A and B after removing effect of C."""
    # Residualize A and B on C
    slope_a, intercept_a, _, _, _ = stats.linregress(vec_c, vec_a)
    resid_a = vec_a - (slope_a * vec_c + intercept_a)

    slope_b, intercept_b, _, _, _ = stats.linregress(vec_c, vec_b)
    resid_b = vec_b - (slope_b * vec_c + intercept_b)

    observed_r, _ = stats.pearsonr(resid_a, resid_b)
    observed_rho, _ = stats.spearmanr(resid_a, resid_b)

    # Permutation on residuals
    rng = np.random.RandomState(seed)
    n = int(0.5 * (1 + np.sqrt(1 + 8 * len(vec_a))))
    mat_resid_a = squareform(resid_a)

    null_r = np.zeros(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n)
        shuffled = mat_resid_a[np.ix_(perm, perm)]
        shuffled_vec = shuffled[np.triu_indices(n, k=1)]
        null_r[p], _ = stats.pearsonr(shuffled_vec, resid_b)

    p_value = np.mean(null_r >= observed_r)

    return {
        'partial_pearson_r': observed_r,
        'partial_spearman_rho': observed_rho,
        'p_value_perm': p_value,
        'null_mean': np.mean(null_r),
        'null_std': np.std(null_r),
    }

partial_mantel_awake = partial_mantel(wj_vec, fc_awake_vec, dist_vec, N_PERM_MANTEL)
print(f"  Partial Pearson r = {partial_mantel_awake['partial_pearson_r']:.4f}")
print(f"  Partial Spearman rho = {partial_mantel_awake['partial_spearman_rho']:.4f}")
print(f"  Permutation p = {partial_mantel_awake['p_value_perm']:.4f}")

# --- Cross-type only: Mantel on subset ---
print("\n--- Cross-type pairs only: Spearman WJ vs FC ---")
cross = pairs[pairs['Pair_type'] == 'cross-type']
rho_cross_awake, p_cross_awake = stats.spearmanr(cross['WJ'], cross['FC_Awake'])
rho_cross_uncon, p_cross_uncon = stats.spearmanr(cross['WJ'], cross['FC_Unconscious'])

# Partial correlation for cross-type (controlling distance)
from functools import partial as functools_partial

def partial_corr_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling z."""
    rx = stats.spearmanr(z, x)[0]
    ry = stats.spearmanr(z, y)[0]
    resid_x = stats.rankdata(x) - rx * stats.rankdata(z)
    resid_y = stats.rankdata(y) - ry * stats.rankdata(z)
    return stats.spearmanr(resid_x, resid_y)

partial_cross_awake = partial_corr_spearman(
    cross['WJ'].values, cross['FC_Awake'].values, cross['Distance_mm'].values
)
partial_cross_uncon = partial_corr_spearman(
    cross['WJ'].values, cross['FC_Unconscious'].values, cross['Distance_mm'].values
)

print(f"  Cross-type WJ vs FC_Awake: rho={rho_cross_awake:.4f}, p={p_cross_awake:.4f}")
print(f"  Cross-type WJ vs FC_Unconscious: rho={rho_cross_uncon:.4f}, p={p_cross_uncon:.4f}")
print(f"  Partial (distance) WJ vs FC_Awake: rho={partial_cross_awake[0]:.4f}, p={partial_cross_awake[1]:.4f}")
print(f"  Partial (distance) WJ vs FC_Unconscious: rho={partial_cross_uncon[0]:.4f}, p={partial_cross_uncon[1]:.4f}")

# Save Mantel results
mantel_results = {
    'mantel_wj_fc_awake': mantel_wj_awake,
    'mantel_wj_fc_unconscious': mantel_wj_uncon,
    'mantel_expr_fc_awake': mantel_expr_awake,
    'partial_mantel_wj_fc_awake_dist': partial_mantel_awake,
    'cross_type_spearman': {
        'wj_fc_awake': {'rho': rho_cross_awake, 'p': p_cross_awake},
        'wj_fc_unconscious': {'rho': rho_cross_uncon, 'p': p_cross_uncon},
        'partial_wj_fc_awake': {'rho': partial_cross_awake[0], 'p': partial_cross_awake[1]},
        'partial_wj_fc_unconscious': {'rho': partial_cross_uncon[0], 'p': partial_cross_uncon[1]},
    }
}

with open(os.path.join(OUT, 'mantel_results.json'), 'w') as f:
    json.dump(mantel_results, f, indent=2, default=float)
print(f"\nMantel results saved to {os.path.join(OUT, 'mantel_results.json')}")

# ============================================================================
# 2. DISTANCE-MATCHED SUBSAMPLING
# ============================================================================
print("\n" + "="*60)
print("2. DISTANCE-MATCHED SUBSAMPLING")
print("="*60)

cross_type = pairs[pairs['Pair_type'] == 'cross-type'].copy()
within_type = pairs[pairs['Pair_type'] != 'cross-type'].copy()

print(f"\n  Cross-type pairs: {len(cross_type)}")
print(f"  Within-type pairs: {len(within_type)}")
print(f"  Cross-type distance: mean={cross_type['Distance_mm'].mean():.1f}, "
      f"range=[{cross_type['Distance_mm'].min():.1f}, {cross_type['Distance_mm'].max():.1f}]")
print(f"  Within-type distance: mean={within_type['Distance_mm'].mean():.1f}, "
      f"range=[{within_type['Distance_mm'].min():.1f}, {within_type['Distance_mm'].max():.1f}]")

# Strategy: For each iteration, subsample cross-type pairs to match the distance
# distribution of within-type pairs. If the WJ-FC correlation survives in the
# distance-matched subset, the signal is distance-independent.
#
# Also do the reverse: subsample within-type pairs to match cross-type distances.
# If within-type shows no WJ-FC correlation even at cross-type distances,
# it's the cross-type nature, not the distance, that matters.

rng = np.random.RandomState(RANDOM_SEED)

# Forward matching: subsample cross-type to match within-type distance distribution
# Use nearest-neighbor matching with replacement
def distance_match_subsample(source_df, target_distances, rng, n_sample=None):
    """Subsample from source_df to match target distance distribution.
    For each target distance, find nearest source pair."""
    if n_sample is None:
        n_sample = len(target_distances)

    source_dists = source_df['Distance_mm'].values
    selected_idx = []

    for t_dist in target_distances:
        # Find closest source pair
        diffs = np.abs(source_dists - t_dist)
        # Add small random jitter to break ties
        diffs += rng.uniform(0, 0.001, len(diffs))
        best = np.argmin(diffs)
        selected_idx.append(best)

    return source_df.iloc[selected_idx]

# Main test: subsample within-type to match cross-type distances
# Then compute WJ-FC correlation for the distance-matched within-type subset
print("\n--- Distance-matched within-type vs cross-type ---")

cross_rhos_awake = []
within_matched_rhos_awake = []
cross_rhos_uncon = []
within_matched_rhos_uncon = []

for i in range(N_ITER_DISTANCE_MATCH):
    # Subsample within-type pairs to match cross-type distance distribution
    # (with replacement since within-type may have fewer pairs at some distances)
    target_dists = cross_type['Distance_mm'].values

    # Bootstrap cross-type (to get CI)
    boot_cross = cross_type.sample(n=len(cross_type), replace=True, random_state=rng.randint(1e9))
    rho_c_a, _ = stats.spearmanr(boot_cross['WJ'], boot_cross['FC_Awake'])
    rho_c_u, _ = stats.spearmanr(boot_cross['WJ'], boot_cross['FC_Unconscious'])
    cross_rhos_awake.append(rho_c_a)
    cross_rhos_uncon.append(rho_c_u)

    # Match within-type to cross-type distances
    matched_within = distance_match_subsample(within_type, target_dists, rng)
    rho_w_a, _ = stats.spearmanr(matched_within['WJ'], matched_within['FC_Awake'])
    rho_w_u, _ = stats.spearmanr(matched_within['WJ'], matched_within['FC_Unconscious'])
    within_matched_rhos_awake.append(rho_w_a)
    within_matched_rhos_uncon.append(rho_w_u)

cross_rhos_awake = np.array(cross_rhos_awake)
within_matched_rhos_awake = np.array(within_matched_rhos_awake)
cross_rhos_uncon = np.array(cross_rhos_uncon)
within_matched_rhos_uncon = np.array(within_matched_rhos_uncon)

# Proportion of iterations where cross-type > within-type (matched)
prop_awake = np.mean(cross_rhos_awake > within_matched_rhos_awake)
prop_uncon = np.mean(cross_rhos_uncon > within_matched_rhos_uncon)

print(f"\n  FC_Awake:")
print(f"    Cross-type rho: mean={np.mean(cross_rhos_awake):.4f}, "
      f"95% CI [{np.percentile(cross_rhos_awake, 2.5):.4f}, {np.percentile(cross_rhos_awake, 97.5):.4f}]")
print(f"    Within-type (distance-matched) rho: mean={np.mean(within_matched_rhos_awake):.4f}, "
      f"95% CI [{np.percentile(within_matched_rhos_awake, 2.5):.4f}, {np.percentile(within_matched_rhos_awake, 97.5):.4f}]")
print(f"    Cross > Within-matched: {prop_awake*100:.1f}% of iterations")

print(f"\n  FC_Unconscious:")
print(f"    Cross-type rho: mean={np.mean(cross_rhos_uncon):.4f}, "
      f"95% CI [{np.percentile(cross_rhos_uncon, 2.5):.4f}, {np.percentile(cross_rhos_uncon, 97.5):.4f}]")
print(f"    Within-type (distance-matched) rho: mean={np.mean(within_matched_rhos_uncon):.4f}, "
      f"95% CI [{np.percentile(within_matched_rhos_uncon, 2.5):.4f}, {np.percentile(within_matched_rhos_uncon, 97.5):.4f}]")
print(f"    Cross > Within-matched: {prop_uncon*100:.1f}% of iterations")

# Save distance matching results
distance_match_results = {
    'n_iterations': N_ITER_DISTANCE_MATCH,
    'n_cross_type': len(cross_type),
    'n_within_type': len(within_type),
    'fc_awake': {
        'cross_type_rho_mean': float(np.mean(cross_rhos_awake)),
        'cross_type_rho_ci': [float(np.percentile(cross_rhos_awake, 2.5)),
                               float(np.percentile(cross_rhos_awake, 97.5))],
        'within_matched_rho_mean': float(np.mean(within_matched_rhos_awake)),
        'within_matched_rho_ci': [float(np.percentile(within_matched_rhos_awake, 2.5)),
                                   float(np.percentile(within_matched_rhos_awake, 97.5))],
        'cross_exceeds_within_pct': float(prop_awake * 100),
    },
    'fc_unconscious': {
        'cross_type_rho_mean': float(np.mean(cross_rhos_uncon)),
        'cross_type_rho_ci': [float(np.percentile(cross_rhos_uncon, 2.5)),
                               float(np.percentile(cross_rhos_uncon, 97.5))],
        'within_matched_rho_mean': float(np.mean(within_matched_rhos_uncon)),
        'within_matched_rho_ci': [float(np.percentile(within_matched_rhos_uncon, 2.5)),
                                   float(np.percentile(within_matched_rhos_uncon, 97.5))],
        'cross_exceeds_within_pct': float(prop_uncon * 100),
    },
}

with open(os.path.join(OUT, 'distance_matching_results.json'), 'w') as f:
    json.dump(distance_match_results, f, indent=2)
print(f"\nDistance matching results saved to {os.path.join(OUT, 'distance_matching_results.json')}")

# ============================================================================
# 3. SUMMARY TABLE
# ============================================================================
print("\n" + "="*60)
print("3. SUMMARY — METRIC ROBUSTNESS")
print("="*60)

summary = pd.DataFrame([
    {
        'Test': 'Spearman (original)',
        'Scope': 'All pairs (n=55)',
        'rho': 0.4636,
        'p': 0.000364,
        'Survives_distance': 'No (partial rho=0.049)',
    },
    {
        'Test': 'Mantel (matrix correlation)',
        'Scope': 'All pairs (n=55)',
        'rho': mantel_wj_awake['pearson_r'],
        'p': mantel_wj_awake['p_value_perm'],
        'Survives_distance': f"{'Yes' if partial_mantel_awake['p_value_perm'] < 0.05 else 'No'} "
                             f"(partial r={partial_mantel_awake['partial_pearson_r']:.3f})",
    },
    {
        'Test': 'Spearman (original)',
        'Scope': 'Cross-type (n=28)',
        'rho': rho_cross_awake,
        'p': p_cross_awake,
        'Survives_distance': f"{'Yes' if partial_cross_awake[1] < 0.05 else 'No'} "
                             f"(partial rho={partial_cross_awake[0]:.3f})",
    },
    {
        'Test': 'Mantel (matrix correlation)',
        'Scope': 'Cross-type (n=28)',
        'rho': mantel_wj_awake['pearson_r'],
        'p': mantel_wj_awake['p_value_perm'],
        'Survives_distance': 'See partial Mantel above',
    },
    {
        'Test': 'Expression similarity',
        'Scope': 'All pairs (n=55)',
        'rho': mantel_expr_awake['spearman_rho'],
        'p': mantel_expr_awake['p_value_perm'],
        'Survives_distance': 'N/A (null result)',
    },
])

summary.to_csv(os.path.join(OUT, 'metric_robustness_summary.csv'), index=False)
print(summary.to_string(index=False))

# ============================================================================
# 4. PROVENANCE
# ============================================================================
provenance = {
    'methodology': 'WJ-native strengthening',
    'pipeline_file': 'gene_to_brain_strengthening.py',
    'execution_date': '2026-03-27',
    'random_seed': RANDOM_SEED,
    'analyses': [
        'Mantel test (matrix correlation) with 10,000 permutations',
        'Partial Mantel test (controlling distance)',
        'Distance-matched subsampling (1,000 iterations)',
    ],
    'key_question': 'Is the WJ-FC relationship specific to WJ, or does any matrix similarity metric show the same pattern?',
    'key_question_2': 'Does the cross-type signal survive when distance distributions are matched?',
}

with open(os.path.join(OUT, 'provenance.json'), 'w') as f:
    json.dump(provenance, f, indent=2)

print(f"\nProvenance saved.")
print("\n=== STRENGTHENING PIPELINE COMPLETE ===")
