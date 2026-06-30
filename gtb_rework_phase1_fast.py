"""
Pipeline: Gene-to-Brain Rework Phase 1 — Fast tasks (1, 3, 4)
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-05-30
Description:
    Foundation-up rebuild items from the 2026-05-30 audit:
    Task 1: Verify whether FC_resistant_min is just FC_unconscious relabeled.
    Task 3: Cross-type Mantel test on the n=28 subset (Layer 4.2).
    Task 4: Formalize Layer 2H Type 6 substrate-projection pairing
            (WJ vs Pearson, Spearman, cosine on the WJ-vs-FC question)
            with bootstrap CIs on the gaps.
    These three tasks all operate on existing CSV outputs and run in minutes.
    Task 2 (Layer 2I) requires GTEx reload and is in a separate script.
Dependencies: numpy, scipy, pandas
Input: results/bulletproof_pairs.csv, results/region_coexpression_wj_matrix.csv,
       results/variance_weighting/variance_weighted_pairs.csv
Output: results/rework_phase1/task1_fc_resistant_verification.json,
        results/rework_phase1/task3_cross_type_mantel.json,
        results/rework_phase1/task4_layer2h_type6.json
"""
from __future__ import annotations
import os
import time
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# ============================================================================
# CONFIG
# ============================================================================
RANDOM_SEED = 42
FORCE_RECOMPUTE = True
N_PERM_MANTEL = 10000
N_BOOTSTRAP = 1000

BASE = Path(r'G:\My Drive\inner_architecture_research\gene_to_brain_wj')
RESULTS = BASE / 'results'
REWORK = RESULTS / 'rework_phase1'
REWORK.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)
START = time.time()


def log(msg):
    print(f"[{(time.time()-START)/60:6.1f}m] {msg}", flush=True)


# ============================================================================
# HELPERS
# ============================================================================
def cosine_sim(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ============================================================================
# TASK 1: Verify FC_resistant_min vs FC_unconscious
# ============================================================================
def task1_verify_fc_resistant():
    log("=" * 70)
    log("TASK 1: Verify FC_resistant_min == FC_unconscious?")
    log("=" * 70)

    pairs = pd.read_csv(RESULTS / 'bulletproof_pairs.csv')
    n_total = len(pairs)
    n_uncon_less = int((pairs['FC_Unconscious'] < pairs['FC_Awake']).sum())
    n_uncon_equal = int((pairs['FC_Unconscious'] == pairs['FC_Awake']).sum())
    n_uncon_greater = int((pairs['FC_Unconscious'] > pairs['FC_Awake']).sum())
    pct_uncon_less = 100.0 * n_uncon_less / n_total

    pairs['FC_resistant_min'] = pairs[['FC_Awake', 'FC_Unconscious']].min(axis=1)
    diff = (pairs['FC_resistant_min'] - pairs['FC_Unconscious']).abs()
    max_diff = float(diff.max())
    equals_uncon = bool(diff.max() < 1e-12)

    log(f"  Total pairs: {n_total}")
    log(f"  FC_uncon < FC_awake: {n_uncon_less}/{n_total} ({pct_uncon_less:.1f}%)")
    log(f"  FC_uncon == FC_awake: {n_uncon_equal}/{n_total}")
    log(f"  FC_uncon > FC_awake: {n_uncon_greater}/{n_total}")
    log(f"  FC_resistant_min equals FC_unconscious (max abs diff): {max_diff:.2e}")
    log(f"  >>> {'YES — FC_resistant_min IS FC_unconscious relabeled' if equals_uncon else 'NO — partially distinct'}")

    # Spearman correlation between resistant_min and unconscious
    r_resistant_uncon, p_resistant_uncon = spearmanr(pairs['FC_resistant_min'],
                                                      pairs['FC_Unconscious'])
    log(f"  Spearman(resistant_min, unconscious): rho={r_resistant_uncon:.6f}, p={p_resistant_uncon:.2e}")

    # If equals, then "WJ predicts FC_resistant" and "WJ predicts FC_unconscious" are
    # THE SAME claim, not two separate findings.

    result = {
        'n_total_pairs': n_total,
        'n_uncon_less_than_awake': n_uncon_less,
        'pct_uncon_less': pct_uncon_less,
        'n_uncon_equal_awake': n_uncon_equal,
        'n_uncon_greater_awake': n_uncon_greater,
        'fc_resistant_min_equals_unconscious': equals_uncon,
        'max_abs_diff_resistant_unconscious': max_diff,
        'spearman_resistant_unconscious': float(r_resistant_uncon),
        'conclusion': ('FC_resistant_min IS FC_unconscious relabeled for all pairs. '
                       'The "WJ predicts FC_resistant" and "WJ predicts FC_unconscious" claims '
                       'are the same finding stated twice.') if equals_uncon else
                       'FC_resistant_min differs from FC_unconscious in some pairs.',
    }
    with open(REWORK / 'task1_fc_resistant_verification.json', 'w') as f:
        json.dump(result, f, indent=2)
    log(f"  Saved: task1_fc_resistant_verification.json")
    return result


# ============================================================================
# TASK 3: Cross-type Mantel test (Layer 4.2 stringency requirement)
# ============================================================================
def task3_cross_type_mantel(n_perm=N_PERM_MANTEL):
    log("\n" + "=" * 70)
    log("TASK 3: Cross-type Mantel test (Layer 4.2)")
    log("=" * 70)

    pairs = pd.read_csv(RESULTS / 'bulletproof_pairs.csv')
    vw = pd.read_csv(RESULTS / 'variance_weighting' / 'variance_weighted_pairs.csv')
    if len(vw) != len(pairs):
        log(f"  WARNING: variance_weighting pair count {len(vw)} != bulletproof pair count {len(pairs)}")
    pairs['Pair_type'] = vw['Pair_type'].values

    wj_mat = pd.read_csv(RESULTS / 'region_coexpression_wj_matrix.csv', index_col=0)
    regions = sorted(wj_mat.index.tolist())
    n_reg = len(regions)
    region_idx = {r: i for i, r in enumerate(regions)}

    # Build full matrices from pair-level data
    fc_awake_mat = np.zeros((n_reg, n_reg))
    fc_uncon_mat = np.zeros((n_reg, n_reg))
    cross_mask = np.zeros((n_reg, n_reg), dtype=bool)
    for _, row in pairs.iterrows():
        i = region_idx.get(row['Region1'])
        j = region_idx.get(row['Region2'])
        if i is None or j is None:
            continue
        fc_awake_mat[i, j] = fc_awake_mat[j, i] = row['FC_Awake']
        fc_uncon_mat[i, j] = fc_uncon_mat[j, i] = row['FC_Unconscious']
        if row['Pair_type'] == 'cross-type':
            cross_mask[i, j] = cross_mask[j, i] = True

    tri = np.triu_indices(n_reg, k=1)
    cross_tri = cross_mask[tri]

    wj_full = wj_mat.values
    wj_tri = wj_full[tri]
    fc_awake_tri = fc_awake_mat[tri]
    fc_uncon_tri = fc_uncon_mat[tri]

    wj_cross = wj_tri[cross_tri]
    fc_awake_cross = fc_awake_tri[cross_tri]
    fc_uncon_cross = fc_uncon_tri[cross_tri]

    n_cross = int(cross_tri.sum())
    n_within = int((~cross_tri).sum())
    log(f"  Cross-type pairs: {n_cross}")
    log(f"  Within-type pairs: {n_within}")

    # Observed cross-type Pearson (= Mantel statistic)
    r_obs_awake, _ = pearsonr(wj_cross, fc_awake_cross)
    r_obs_uncon, _ = pearsonr(wj_cross, fc_uncon_cross)
    log(f"\n  Observed cross-type Pearson:")
    log(f"    WJ vs FC_awake: r = {r_obs_awake:.4f}")
    log(f"    WJ vs FC_uncon: r = {r_obs_uncon:.4f}")

    # Mantel-style null: permute the 11 region labels in WJ matrix,
    # extract cross-type subset (using FIXED cross_mask in original layout),
    # correlate with fixed FC subset.
    rng = np.random.RandomState(RANDOM_SEED)
    null_awake = np.zeros(n_perm)
    null_uncon = np.zeros(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n_reg)
        wj_perm = wj_full[np.ix_(perm, perm)]
        wj_perm_tri = wj_perm[tri]
        wj_perm_cross = wj_perm_tri[cross_tri]
        null_awake[p], _ = pearsonr(wj_perm_cross, fc_awake_cross)
        null_uncon[p], _ = pearsonr(wj_perm_cross, fc_uncon_cross)

    # Two-sided p (consistent with main manuscript Mantel reporting)
    p_awake_one = float(np.mean(null_awake >= r_obs_awake))
    p_uncon_one = float(np.mean(null_uncon >= r_obs_uncon))
    p_awake_two = float(np.mean(np.abs(null_awake) >= abs(r_obs_awake)))
    p_uncon_two = float(np.mean(np.abs(null_uncon) >= abs(r_obs_uncon)))

    log(f"\n  Mantel null distribution (n={n_perm} perms):")
    log(f"    Awake null mean = {null_awake.mean():.4f}, std = {null_awake.std():.4f}")
    log(f"    Uncon null mean = {null_uncon.mean():.4f}, std = {null_uncon.std():.4f}")
    log(f"\n  Cross-type Mantel p-values:")
    log(f"    WJ vs FC_awake: p_one = {p_awake_one:.4f}, p_two = {p_awake_two:.4f}")
    log(f"    WJ vs FC_uncon: p_one = {p_uncon_one:.4f}, p_two = {p_uncon_two:.4f}")

    # Also within-type for symmetry
    wj_within = wj_tri[~cross_tri]
    fc_awake_within = fc_awake_tri[~cross_tri]
    fc_uncon_within = fc_uncon_tri[~cross_tri]
    r_within_awake, _ = pearsonr(wj_within, fc_awake_within)
    r_within_uncon, _ = pearsonr(wj_within, fc_uncon_within)

    rng2 = np.random.RandomState(RANDOM_SEED + 1)
    null_w_awake = np.zeros(n_perm)
    null_w_uncon = np.zeros(n_perm)
    for p in range(n_perm):
        perm = rng2.permutation(n_reg)
        wj_perm = wj_full[np.ix_(perm, perm)]
        wj_perm_tri = wj_perm[tri]
        wj_perm_within = wj_perm_tri[~cross_tri]
        null_w_awake[p], _ = pearsonr(wj_perm_within, fc_awake_within)
        null_w_uncon[p], _ = pearsonr(wj_perm_within, fc_uncon_within)
    p_w_awake = float(np.mean(np.abs(null_w_awake) >= abs(r_within_awake)))
    p_w_uncon = float(np.mean(np.abs(null_w_uncon) >= abs(r_within_uncon)))

    log(f"\n  Within-type Mantel (for comparison):")
    log(f"    WJ vs FC_awake: r = {r_within_awake:.4f}, p_two = {p_w_awake:.4f}")
    log(f"    WJ vs FC_uncon: r = {r_within_uncon:.4f}, p_two = {p_w_uncon:.4f}")

    result = {
        'n_perm': n_perm,
        'random_seed': RANDOM_SEED,
        'n_cross_type': n_cross,
        'n_within_type': n_within,
        'cross_type': {
            'observed_pearson_awake': float(r_obs_awake),
            'observed_pearson_uncon': float(r_obs_uncon),
            'mantel_p_one_awake': p_awake_one,
            'mantel_p_one_uncon': p_uncon_one,
            'mantel_p_two_awake': p_awake_two,
            'mantel_p_two_uncon': p_uncon_two,
            'null_mean_awake': float(null_awake.mean()),
            'null_std_awake': float(null_awake.std()),
            'null_mean_uncon': float(null_uncon.mean()),
            'null_std_uncon': float(null_uncon.std()),
        },
        'within_type': {
            'observed_pearson_awake': float(r_within_awake),
            'observed_pearson_uncon': float(r_within_uncon),
            'mantel_p_two_awake': p_w_awake,
            'mantel_p_two_uncon': p_w_uncon,
        },
        'layer_4_2_assessment': (
            'Cross-type subset Mantel-tested at full-dataset stringency. '
            'See observed_pearson_* and mantel_p_* values for the formal subset result.'
        ),
    }
    with open(REWORK / 'task3_cross_type_mantel.json', 'w') as f:
        json.dump(result, f, indent=2)
    log(f"  Saved: task3_cross_type_mantel.json")
    return result


# ============================================================================
# TASK 4: Layer 2H Type 6 substrate-projection pairing
# ============================================================================
def task4_layer2h_type6(n_boot=N_BOOTSTRAP):
    log("\n" + "=" * 70)
    log("TASK 4: Layer 2H Type 6 substrate-projection pairing")
    log("=" * 70)
    log("  Question: Which similarity metric used to compare WJ_vec vs FC_vec")
    log("            produces strongest correspondence? Gap localizes which")
    log("            feature (linearity vs monotonicity vs vector alignment) drives it.")

    pairs = pd.read_csv(RESULTS / 'bulletproof_pairs.csv')
    wj_mat = pd.read_csv(RESULTS / 'region_coexpression_wj_matrix.csv', index_col=0)
    regions = sorted(wj_mat.index.tolist())
    n_reg = len(regions)
    region_idx = {r: i for i, r in enumerate(regions)}

    fc_awake_mat = np.zeros((n_reg, n_reg))
    fc_uncon_mat = np.zeros((n_reg, n_reg))
    expr_mat = np.zeros((n_reg, n_reg))
    for _, row in pairs.iterrows():
        i = region_idx.get(row['Region1'])
        j = region_idx.get(row['Region2'])
        if i is None or j is None:
            continue
        fc_awake_mat[i, j] = fc_awake_mat[j, i] = row['FC_Awake']
        fc_uncon_mat[i, j] = fc_uncon_mat[j, i] = row['FC_Unconscious']
        expr_mat[i, j] = expr_mat[j, i] = row['Expr_Sim']

    tri = np.triu_indices(n_reg, k=1)
    wj_arr = wj_mat.values[tri]
    fc_awake_arr = fc_awake_mat[tri]
    fc_uncon_arr = fc_uncon_mat[tri]
    expr_arr = expr_mat[tri]

    # Point estimates: three projections (Pearson, Spearman, cosine) of the WJ-FC question
    conditions = {
        'FC_Awake': fc_awake_arr,
        'FC_Unconscious': fc_uncon_arr,
        'FC_Activity_AminusU': fc_awake_arr - fc_uncon_arr,
    }
    # Expression-side null too
    predictors = {
        'WJ': wj_arr,
        'ExprProfile': expr_arr,
    }

    log(f"\n  Point estimates (n=55 pairs):")
    point_estimates = {}
    for pred_name, pred_arr in predictors.items():
        point_estimates[pred_name] = {}
        for cond_name, cond_arr in conditions.items():
            r_p, _ = pearsonr(pred_arr, cond_arr)
            r_s, _ = spearmanr(pred_arr, cond_arr)
            r_c = cosine_sim(pred_arr, cond_arr)
            point_estimates[pred_name][cond_name] = {
                'pearson': float(r_p),
                'spearman': float(r_s),
                'cosine': r_c,
                'pearson_minus_spearman': float(r_p - r_s),
                'pearson_minus_cosine': float(r_p - r_c),
                'spearman_minus_cosine': float(r_s - r_c),
            }
            log(f"    {pred_name} vs {cond_name}: pearson={r_p:.4f} spearman={r_s:.4f} cosine={r_c:.4f}")

    # Bootstrap CIs on gaps
    log(f"\n  Bootstrap CIs on gaps (n_boot={n_boot})...")
    rng = np.random.RandomState(RANDOM_SEED)
    n_pairs = len(wj_arr)

    boot_gaps = {pred_name: {cond_name: {'p_minus_s': [], 'p_minus_c': [], 's_minus_c': []}
                              for cond_name in conditions}
                  for pred_name in predictors}

    for b in range(n_boot):
        idx = rng.choice(n_pairs, n_pairs, replace=True)
        for pred_name, pred_arr in predictors.items():
            pred_b = pred_arr[idx]
            for cond_name, cond_arr in conditions.items():
                cond_b = cond_arr[idx]
                try:
                    r_p, _ = pearsonr(pred_b, cond_b)
                    r_s, _ = spearmanr(pred_b, cond_b)
                    r_c = cosine_sim(pred_b, cond_b)
                except Exception:
                    r_p = r_s = r_c = np.nan
                boot_gaps[pred_name][cond_name]['p_minus_s'].append(r_p - r_s)
                boot_gaps[pred_name][cond_name]['p_minus_c'].append(r_p - r_c)
                boot_gaps[pred_name][cond_name]['s_minus_c'].append(r_s - r_c)

    log(f"\n  Bootstrap CI summary (Pearson-Spearman gap):")
    gap_cis = {}
    for pred_name in boot_gaps:
        gap_cis[pred_name] = {}
        for cond_name in boot_gaps[pred_name]:
            gap_cis[pred_name][cond_name] = {}
            for gap_name, vals in boot_gaps[pred_name][cond_name].items():
                vals_arr = np.array(vals)
                valid = ~np.isnan(vals_arr)
                if valid.sum() < 100:
                    continue
                ci_lo = float(np.percentile(vals_arr[valid], 2.5))
                ci_hi = float(np.percentile(vals_arr[valid], 97.5))
                mean = float(np.mean(vals_arr[valid]))
                excludes_zero = bool(ci_lo > 0 or ci_hi < 0)
                gap_cis[pred_name][cond_name][gap_name] = {
                    'mean': mean,
                    'ci_lo': ci_lo,
                    'ci_hi': ci_hi,
                    'excludes_zero': excludes_zero,
                    'n_bootstrap_valid': int(valid.sum()),
                }
                if gap_name == 'p_minus_s':
                    sig = "EXCL 0" if excludes_zero else "incl 0"
                    log(f"    {pred_name} vs {cond_name}: {mean:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] {sig}")

    result = {
        'n_bootstrap': n_boot,
        'random_seed': RANDOM_SEED,
        'pairing_type': 'Layer 2H Type 6 substrate-projection',
        'transformation_axis': ('Choice of matrix similarity metric (Pearson vs Spearman vs cosine) '
                                 'applied to vectorized region-pair WJ values and region-pair FC values. '
                                 'Pearson captures linear correspondence, Spearman captures monotonic, '
                                 'cosine captures geometric alignment ignoring mean.'),
        'point_estimates': point_estimates,
        'gap_cis_bootstrap': gap_cis,
        'interpretation_template': (
            'If pearson_minus_spearman CI excludes 0, the WJ-FC correspondence is sensitive to '
            'specifically linear (vs monotonic-but-nonlinear) structure. If pearson_minus_cosine '
            'CI excludes 0, the correspondence depends on the mean-subtracted vs raw alignment.'
        ),
    }
    with open(REWORK / 'task4_layer2h_type6.json', 'w') as f:
        json.dump(result, f, indent=2)
    log(f"  Saved: task4_layer2h_type6.json")
    return result


# ============================================================================
# PROVENANCE
# ============================================================================
def write_provenance(elapsed_min):
    prov = {
        'methodology': 'WJ-native (foundation-up rebuild Phase 1)',
        'fundamental_unit': 'individual gene (GTEx) and individual ROI (fMRI)',
        'pairwise_matrix': 'full gene-gene Spearman within region; region-region WJ',
        'correlation_method': 'Spearman (primary), Pearson + cosine reported as Layer 2H Type 6',
        'fdr_scope': 'N/A for cross-scale correlation; Mantel and bootstrap report uncertainty directly',
        'domain_conventional_methods': 'none',
        'random_seed': RANDOM_SEED,
        'n_permutations_mantel': N_PERM_MANTEL,
        'n_bootstrap': N_BOOTSTRAP,
        'pipeline_file': os.path.basename(__file__),
        'execution_date': time.strftime('%Y-%m-%d'),
        'execution_time_minutes': elapsed_min,
        'wj_compliance_status': 'PASS (phase 1 of rebuild)',
        'tasks_completed': ['task1_fc_resistant_verification',
                             'task3_cross_type_mantel',
                             'task4_layer2h_type6'],
        'tasks_pending': ['task2_layer2i_sign_flip (run separately due to GTEx reload cost)'],
        'methodology_extensions_applied': [
            'Layer 4.2 subset-stringency (Mantel on cross-type subset)',
            'Layer 2H Type 6 substrate-projection pairing with bootstrap CIs',
        ],
        'data_sources': {
            'gene_expression': 'GTEx v8 individual-level samples (16,273 genes, 11 brain regions, 2,268 samples)',
            'brain_connectivity': 'OpenNeuro ds006623 (Michigan propofol fMRI, 4S456 parcellation)',
        },
        'note': ('Phase 1 of the foundation-up rebuild. Builds on existing pair-level outputs '
                 '(bulletproof_pairs.csv, region_coexpression_wj_matrix.csv) without re-running '
                 'the upstream GTEx-to-WJ pipeline. Phase 2 (task 2 Layer 2I) reloads GTEx samples '
                 'and recomputes per-region gene-gene matrices for sign-flip stratification.'),
    }
    with open(REWORK / 'provenance.json', 'w') as f:
        json.dump(prov, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================
def main():
    log("=" * 70)
    log("GENE-TO-BRAIN REWORK PHASE 1 — Items 1, 3, 4")
    log(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Random seed: {RANDOM_SEED}")
    log(f"Output dir: {REWORK}")
    log("=" * 70)

    log("\n>>> TASK 1")
    task1_verify_fc_resistant()

    log("\n>>> TASK 3")
    task3_cross_type_mantel()

    log("\n>>> TASK 4")
    task4_layer2h_type6()

    elapsed = (time.time() - START) / 60
    write_provenance(elapsed)
    log(f"\n{'='*70}")
    log(f"PHASE 1 (FAST) COMPLETE in {elapsed:.1f} minutes")
    log(f"Outputs in: {REWORK}")
    log(f"{'='*70}")


if __name__ == '__main__':
    main()
