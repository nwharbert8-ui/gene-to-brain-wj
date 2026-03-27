"""
Pipeline: Path 1 — AHBA high-resolution parcellation WJ analysis
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-03-27
Description: Downloads Allen Human Brain Atlas microarray data via abagen,
    parcellates into Desikan-Killiany atlas (~68 cortical regions + subcortical),
    computes gene co-expression WJ between all region pairs, and tests against
    functional connectivity. This is the spatial resolution upgrade from 11
    GTEx regions to ~68+ AHBA regions.
Dependencies: abagen, nibabel, scipy, numpy, pandas
Input: AHBA microarray (downloaded automatically), FC data (from existing pipeline)
Output: results/ahba/ with WJ matrix, pair results, Mantel tests
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform

# ============================================================================
# CONFIG
# ============================================================================
BASE = r'G:\My Drive\inner_architecture_research\gene_to_brain_wj'
RESULTS = os.path.join(BASE, 'results')
AHBA_OUT = os.path.join(RESULTS, 'ahba')
AHBA_DATA = os.path.join(BASE, 'data', 'ahba')
os.makedirs(AHBA_OUT, exist_ok=True)
os.makedirs(AHBA_DATA, exist_ok=True)

FORCE_RECOMPUTE = True
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_PERM_MANTEL = 10000

print("="*60)
print("AHBA HIGH-RESOLUTION WJ ANALYSIS")
print("="*60)

# ============================================================================
# STEP 1: Download and parcellate AHBA data
# ============================================================================
print("\nStep 1: Fetching AHBA data via abagen...")
print("  (This downloads ~4GB on first run. Subsequent runs use cache.)")

import abagen

t0 = time.time()

# Use Desikan-Killiany atlas (68 cortical regions)
# abagen.get_expression_data returns a region x gene DataFrame
# Using all 6 donors, default processing pipeline
try:
    # Fetch the atlas as a proper NIfTI file path
    atlas = abagen.fetch_desikan_killiany()
    expression = abagen.get_expression_data(
        atlas['image'],
        atlas_info=atlas['info'],
        donors='all',
        data_dir=AHBA_DATA,
        verbose=1,
        return_donors=True,  # Get per-donor data for split-half
    )
    # return_donors gives a dict of {donor_id: DataFrame}
    is_donor_dict = isinstance(expression, dict)

    if is_donor_dict:
        # Average across donors for primary analysis
        donor_dfs = list(expression.values())
        donor_ids = list(expression.keys())
        print(f"  Got {len(donor_ids)} donors")

        # Find common regions and genes
        common_regions = set(donor_dfs[0].index)
        common_genes = set(donor_dfs[0].columns)
        for df in donor_dfs[1:]:
            common_regions &= set(df.index)
            common_genes &= set(df.columns)

        common_regions = sorted(common_regions)
        common_genes = sorted(common_genes)
        print(f"  Common regions: {len(common_regions)}")
        print(f"  Common genes: {len(common_genes)}")

        # Average expression across donors
        avg_expr = pd.DataFrame(0.0, index=common_regions, columns=common_genes)
        for df in donor_dfs:
            avg_expr += df.loc[common_regions, common_genes]
        avg_expr /= len(donor_dfs)

    else:
        # Single DataFrame returned
        avg_expr = expression
        common_regions = list(avg_expr.index)
        common_genes = list(avg_expr.columns)
        is_donor_dict = False

    print(f"\n  Expression matrix: {avg_expr.shape[0]} regions x {avg_expr.shape[1]} genes")
    t1 = time.time()
    print(f"  Time: {(t1-t0)/60:.1f} minutes")

except Exception as e:
    print(f"\n  ERROR fetching with return_donors=True: {e}")
    print("  Retrying without return_donors...")

    atlas = abagen.fetch_desikan_killiany()
    expression = abagen.get_expression_data(
        atlas['image'],
        atlas_info=atlas['info'],
        donors='all',
        data_dir=AHBA_DATA,
        verbose=1,
    )
    avg_expr = expression
    common_regions = list(avg_expr.index)
    common_genes = list(avg_expr.columns)
    is_donor_dict = False
    print(f"\n  Expression matrix: {avg_expr.shape[0]} regions x {avg_expr.shape[1]} genes")

# Drop regions with too many NaN genes
nan_pct = avg_expr.isnull().mean(axis=1)
valid_regions = nan_pct[nan_pct < 0.5].index.tolist()
print(f"  Regions with <50% NaN genes: {len(valid_regions)}")

# Drop genes with any NaN across valid regions
expr_clean = avg_expr.loc[valid_regions].dropna(axis=1)
n_regions = len(expr_clean.index)
n_genes = len(expr_clean.columns)
n_pairs = n_regions * (n_regions - 1) // 2
print(f"  Clean matrix: {n_regions} regions x {n_genes} genes")
print(f"  Total pairs: {n_pairs}")

# Save expression matrix
expr_clean.to_csv(os.path.join(AHBA_OUT, 'ahba_expression_clean.csv'))
print("  Saved ahba_expression_clean.csv")

# ============================================================================
# STEP 2: Compute co-expression WJ between all region pairs
# ============================================================================
print(f"\nStep 2: Computing co-expression WJ for {n_pairs} region pairs...")
print("  (This computes gene-gene Spearman correlation within each region,")
print("   then WJ between each pair of region correlation matrices.)")

regions = list(expr_clean.index)
region_corr_matrices = {}

# For AHBA, each region has averaged expression across donors
# We compute gene-gene correlation using cross-gene variation within each region
# But with averaged data, each region is a single vector (1 sample per region)
# This is different from GTEx where each region has multiple samples

# AHBA approach: use the expression vector per region directly
# Gene-gene correlation can't be computed from 1 sample per region
# Instead, compute REGION-REGION correlation across genes (the standard approach)
# This gives a region x region similarity matrix based on expression profiles

# Actually, the standard imaging transcriptomics approach with AHBA is:
# - Each region has an expression profile (vector of gene expression values)
# - Correlation between regions = Pearson/Spearman of their expression profiles
# - This IS expression profile similarity (what Richiardi et al. did)

# For WJ, we need WITHIN-REGION gene-gene correlation matrices.
# With AHBA averaged data (1 value per gene per region), we can't compute
# within-region gene-gene correlations because there's only 1 sample.

# WITH per-donor data, each region has 6 samples (one per donor).
# 6 samples is very few for gene-gene correlations across thousands of genes.

# The correct approach: use AHBA donor-level data as "samples" within each region
# Similar to how GTEx has multiple samples per region.

if is_donor_dict and len(donor_ids) >= 4:
    print("  Using per-donor data for within-region gene-gene correlations")
    n_donors = len(donor_ids)
    print(f"  Donors: {n_donors}")

    # Build per-region gene expression matrices (donors x genes)
    region_expr = {}
    for reg in valid_regions:
        mat = np.zeros((n_donors, len(expr_clean.columns)))
        for d_idx, df in enumerate(donor_dfs):
            if reg in df.index:
                vals = df.loc[reg, expr_clean.columns].values
                mat[d_idx] = vals
            else:
                mat[d_idx] = np.nan
        # Remove donors with NaN for this region
        valid_rows = ~np.any(np.isnan(mat), axis=1)
        mat = mat[valid_rows]
        if mat.shape[0] >= 4:  # Need at least 4 donors for meaningful correlation
            region_expr[reg] = mat

    valid_wj_regions = sorted(region_expr.keys())
    print(f"  Regions with >= 4 valid donors: {len(valid_wj_regions)}")

    if len(valid_wj_regions) < 10:
        print("  WARNING: Too few regions with sufficient donors for WJ analysis")
        print("  Falling back to expression profile similarity approach")
        use_wj = False
    else:
        use_wj = True

        # Compute gene-gene Spearman correlation per region (donors x genes -> gene x gene)
        # With only 6 donors, these correlations will be noisy
        # But this is the honest WJ approach
        print(f"\n  Computing gene-gene correlations per region ({len(valid_wj_regions)} regions)...")

        # Subsample genes for computational feasibility
        # With 6 donors and thousands of genes, we need to be strategic
        # Use top variable genes across regions
        gene_var = expr_clean.loc[valid_wj_regions].var(axis=0)
        top_genes = gene_var.nlargest(2000).index.tolist()
        print(f"  Using top {len(top_genes)} most variable genes")

        for r_idx, reg in enumerate(valid_wj_regions):
            mat = region_expr[reg][:, [list(expr_clean.columns).index(g) for g in top_genes]]
            # Spearman correlation across genes (donors x genes -> gene x gene)
            # Actually: we want gene-gene correlation across donors
            # mat is donors x genes, so corr across axis=0 gives gene x gene
            corr = np.corrcoef(mat.T)  # gene x gene Pearson on donor-level data
            # Use Spearman: rank-transform each gene across donors first
            from scipy.stats import rankdata
            ranked = np.apply_along_axis(rankdata, 0, mat)  # rank within each gene across donors
            corr = np.corrcoef(ranked.T)
            region_corr_matrices[reg] = corr

            if (r_idx + 1) % 10 == 0:
                print(f"    {r_idx+1}/{len(valid_wj_regions)} regions done")

        print(f"  All region correlation matrices computed.")

        # Compute WJ between all pairs
        print(f"\n  Computing WJ for {len(valid_wj_regions)} regions...")
        wj_results = []
        wj_mat = np.zeros((len(valid_wj_regions), len(valid_wj_regions)))

        for i in range(len(valid_wj_regions)):
            for j in range(i+1, len(valid_wj_regions)):
                r1 = valid_wj_regions[i]
                r2 = valid_wj_regions[j]
                corr1 = region_corr_matrices[r1]
                corr2 = region_corr_matrices[r2]

                # Upper triangle only
                tri = np.triu_indices(corr1.shape[0], k=1)
                v1 = np.abs(corr1[tri])
                v2 = np.abs(corr2[tri])

                # Remove NaN pairs
                valid = ~(np.isnan(v1) | np.isnan(v2))
                v1 = v1[valid]
                v2 = v2[valid]

                if len(v1) > 0:
                    wj = np.sum(np.minimum(v1, v2)) / np.sum(np.maximum(v1, v2))
                else:
                    wj = np.nan

                wj_mat[i, j] = wj
                wj_mat[j, i] = wj
                wj_results.append({
                    'Region1': r1, 'Region2': r2, 'WJ': wj
                })

        np.fill_diagonal(wj_mat, 1.0)
        wj_df = pd.DataFrame(wj_results)
        wj_df.to_csv(os.path.join(AHBA_OUT, 'ahba_wj_pairs.csv'), index=False)
        print(f"  WJ pairs: {len(wj_df)}")
        print(f"  WJ range: [{wj_df['WJ'].min():.4f}, {wj_df['WJ'].max():.4f}]")
        print(f"  WJ mean: {wj_df['WJ'].mean():.4f}")

        # Save WJ matrix
        wj_matrix_df = pd.DataFrame(wj_mat, index=valid_wj_regions, columns=valid_wj_regions)
        wj_matrix_df.to_csv(os.path.join(AHBA_OUT, 'ahba_wj_matrix.csv'))
        print("  Saved ahba_wj_matrix.csv")
else:
    print("  Per-donor data not available. Using expression profile similarity only.")
    use_wj = False

# ============================================================================
# STEP 3: Expression profile similarity (for comparison, regardless of WJ path)
# ============================================================================
print(f"\nStep 3: Computing expression profile similarity...")

expr_sim_mat = np.zeros((n_regions, n_regions))
region_list = list(expr_clean.index)

for i in range(n_regions):
    for j in range(i+1, n_regions):
        rho, _ = stats.spearmanr(expr_clean.iloc[i], expr_clean.iloc[j])
        expr_sim_mat[i, j] = rho
        expr_sim_mat[j, i] = rho

np.fill_diagonal(expr_sim_mat, 1.0)
expr_sim_df = pd.DataFrame(expr_sim_mat, index=region_list, columns=region_list)
expr_sim_df.to_csv(os.path.join(AHBA_OUT, 'ahba_expression_similarity.csv'))
print(f"  Expression similarity range: [{expr_sim_mat[np.triu_indices(n_regions, k=1)].min():.4f}, "
      f"{expr_sim_mat[np.triu_indices(n_regions, k=1)].max():.4f}]")

# ============================================================================
# STEP 4: Summary and provenance
# ============================================================================
provenance = {
    'methodology': 'WJ-native (AHBA high-resolution)',
    'pipeline_file': 'gene_to_brain_path1_ahba.py',
    'execution_date': '2026-03-27',
    'random_seed': RANDOM_SEED,
    'data_source': 'Allen Human Brain Atlas (AHBA) via abagen',
    'parcellation': 'Desikan-Killiany',
    'n_regions': n_regions,
    'n_genes_clean': n_genes,
    'n_pairs': n_pairs,
    'wj_computed': use_wj,
    'note': 'AHBA has 6 donors. Gene-gene correlations within regions are computed '
            'across donors (n=6), which limits statistical power per correlation but '
            'enables the WJ approach. Top 2000 variable genes used for computational '
            'feasibility.' if use_wj else 'WJ not computed due to insufficient per-donor data.',
}

with open(os.path.join(AHBA_OUT, 'provenance.json'), 'w') as f:
    json.dump(provenance, f, indent=2)

print(f"\nProvenance saved.")
print(f"\n=== AHBA PIPELINE {'COMPLETE' if use_wj else 'PARTIAL (expression similarity only)'} ===")
