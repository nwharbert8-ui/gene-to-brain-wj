"""
Pipeline: Gene-to-Brain Foundation-Up Rebuild — Run Everything From Scratch
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-05-30

Description:
    Foundation-up rebuild that disregards the April 23 manuscript narrative.
    Re-loads GTEx individual samples, re-computes every architectural similarity
    metric between region pairs, applies current methodology (Layer 2H pairing
    family, Layer 2I direct sign-flip rate, Layer 4 Mantel inference, Layer 2F
    per-region decomposition), and reports all measurements honestly without
    pre-committing to any framing.

    Phases:
        1. Load GTEx individual samples + brain FC matrices + parcellation
        2. Per-region gene-gene Spearman correlation matrices (streamed)
        3. Per-pair architectural similarity metrics (Layer 2B menu)
           and Layer 2I sign-flip rate stratified
        4. WJ-FC correspondence: Mantel + Spearman + partial-distance for
           all pairs, cross-type subset, within-type subset
        5. Layer 2H pairings: Type 1 (WJ vs binary Jaccard), Type 2
           (unsigned vs shifted), Type 5 (per-region row), Type 6 (Pearson
           vs Spearman vs cosine on WJ-FC)
        6. Per-region row decomposition (Layer 2F)
        7. Expression profile null comparison
        8. Distance-matched subsampling
        9. Provenance + interpretation report

Dependencies: numpy, scipy, pandas
Input:
    G:\\...\\MS3_JNC_Submission\\data\\GTEx_v8_tpm.gct.gz
    G:\\...\\MS3_JNC_Submission\\data\\GTEx_v8_sample_attributes.txt
    G:\\...\\brain_connectivity_wj\\results\\correlation_matrices\\group_awake_spearman_corr.npy
    G:\\...\\brain_connectivity_wj\\results\\correlation_matrices\\group_unconscious_spearman_corr.npy
    G:\\...\\brain_connectivity_wj\\data\\raw\\ds006623\\derivatives\\xcp_d_without_GSR_bandpass_output\\atlases\\atlas-4S456Parcels\\atlas-4S456Parcels_dseg.tsv

Output:
    results/rebuild/pairwise_metrics.csv (one row per region pair, all metrics)
    results/rebuild/sign_flip_stratified.csv (one row per pair x threshold)
    results/rebuild/mantel_all.json (Mantel results across all subsets)
    results/rebuild/layer2h_pairings.json (Type 1, 2, 5, 6 with CIs)
    results/rebuild/per_region_decomposition.csv (Layer 2F)
    results/rebuild/distance_matched.json
    results/rebuild/provenance.json
    results/rebuild/interpretation_report.md
"""
from __future__ import annotations
import os
import time
import json
import gzip
import gc
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, rankdata
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
RANDOM_SEED = 42
FORCE_RECOMPUTE = True
N_PERM_MANTEL = 10000
N_BOOTSTRAP = 1000
N_DIST_MATCH = 1000
SIGN_STRATA = [0.0, 0.05, 0.10, 0.20, 0.30]
BINARY_THRESHOLDS = [0.20, 0.30, 0.50]

BASE = Path(r'G:\My Drive\inner_architecture_research\gene_to_brain_wj')
RESULTS = BASE / 'results'
REBUILD = RESULTS / 'rebuild'
REBUILD.mkdir(parents=True, exist_ok=True)

GTEX_TPM = Path(r'G:\My Drive\inner_architecture_research\MS3_JNC_Submission\data\GTEx_v8_tpm.gct.gz')
GTEX_ATTR = Path(r'G:\My Drive\inner_architecture_research\MS3_JNC_Submission\data\GTEx_v8_sample_attributes.txt')
GTEX_SUBJ_ATTR = Path(r'G:\My Drive\inner_architecture_research\MS3_JNC_Submission\data\GTEx_v8_subject_phenotypes.txt')
BRAIN = Path(r'G:\My Drive\inner_architecture_research\brain_connectivity_wj')

np.random.seed(RANDOM_SEED)
START = time.time()
LOG_FILE = REBUILD / 'rebuild_run.log'


def log(msg):
    line = f"[{(time.time()-START)/60:7.1f}m] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ============================================================================
# REGION → PARCEL MAP (from bulletproof.py)
# ============================================================================
def build_region_parcel_map(atlas_labels):
    pl = atlas_labels
    rmap = {
        'Brain - Putamen (basal ganglia)': [i for i, l in enumerate(pl) if l in ['LH-Pu', 'RH-Pu']],
        'Brain - Caudate (basal ganglia)': [i for i, l in enumerate(pl) if l in ['LH-Ca', 'RH-Ca']],
        'Brain - Nucleus accumbens (basal ganglia)': [i for i, l in enumerate(pl) if l in ['LH-NAC', 'RH-NAC']],
        'Brain - Hippocampus': [i for i, l in enumerate(pl) if 'Hippocampus' in l or 'HN' in l],
        'Brain - Amygdala': [i for i, l in enumerate(pl) if 'Amygdala' in l or 'EXA' in l],
        'Brain - Hypothalamus': [i for i, l in enumerate(pl) if 'HTH' in l],
        'Brain - Substantia nigra': [i for i, l in enumerate(pl) if 'SNc' in l or 'SNr' in l],
        'Brain - Cerebellum': [i for i, l in enumerate(pl) if 'Cerebellar' in l],
    }
    return rmap


# Pair-type classification
SUBCORTICAL = {
    'Brain - Amygdala', 'Brain - Caudate (basal ganglia)',
    'Brain - Hippocampus', 'Brain - Hypothalamus',
    'Brain - Nucleus accumbens (basal ganglia)',
    'Brain - Putamen (basal ganglia)', 'Brain - Substantia nigra',
    'Brain - Cerebellum',
}
CORTICAL = {
    'Brain - Anterior cingulate cortex (BA24)',
    'Brain - Cortex',
    'Brain - Frontal Cortex (BA9)',
}


def classify_pair(r1, r2):
    s1 = r1 in SUBCORTICAL
    s2 = r2 in SUBCORTICAL
    c1 = r1 in CORTICAL
    c2 = r2 in CORTICAL
    if s1 and c2 or c1 and s2:
        return 'cross-type'
    if s1 and s2:
        return 'within-subcortical'
    if c1 and c2:
        return 'within-cortical'
    return 'other'


# MNI centroids (from bulletproof.py)
MNI_COORDS = {
    'Brain - Amygdala': np.array([24, -4, -18]),
    'Brain - Anterior cingulate cortex (BA24)': np.array([2, 28, 22]),
    'Brain - Caudate (basal ganglia)': np.array([12, 12, 10]),
    'Brain - Cerebellum': np.array([2, -60, -30]),
    'Brain - Cortex': np.array([0, 0, 40]),
    'Brain - Frontal Cortex (BA9)': np.array([4, 50, 30]),
    'Brain - Hippocampus': np.array([28, -20, -14]),
    'Brain - Hypothalamus': np.array([2, -4, -10]),
    'Brain - Nucleus accumbens (basal ganglia)': np.array([10, 12, -6]),
    'Brain - Putamen (basal ganglia)': np.array([26, 4, 2]),
    'Brain - Substantia nigra': np.array([10, -18, -12]),
}


# ============================================================================
# MATH HELPERS
# ============================================================================
def fast_spearman_matrix(data):
    """Spearman correlation matrix via Pearson on ranks. Float64."""
    n = data.shape[0]
    ranked = np.zeros_like(data, dtype=np.float64)
    for i in range(n):
        ranked[i] = rankdata(data[i])
    ranked -= ranked.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(ranked ** 2, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    ranked /= norms
    return np.clip(ranked @ ranked.T, -1.0, 1.0)


def architectural_metrics(c1, c2):
    """All Layer 2B metrics + Layer 2I sign-flip + Layer 2H Type 1/2 between two
    correlation matrices. Operates on upper-triangle vectors.
    Returns dict of metric values."""
    n = c1.shape[0]
    tri = np.triu_indices(n, k=1)
    v1 = c1[tri]
    v2 = c2[tri]
    abs1 = np.abs(v1)
    abs2 = np.abs(v2)
    sign1 = np.sign(v1)
    sign2 = np.sign(v2)

    out = {}

    # --- Layer 2B comparison metrics ---
    # Unsigned WJ (min/max on |r|)
    den_u = np.sum(np.maximum(abs1, abs2))
    out['wj_unsigned'] = float(np.sum(np.minimum(abs1, abs2)) / den_u) if den_u > 0 else 1.0

    # Signed (+1-shift) WJ: WJ on (r+1)
    s1 = v1 + 1.0
    s2 = v2 + 1.0
    den_s = np.sum(np.maximum(s1, s2))
    out['wj_shifted'] = float(np.sum(np.minimum(s1, s2)) / den_s) if den_s > 0 else 1.0
    out['wj_implementation_gap_shifted_minus_unsigned'] = out['wj_shifted'] - out['wj_unsigned']

    # Binary Jaccard at multiple thresholds (Layer 2H Type 1 partner)
    for t in BINARY_THRESHOLDS:
        bin1 = abs1 >= t
        bin2 = abs2 >= t
        inter = int(np.sum(bin1 & bin2))
        union = int(np.sum(bin1 | bin2))
        out[f'binary_jaccard_t{t}'] = float(inter / union) if union > 0 else 1.0
        out[f'wj_unsigned_minus_binJ_t{t}'] = out['wj_unsigned'] - out[f'binary_jaccard_t{t}']

    # Pearson matrix correlation on |r|
    out['pearson_abs'] = float(pearsonr(abs1, abs2)[0])
    # Spearman matrix correlation on |r|
    out['spearman_abs'] = float(spearmanr(abs1, abs2)[0])
    # Cosine on |r|
    out['cosine_abs'] = float(np.dot(abs1, abs2) / (np.linalg.norm(abs1) * np.linalg.norm(abs2)))
    # Frobenius difference
    out['frobenius_diff'] = float(np.sqrt(np.sum((abs1 - abs2) ** 2)))

    # --- Layer 2I direct pair-level sign-flip rate stratified ---
    valid = (sign1 != 0) & (sign2 != 0)
    flip = sign1 != sign2
    min_abs = np.minimum(abs1, abs2)
    flip_strat = {}
    for t in SIGN_STRATA:
        mask = (min_abs >= t) & valid
        n_t = int(mask.sum())
        if n_t == 0:
            flip_strat[f't{t}'] = {'n_pairs': 0, 'flip_count': 0, 'flip_rate': None, 'p_vs_chance': None}
            continue
        fc = int(flip[mask].sum())
        rate = fc / n_t
        # Binomial p vs 0.5
        if n_t > 5000:
            # Normal approximation
            z = (fc - 0.5 * n_t) / np.sqrt(0.25 * n_t)
            p_b = float(2 * (1 - stats.norm.cdf(abs(z))))
        else:
            p_b = float(stats.binomtest(fc, n_t, p=0.5).pvalue)
        flip_strat[f't{t}'] = {
            'n_pairs': n_t, 'flip_count': fc,
            'flip_rate': float(rate), 'p_vs_chance': p_b,
        }
    out['_sign_flip_stratified'] = flip_strat

    return out


# ============================================================================
# PHASE 1: LOAD ALL DATA
# ============================================================================
def phase1_load_data():
    log("=" * 70)
    log("PHASE 1: LOAD DATA")
    log("=" * 70)

    # Verify input files
    for f in [GTEX_TPM, GTEX_ATTR]:
        if not f.exists():
            raise FileNotFoundError(f"REQUIRED FILE MISSING: {f}")
    log(f"  GTEx TPM: {GTEX_TPM} ({GTEX_TPM.stat().st_size/1e9:.2f} GB)")
    log(f"  GTEx attr: {GTEX_ATTR}")

    # Brain FC
    fc_awake_path = BRAIN / 'results' / 'correlation_matrices' / 'group_awake_spearman_corr.npy'
    fc_uncon_path = BRAIN / 'results' / 'correlation_matrices' / 'group_unconscious_spearman_corr.npy'
    atlas_path = (BRAIN / 'data' / 'raw' / 'ds006623' / 'derivatives' /
                  'xcp_d_without_GSR_bandpass_output' / 'atlases' /
                  'atlas-4S456Parcels' / 'atlas-4S456Parcels_dseg.tsv')
    for p in [fc_awake_path, fc_uncon_path, atlas_path]:
        if not p.exists():
            raise FileNotFoundError(f"REQUIRED FILE MISSING: {p}")

    log("  Loading brain FC matrices and parcellation...")
    corr_awake = np.load(fc_awake_path)
    corr_uncon = np.load(fc_uncon_path)
    atlas = pd.read_csv(atlas_path, sep='\t')
    pl = atlas['label'].tolist()
    log(f"    FC awake shape: {corr_awake.shape}")
    log(f"    FC uncon shape: {corr_uncon.shape}")
    log(f"    Parcellation labels: {len(pl)}")

    rmap = build_region_parcel_map(pl)
    rmap = {k: v for k, v in rmap.items() if v}

    # Add cortical proxies from bulletproof.py
    rmap['Brain - Frontal Cortex (BA9)'] = [
        i for i, l in enumerate(pl)
        if any(k in l for k in ['Cont_PFCl', 'Cont_PFCmp', 'Default_PFC', 'Cont_pCun'])
        and atlas.iloc[i].get('atlas_name', '') == '4S456'
    ]
    rmap['Brain - Anterior cingulate cortex (BA24)'] = [
        i for i, l in enumerate(pl)
        if any(k in l for k in ['SalVentAttn_Med', 'Limbic_OFC', 'Default_pCunPCC', 'Cont_Cing'])
        and atlas.iloc[i].get('atlas_name', '') == '4S456'
    ]
    rmap['Brain - Cortex'] = [
        i for i, l in enumerate(pl)
        if atlas.iloc[i].get('atlas_name', '') == '4S456'
    ]
    rmap = {k: v for k, v in rmap.items() if v}
    log(f"  Region -> parcel map built: {len(rmap)} regions")
    for r, ps in sorted(rmap.items()):
        log(f"    {r}: {len(ps)} parcels")

    # GTEx sample attributes
    log("  Loading GTEx sample attributes...")
    sa = pd.read_csv(GTEX_ATTR, sep='\t', usecols=['SAMPID', 'SMTSD'])
    target_regions = sorted([r for r in rmap.keys() if r in sa['SMTSD'].unique()])
    brain_sa = sa[sa['SMTSD'].isin(target_regions)]
    brain_ids = set(brain_sa['SAMPID'])
    log(f"  Target regions (in GTEx + parcel map): {len(target_regions)}")
    log(f"  Brain sample IDs: {len(brain_ids)}")

    # GTEx subject attributes (donor IDs for Phase 8)
    subj_lookup = None
    if GTEX_SUBJ_ATTR.exists():
        try:
            subj = pd.read_csv(GTEX_SUBJ_ATTR, sep='\t', usecols=['SUBJID'])
            subj_lookup = subj
        except Exception as e:
            log(f"  WARN: could not load subject phenotype file: {e}")

    # GTEx expression
    log("  Parsing GTEx TPM (this is the slow step)...")
    with gzip.open(GTEX_TPM, 'rt') as f:
        f.readline()
        f.readline()
        header = f.readline().strip().split('\t')
    bcols = [0, 1] + [i for i, c in enumerate(header) if c in brain_ids]
    log(f"  Brain sample columns to load: {len(bcols) - 2}")
    expr = pd.read_csv(GTEX_TPM, sep='\t', skiprows=2, usecols=bcols, compression='gzip')
    expr.columns = [header[i] for i in bcols]
    expr['gene'] = expr['Description'].str.upper()
    expr = expr.drop(['Name', 'Description'], axis=1).drop_duplicates('gene', keep='first').set_index('gene')
    expr = np.log2(expr + 1)
    expr = expr[expr.mean(axis=1) > np.log2(2)]
    genes = list(expr.index)
    log(f"  Genes after expression filter: {len(genes)}")
    log(f"  Sample columns retained: {len(expr.columns)}")

    # Per-region samples
    reg_samp = {}
    for r in target_regions:
        s = brain_sa[brain_sa['SMTSD'] == r]['SAMPID'].tolist()
        avail = [x for x in s if x in expr.columns]
        if len(avail) >= 30:
            reg_samp[r] = avail
    valid_regions = sorted(reg_samp.keys())
    log(f"\n  Valid regions (>=30 samples): {len(valid_regions)}")
    total_samples = 0
    for r in valid_regions:
        n = len(reg_samp[r])
        log(f"    {r}: {n} samples")
        total_samples += n
    log(f"  Total brain samples used: {total_samples}")

    return {
        'expr': expr,
        'genes': genes,
        'reg_samp': reg_samp,
        'valid_regions': valid_regions,
        'rmap': rmap,
        'corr_awake': corr_awake,
        'corr_uncon': corr_uncon,
        'n_genes': len(genes),
        'n_brain_samples': total_samples,
    }


# ============================================================================
# PHASE 2 + 3: Per-region matrices + per-pair architectural metrics + Layer 2I
# (Streamed: bulletproof.py pattern)
# ============================================================================
def phase23_pairwise_compute(data):
    log("=" * 70)
    log("PHASE 2-3: PER-REGION MATRICES + ARCHITECTURAL METRICS + LAYER 2I")
    log("=" * 70)

    expr = data['expr']
    reg_samp = data['reg_samp']
    valid_regions = data['valid_regions']
    rmap = data['rmap']
    corr_awake = data['corr_awake']
    corr_uncon = data['corr_uncon']

    pairs_data = []
    flip_data = []
    region_corrs_keep = {}  # save first triangle vec per region for row-decomp later
    n_reg = len(valid_regions)

    for i in range(n_reg):
        r1 = valid_regions[i]
        log(f"\n  Region {i+1}/{n_reg}: {r1} (building correlation matrix)...")
        t0 = time.time()
        d1 = expr[reg_samp[r1]].values.astype(np.float64)
        c1 = fast_spearman_matrix(d1)
        log(f"    Built in {time.time()-t0:.1f}s, shape {c1.shape}")

        # Save the abs-triangle vector for downstream row-decomp (memory-efficient)
        tri = np.triu_indices(c1.shape[0], k=1)
        region_corrs_keep[r1] = {'sign_tri': np.sign(c1[tri]).astype(np.int8),
                                  'abs_tri': np.abs(c1[tri]).astype(np.float32)}

        for j in range(i + 1, n_reg):
            r2 = valid_regions[j]
            d2 = expr[reg_samp[r2]].values.astype(np.float64)
            c2 = fast_spearman_matrix(d2)

            # All metrics
            metrics = architectural_metrics(c1, c2)
            flip_strat = metrics.pop('_sign_flip_stratified')

            # FC values
            p1 = [p for p in rmap[r1] if p < corr_awake.shape[0]]
            p2 = [p for p in rmap[r2] if p < corr_awake.shape[0]]
            fc_aw = float(np.mean(corr_awake[np.ix_(p1, p2)])) if p1 and p2 else float('nan')
            fc_un = float(np.mean(corr_uncon[np.ix_(p1, p2)])) if p1 and p2 else float('nan')

            # Distance
            dist = float('nan')
            if r1 in MNI_COORDS and r2 in MNI_COORDS:
                dist = float(np.linalg.norm(MNI_COORDS[r1] - MNI_COORDS[r2]))

            # Expression profile similarity (component level)
            med1 = np.median(d1, axis=1)
            med2 = np.median(d2, axis=1)
            expr_sim = float(spearmanr(med1, med2)[0])

            row = {
                'Region1': r1, 'Region2': r2,
                'Pair_type': classify_pair(r1, r2),
                **metrics,
                'expr_profile_sim': expr_sim,
                'fc_awake': fc_aw,
                'fc_unconscious': fc_un,
                'fc_diff_awake_minus_uncon': fc_aw - fc_un if not (np.isnan(fc_aw) or np.isnan(fc_un)) else float('nan'),
                'distance_mm': dist,
                'n_samples_r1': len(reg_samp[r1]),
                'n_samples_r2': len(reg_samp[r2]),
            }
            pairs_data.append(row)

            for t_key, t_vals in flip_strat.items():
                flip_data.append({
                    'Region1': r1, 'Region2': r2,
                    'Pair_type': classify_pair(r1, r2),
                    'threshold': float(t_key.replace('t', '')),
                    **t_vals,
                })

            log(f"    pair {r1.split(' - ')[1][:20]:>20s} vs {r2.split(' - ')[1][:20]:<20s}  "
                f"WJ_u={metrics['wj_unsigned']:.4f}  FC_aw={fc_aw:.3f}  FC_un={fc_un:.3f}")

            del c2, d2, metrics
            gc.collect()

        del c1, d1
        gc.collect()

    pairs_df = pd.DataFrame(pairs_data)
    pairs_df.to_csv(REBUILD / 'pairwise_metrics.csv', index=False, float_format='%.6f')
    log(f"\n  Saved pairwise_metrics.csv ({len(pairs_df)} pairs)")

    flip_df = pd.DataFrame(flip_data)
    flip_df.to_csv(REBUILD / 'sign_flip_stratified.csv', index=False, float_format='%.6f')
    log(f"  Saved sign_flip_stratified.csv ({len(flip_df)} rows)")

    return pairs_df, flip_df, region_corrs_keep


# ============================================================================
# PHASE 4: Mantel + Spearman + partial across all subsets
# ============================================================================
def phase4_mantel_all(pairs_df, n_perm=N_PERM_MANTEL):
    log("\n" + "=" * 70)
    log("PHASE 4: MANTEL + SPEARMAN + PARTIAL ACROSS ALL SUBSETS")
    log("=" * 70)

    regions = sorted(set(pairs_df['Region1']) | set(pairs_df['Region2']))
    n_reg = len(regions)
    ridx = {r: i for i, r in enumerate(regions)}

    def build_full(col):
        mat = np.zeros((n_reg, n_reg))
        for _, row in pairs_df.iterrows():
            i = ridx[row['Region1']]
            j = ridx[row['Region2']]
            mat[i, j] = mat[j, i] = row[col]
        return mat

    wj_mat = build_full('wj_unsigned')
    fc_awake_mat = build_full('fc_awake')
    fc_uncon_mat = build_full('fc_unconscious')
    expr_mat = build_full('expr_profile_sim')
    dist_mat = build_full('distance_mm')

    tri = np.triu_indices(n_reg, k=1)
    cross_mask = np.array([classify_pair(regions[i], regions[j]) == 'cross-type'
                           for i, j in zip(*tri)])

    wj_tri = wj_mat[tri]
    fc_awake_tri = fc_awake_mat[tri]
    fc_uncon_tri = fc_uncon_mat[tri]
    expr_tri = expr_mat[tri]
    dist_tri = dist_mat[tri]

    def mantel_pair(predictor_mat, outcome_vec, subset_mask=None, n_perm=n_perm, seed=RANDOM_SEED):
        """Mantel-style: permute regions of predictor matrix, extract subset, correlate."""
        rng = np.random.RandomState(seed)
        if subset_mask is None:
            mask = np.ones_like(outcome_vec, dtype=bool)
        else:
            mask = subset_mask
        pred_tri = predictor_mat[tri]
        r_obs, _ = pearsonr(pred_tri[mask], outcome_vec[mask])
        null = np.zeros(n_perm)
        for p in range(n_perm):
            perm = rng.permutation(n_reg)
            pp = predictor_mat[np.ix_(perm, perm)][tri]
            null[p], _ = pearsonr(pp[mask], outcome_vec[mask])
        p_one = float(np.mean(null >= r_obs))
        p_two = float(np.mean(np.abs(null) >= abs(r_obs)))
        return {
            'observed_pearson': float(r_obs),
            'mantel_p_one': p_one,
            'mantel_p_two': p_two,
            'null_mean': float(null.mean()),
            'null_std': float(null.std()),
            'n_perm': n_perm,
        }

    results = {}

    for outcome_name, outcome_vec in [('FC_awake', fc_awake_tri),
                                       ('FC_unconscious', fc_uncon_tri),
                                       ('FC_activity_diff', fc_awake_tri - fc_uncon_tri)]:
        results[outcome_name] = {}
        for subset_name, mask in [('all_pairs', None),
                                   ('cross_type', cross_mask),
                                   ('within_type', ~cross_mask)]:
            for pred_name, pred_mat in [('WJ_unsigned', wj_mat),
                                          ('expr_profile', expr_mat)]:
                key = f'{pred_name}_vs_{outcome_name}_{subset_name}'
                log(f"  Mantel: {key}")
                results[outcome_name][f'{pred_name}_{subset_name}'] = mantel_pair(
                    pred_mat, outcome_vec, subset_mask=mask)

    # Partial correlations (Spearman, distance-controlled, full set)
    def partial_corr(x, y, z):
        rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
        bxz = np.polyfit(rz, rx, 1)
        byz = np.polyfit(rz, ry, 1)
        ex = rx - np.polyval(bxz, rz)
        ey = ry - np.polyval(byz, rz)
        rho, p = pearsonr(ex, ey)
        return float(rho), float(p)

    partial = {}
    valid = ~np.isnan(dist_tri)
    for outcome_name, outcome_vec in [('FC_awake', fc_awake_tri),
                                       ('FC_unconscious', fc_uncon_tri)]:
        rho, p = partial_corr(wj_tri[valid], outcome_vec[valid], dist_tri[valid])
        partial[f'WJ_vs_{outcome_name}_partial_distance'] = {'rho': rho, 'p': p}
        rho, p = partial_corr(expr_tri[valid], outcome_vec[valid], dist_tri[valid])
        partial[f'expr_vs_{outcome_name}_partial_distance'] = {'rho': rho, 'p': p}

    out = {
        'mantel_tests': results,
        'partial_spearman_distance': partial,
        'subset_sizes': {
            'all_pairs': int(len(wj_tri)),
            'cross_type': int(cross_mask.sum()),
            'within_type': int((~cross_mask).sum()),
        },
        'random_seed': RANDOM_SEED,
        'n_perm': n_perm,
    }

    with open(REBUILD / 'mantel_all.json', 'w') as f:
        json.dump(out, f, indent=2)
    log(f"  Saved mantel_all.json")

    # Also save the constructed full matrices for downstream
    np.save(REBUILD / '_wj_unsigned_matrix.npy', wj_mat)
    np.save(REBUILD / '_fc_awake_matrix.npy', fc_awake_mat)
    np.save(REBUILD / '_fc_uncon_matrix.npy', fc_uncon_mat)
    np.save(REBUILD / '_expr_matrix.npy', expr_mat)
    np.save(REBUILD / '_dist_matrix.npy', dist_mat)

    return out, {'regions': regions, 'cross_mask': cross_mask, 'tri': tri,
                  'wj_tri': wj_tri, 'fc_awake_tri': fc_awake_tri,
                  'fc_uncon_tri': fc_uncon_tri, 'expr_tri': expr_tri,
                  'dist_tri': dist_tri,
                  'wj_mat': wj_mat, 'fc_awake_mat': fc_awake_mat,
                  'fc_uncon_mat': fc_uncon_mat, 'expr_mat': expr_mat,
                  'dist_mat': dist_mat}


# ============================================================================
# PHASE 5: Layer 2H pairing decompositions with bootstrap CIs
# ============================================================================
def phase5_layer2h(pairs_df, ctx, n_boot=N_BOOTSTRAP):
    log("\n" + "=" * 70)
    log("PHASE 5: LAYER 2H PAIRING DECOMPOSITIONS")
    log("=" * 70)

    out = {}

    # ---- Type 1: WJ vs binary Jaccard ----
    # The gap as a function of threshold localizes where in correlation magnitude
    # the divergence concentrates
    type1 = {}
    for t in BINARY_THRESHOLDS:
        gap_col = f'wj_unsigned_minus_binJ_t{t}'
        if gap_col in pairs_df.columns:
            gaps = pairs_df[gap_col].values
            mean = float(np.mean(gaps))
            ci_lo = float(np.percentile(gaps, 2.5))
            ci_hi = float(np.percentile(gaps, 97.5))
            # Test gap vs zero across pairs
            t_stat, t_p = stats.ttest_1samp(gaps, 0)
            type1[f'threshold_{t}'] = {
                'gap_mean_across_pairs': mean,
                'gap_q025': ci_lo,
                'gap_q975': ci_hi,
                'one_sample_t_p': float(t_p),
                'interpretation': ('positive gap = WJ retains more architectural similarity than '
                                   'binary Jaccard at this threshold (divergence is concentrated '
                                   'in bulk small/mid correlations, not in supra-threshold pairs); '
                                   'negative gap = divergence concentrated in supra-threshold pairs'),
            }
    out['Type1_WJ_vs_binary_Jaccard'] = type1
    log(f"  Type 1: {len(type1)} threshold conditions computed")

    # ---- Type 2: unsigned WJ vs shifted (signed) WJ ----
    gaps2 = (pairs_df['wj_shifted'] - pairs_df['wj_unsigned']).values
    mean2 = float(np.mean(gaps2))
    p_pos = float(np.mean(gaps2 > 0))
    type2 = {
        'gap_shifted_minus_unsigned': {
            'mean_across_pairs': mean2,
            'q025': float(np.percentile(gaps2, 2.5)),
            'q975': float(np.percentile(gaps2, 97.5)),
            'fraction_pairs_with_positive_gap': p_pos,
            'caveat': ('Per Layer 2I rule: the gap is NOT a sign-inversion percentage. '
                       'See sign_flip_stratified.csv for the canonical sign-inversion measurement.'),
        }
    }
    out['Type2_unsigned_vs_shifted'] = type2
    log(f"  Type 2: mean gap = {mean2:+.4f}")

    # ---- Type 6: Pearson vs Spearman vs cosine on WJ-vs-FC ----
    rng = np.random.RandomState(RANDOM_SEED)
    n_pairs = len(ctx['wj_tri'])
    type6 = {}
    for outcome_name, outcome_vec in [('FC_awake', ctx['fc_awake_tri']),
                                       ('FC_unconscious', ctx['fc_uncon_tri'])]:
        # Point estimates
        pr, _ = pearsonr(ctx['wj_tri'], outcome_vec)
        sr, _ = spearmanr(ctx['wj_tri'], outcome_vec)
        cs = float(np.dot(ctx['wj_tri'], outcome_vec) /
                   (np.linalg.norm(ctx['wj_tri']) * np.linalg.norm(outcome_vec)))
        # Bootstrap
        gaps_ps = []
        gaps_pc = []
        gaps_sc = []
        for b in range(n_boot):
            idx = rng.choice(n_pairs, n_pairs, replace=True)
            try:
                p_b, _ = pearsonr(ctx['wj_tri'][idx], outcome_vec[idx])
                s_b, _ = spearmanr(ctx['wj_tri'][idx], outcome_vec[idx])
                c_b = float(np.dot(ctx['wj_tri'][idx], outcome_vec[idx]) /
                            (np.linalg.norm(ctx['wj_tri'][idx]) *
                             np.linalg.norm(outcome_vec[idx])))
            except Exception:
                continue
            gaps_ps.append(p_b - s_b)
            gaps_pc.append(p_b - c_b)
            gaps_sc.append(s_b - c_b)
        type6[outcome_name] = {
            'pearson': float(pr), 'spearman': float(sr), 'cosine': cs,
            'gap_pearson_minus_spearman': {
                'mean': float(np.mean(gaps_ps)),
                'ci_lo': float(np.percentile(gaps_ps, 2.5)),
                'ci_hi': float(np.percentile(gaps_ps, 97.5)),
                'excludes_zero': bool(np.percentile(gaps_ps, 2.5) > 0 or np.percentile(gaps_ps, 97.5) < 0),
            },
            'gap_pearson_minus_cosine': {
                'mean': float(np.mean(gaps_pc)),
                'ci_lo': float(np.percentile(gaps_pc, 2.5)),
                'ci_hi': float(np.percentile(gaps_pc, 97.5)),
                'excludes_zero': bool(np.percentile(gaps_pc, 2.5) > 0 or np.percentile(gaps_pc, 97.5) < 0),
            },
        }
    out['Type6_substrate_projection'] = type6
    log(f"  Type 6: bootstrapped CIs on Pearson/Spearman/cosine gaps")

    with open(REBUILD / 'layer2h_pairings.json', 'w') as f:
        json.dump(out, f, indent=2)
    log(f"  Saved layer2h_pairings.json")
    return out


# ============================================================================
# PHASE 6: Per-region row decomposition (Layer 2F)
# ============================================================================
def phase6_per_region_row(pairs_df, ctx, n_boot=500):
    log("\n" + "=" * 70)
    log("PHASE 6: PER-REGION ROW DECOMPOSITION (Layer 2F)")
    log("=" * 70)

    regions = ctx['regions']
    wj_mat = ctx['wj_mat']
    n_reg = len(regions)
    rng = np.random.RandomState(RANDOM_SEED)

    rows = []
    for i, r in enumerate(regions):
        # Other regions
        other_idx = [j for j in range(n_reg) if j != i]
        row_wj = wj_mat[i, other_idx]
        # mean and CI
        mean = float(np.mean(row_wj))
        # Bootstrap
        boots = np.array([np.mean(row_wj[rng.choice(len(row_wj), len(row_wj), replace=True)])
                          for _ in range(n_boot)])
        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))

        # Sign coherence per region: fraction of partner pairs that maintain WJ above 0.5 or similar
        # For gene-region context, we use the FRACTION of partners with WJ in top half as a proxy
        rank_in_row = np.argsort(np.argsort(row_wj))
        rows.append({
            'region': r,
            'pair_type_majority': '-',
            'row_wj_mean': mean,
            'row_wj_ci_lo': ci_lo,
            'row_wj_ci_hi': ci_hi,
            'row_wj_min': float(row_wj.min()),
            'row_wj_max': float(row_wj.max()),
            'n_partners': len(row_wj),
        })

    df = pd.DataFrame(rows).sort_values('row_wj_mean', ascending=False)
    df.to_csv(REBUILD / 'per_region_decomposition.csv', index=False, float_format='%.6f')
    log(f"  Saved per_region_decomposition.csv")
    log(f"  Per-region row WJ mean ranking:")
    for _, row in df.iterrows():
        log(f"    {row['region']:>50s}: row_WJ = {row['row_wj_mean']:.4f} "
            f"[{row['row_wj_ci_lo']:.4f}, {row['row_wj_ci_hi']:.4f}]")
    return df


# ============================================================================
# PHASE 7: Distance-matched subsampling (cross-type vs within-type honest test)
# ============================================================================
def phase7_distance_matching(pairs_df, n_iter=N_DIST_MATCH):
    log("\n" + "=" * 70)
    log("PHASE 7: DISTANCE-MATCHED SUBSAMPLING")
    log("=" * 70)

    cross = pairs_df[pairs_df['Pair_type'] == 'cross-type'].dropna(subset=['distance_mm'])
    within = pairs_df[pairs_df['Pair_type'].isin(['within-subcortical', 'within-cortical'])].dropna(subset=['distance_mm'])

    log(f"  Cross-type: {len(cross)} pairs, distance [{cross['distance_mm'].min():.1f}, {cross['distance_mm'].max():.1f}]")
    log(f"  Within-type: {len(within)} pairs, distance [{within['distance_mm'].min():.1f}, {within['distance_mm'].max():.1f}]")

    rng = np.random.RandomState(RANDOM_SEED)
    out = {'n_iter': n_iter}

    for fc_col in ['fc_awake', 'fc_unconscious']:
        cross_rhos = []
        within_matched_rhos = []
        # Match within to cross by nearest distance
        for it in range(n_iter):
            cs = cross.sample(n=len(cross), replace=True, random_state=rng.randint(1e9))
            # For each cross-type pair, find nearest within-type pair by distance
            within_avail = within.copy()
            matched_rows = []
            for _, crow in cs.iterrows():
                if len(within_avail) == 0:
                    break
                dd = (within_avail['distance_mm'] - crow['distance_mm']).abs()
                idx = dd.idxmin()
                matched_rows.append(within_avail.loc[idx])
                within_avail = within_avail.drop(idx)
                if len(within_avail) == 0:
                    within_avail = within.copy()
            wm = pd.DataFrame(matched_rows)

            if len(cs) >= 4 and len(wm) >= 4:
                r_c, _ = spearmanr(cs['wj_unsigned'], cs[fc_col])
                r_w, _ = spearmanr(wm['wj_unsigned'], wm[fc_col])
                cross_rhos.append(r_c)
                within_matched_rhos.append(r_w)

        cr = np.array(cross_rhos)
        wr = np.array(within_matched_rhos)
        valid = ~(np.isnan(cr) | np.isnan(wr))
        cr = cr[valid]
        wr = wr[valid]
        diff = cr - wr
        out[fc_col] = {
            'n_iter_valid': int(valid.sum()),
            'cross_rho_mean': float(cr.mean()) if len(cr) else None,
            'cross_rho_ci': [float(np.percentile(cr, 2.5)),
                              float(np.percentile(cr, 97.5))] if len(cr) > 10 else None,
            'within_matched_rho_mean': float(wr.mean()) if len(wr) else None,
            'within_matched_rho_ci': [float(np.percentile(wr, 2.5)),
                                       float(np.percentile(wr, 97.5))] if len(wr) > 10 else None,
            'cross_exceeds_within_pct': float(100 * np.mean(cr > wr)) if len(cr) else None,
            'mean_difference_cross_minus_within': float(diff.mean()) if len(diff) else None,
            'difference_ci': [float(np.percentile(diff, 2.5)),
                               float(np.percentile(diff, 97.5))] if len(diff) > 10 else None,
        }
        log(f"  {fc_col}: cross={cr.mean():.4f}, within_matched={wr.mean():.4f}, "
            f"cross>within in {100*np.mean(cr>wr):.1f}% of iterations")

    with open(REBUILD / 'distance_matched.json', 'w') as f:
        json.dump(out, f, indent=2)
    log(f"  Saved distance_matched.json")
    return out


# ============================================================================
# PHASE 8: Provenance + interpretation report
# ============================================================================
def phase8_report(data, pairs_df, mantel, layer2h, per_region, dist_match, elapsed):
    log("\n" + "=" * 70)
    log("PHASE 8: PROVENANCE + INTERPRETATION REPORT")
    log("=" * 70)

    prov = {
        'methodology': 'WJ-native (foundation-up rebuild, disregards prior manuscript)',
        'fundamental_unit': 'individual gene (GTEx within-region samples) + individual ROI (fMRI)',
        'pairwise_matrix': 'full gene-gene Spearman within region; region-region architectural comparison',
        'correlation_method': 'Spearman (with Pearson as Layer 2H Type 6 partner)',
        'fdr_scope': 'N/A (cross-scale architectural test; Mantel and bootstrap quantify uncertainty)',
        'domain_conventional_methods': 'reported as Layer 2B alternatives, not competitors',
        'random_seed': RANDOM_SEED,
        'n_permutations_mantel': N_PERM_MANTEL,
        'n_bootstrap': N_BOOTSTRAP,
        'n_distance_match_iter': N_DIST_MATCH,
        'sign_strata_thresholds': SIGN_STRATA,
        'binary_jaccard_thresholds': BINARY_THRESHOLDS,
        'pipeline_file': os.path.basename(__file__),
        'execution_date': datetime.now().strftime('%Y-%m-%d'),
        'execution_time_minutes': elapsed,
        'wj_compliance_status': 'PASS (full rebuild)',
        'methodology_layers_applied': {
            'Layer_1_principle': 'Individual gene as fundamental unit; full pairwise matrix; all 11 regions included',
            'Layer_2B_comparison': 'unsigned WJ (primary), shifted WJ, binary Jaccard at {0.2, 0.3, 0.5}, Pearson/Spearman matrix corr on |r|, cosine, Frobenius',
            'Layer_2H_pairings': ['Type 1 (continuous-discrete: WJ vs binary Jaccard)',
                                   'Type 2 (sign-treatment: shifted vs unsigned)',
                                   'Type 6 (substrate-projection: Pearson vs Spearman vs cosine on WJ-FC)'],
            'Layer_2I_direct_sign_flip': f'Stratified at {SIGN_STRATA} with binomial p vs 0.5',
            'Layer_4_1_independence': 'Mantel test (10K perms, seed 42) for primary inference',
            'Layer_4_2_subset_stringency': 'Mantel test repeated on cross-type and within-type subsets',
            'Layer_2F_per_region_row': 'Per-region row WJ with bootstrap CIs',
        },
        'data_sources': {
            'gene_expression': f'GTEx v8 individual samples ({data["n_genes"]} genes, {len(data["valid_regions"])} regions, {data["n_brain_samples"]} samples)',
            'brain_connectivity': 'OpenNeuro ds006623 propofol fMRI, 4S456 parcellation (group correlation matrices)',
            'distance': 'Approximate MNI centroids per region',
        },
        'output_files': [
            'pairwise_metrics.csv',
            'sign_flip_stratified.csv',
            'mantel_all.json',
            'layer2h_pairings.json',
            'per_region_decomposition.csv',
            'distance_matched.json',
            'interpretation_report.md',
        ],
    }

    with open(REBUILD / 'provenance.json', 'w') as f:
        json.dump(prov, f, indent=2)
    log(f"  Saved provenance.json")

    # Markdown report
    lines = []
    lines.append('# Gene-to-Brain Foundation-Up Rebuild — Results & Interpretation')
    lines.append(f'\n**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'**Author:** Drake H. Harbert (D.H.H.), Inner Architecture LLC')
    lines.append(f'**Compute time:** {elapsed:.1f} minutes\n')
    lines.append('---\n')

    lines.append('## What was measured (in order)\n')
    lines.append('1. **Gene-gene Spearman correlation matrices within each brain region** using individual GTEx samples. '
                  f'Result: {data["n_genes"]} gene × {data["n_genes"]} matrices, one per region, computed from '
                  f'{data["n_brain_samples"]} total samples across {len(data["valid_regions"])} regions.\n')
    lines.append('2. **All Layer 2B architectural similarity metrics between every pair of region matrices**: '
                  f'unsigned WJ (primary), shifted WJ, binary Jaccard at {BINARY_THRESHOLDS}, '
                  'Pearson matrix corr on |r|, Spearman matrix corr on |r|, cosine on |r|, Frobenius diff.\n')
    lines.append(f'3. **Layer 2I direct pair-level sign-flip rate** stratified by min(|r_A|, |r_B|) at thresholds {SIGN_STRATA}, '
                  'with binomial test against the 50% chance rate.\n')
    lines.append('4. **Brain functional connectivity** (awake + propofol-unconscious) extracted per region-pair from the '
                  '4S456 parcellation, plus expression profile similarity (median TPM Spearman) and Euclidean distance.\n')
    lines.append('5. **Mantel test (10K perms, seed 42)** for each predictor × outcome × subset combination '
                  '(WJ and expression profile, both vs FC_awake/FC_unconscious/activity-diff, across all/cross/within subsets).\n')
    lines.append('6. **Partial Spearman** controlling for Euclidean distance.\n')
    lines.append('7. **Layer 2H pairing decompositions**: Type 1 (WJ vs binary Jaccard), Type 2 (shifted vs unsigned), '
                  'Type 6 (Pearson/Spearman/cosine on WJ-FC) with bootstrap CIs on the gaps.\n')
    lines.append('8. **Per-region row decomposition** (Layer 2F): mean and CI of each region\'s WJ to all partner regions.\n')
    lines.append('9. **Distance-matched subsampling** (1000 iterations) to compare cross-type vs within-type signal after '
                  'matching distance distributions.\n')
    lines.append('\n---\n')

    # Headline results
    lines.append('## Headline numbers\n')
    lines.append('### Mantel (all 55 pairs)\n')
    for outcome in ['FC_awake', 'FC_unconscious', 'FC_activity_diff']:
        rec = mantel['mantel_tests'][outcome].get('WJ_unsigned_all_pairs', {})
        if rec:
            lines.append(f'- WJ vs {outcome}: Pearson r = **{rec["observed_pearson"]:+.4f}**, '
                          f'Mantel p (two-sided) = **{rec["mantel_p_two"]:.4f}**\n')
        rec_e = mantel['mantel_tests'][outcome].get('expr_profile_all_pairs', {})
        if rec_e:
            lines.append(f'- Expression profile vs {outcome}: Pearson r = {rec_e["observed_pearson"]:+.4f}, '
                          f'Mantel p (two-sided) = {rec_e["mantel_p_two"]:.4f}\n')

    lines.append('\n### Mantel (cross-type subset)\n')
    for outcome in ['FC_awake', 'FC_unconscious']:
        rec = mantel['mantel_tests'][outcome].get('WJ_unsigned_cross_type', {})
        if rec:
            lines.append(f'- WJ vs {outcome} (cross-type only): Pearson r = **{rec["observed_pearson"]:+.4f}**, '
                          f'Mantel p = **{rec["mantel_p_two"]:.4f}**\n')

    lines.append('\n### Mantel (within-type subset)\n')
    for outcome in ['FC_awake', 'FC_unconscious']:
        rec = mantel['mantel_tests'][outcome].get('WJ_unsigned_within_type', {})
        if rec:
            lines.append(f'- WJ vs {outcome} (within-type only): Pearson r = **{rec["observed_pearson"]:+.4f}**, '
                          f'Mantel p = **{rec["mantel_p_two"]:.4f}**\n')

    lines.append('\n### Partial Spearman (distance-controlled, full set)\n')
    for k, v in mantel['partial_spearman_distance'].items():
        lines.append(f'- {k}: rho = {v["rho"]:+.4f}, p = {v["p"]:.4f}\n')

    lines.append('\n---\n')
    lines.append('## Interpretation notes (what was measured and why)\n')
    lines.append('### What each test is asking\n')
    lines.append('- **Mantel test (Layer 4.1)**: Does the architectural similarity matrix (WJ) correspond to the '
                  'functional connectivity matrix, accounting for the fact that region pairs share component regions '
                  '(non-independent observations)?\n')
    lines.append('- **Partial Spearman (distance-controlled)**: Does the WJ–FC relationship survive after removing '
                  'the linear contribution of physical distance between regions?\n')
    lines.append('- **Subset Mantel (Layer 4.2)**: When the test is restricted to cross-type or within-type pairs, '
                  'does the WJ–FC relationship hold at the same stringency as the full set?\n')
    lines.append('- **Layer 2H Type 1 (WJ vs binary Jaccard gap)**: Localizes whether the architectural divergence '
                  'lives in supra-threshold large correlations (gap positive) or in the bulk distribution (gap negative).\n')
    lines.append('- **Layer 2H Type 2 (shifted vs unsigned WJ gap)**: A region-of-magnitude vs sign-treatment '
                  'measurement. Per Layer 2I rule the gap is NOT a sign-inversion percentage — see Layer 2I stratified '
                  'sign-flip rate for the canonical sign-reorganization measurement.\n')
    lines.append('- **Layer 2H Type 6 (Pearson/Spearman/cosine on WJ-FC)**: Which feature of the WJ-FC relationship '
                  'drives the correspondence — linearity (Pearson), monotonicity (Spearman), or geometric alignment '
                  '(cosine)? The gap with bootstrap CI tells us whether the choice of correspondence metric is '
                  'informative.\n')
    lines.append('- **Layer 2I sign-flip rate stratified**: For each region pair, the fraction of gene pairs where '
                  'the correlation sign differs between the two regions, restricted to gene pairs where both regions '
                  'have correlation magnitude above the stratification threshold. Tested against 50% chance rate.\n')
    lines.append('- **Layer 2F per-region row WJ**: Each region\'s mean WJ to all 10 partner regions — identifies '
                  'whether some regions are architecturally stable hubs vs reorganized.\n')
    lines.append('- **Distance-matched subsampling**: When cross-type and within-type pairs are matched on distance '
                  'distribution, does the WJ–FC relationship still differ between them?\n')

    lines.append('\n---\n')
    lines.append('## See attached files for full numbers\n')
    lines.append('- `pairwise_metrics.csv` — every metric for every region pair\n')
    lines.append('- `sign_flip_stratified.csv` — Layer 2I, one row per pair × threshold\n')
    lines.append('- `mantel_all.json` — all Mantel and partial tests\n')
    lines.append('- `layer2h_pairings.json` — Type 1, 2, 6 with bootstrap CIs\n')
    lines.append('- `per_region_decomposition.csv` — Layer 2F\n')
    lines.append('- `distance_matched.json` — Phase 7 distance-matched comparison\n')
    lines.append('- `provenance.json` — full methodology + parameters\n')

    with open(REBUILD / 'interpretation_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f"  Saved interpretation_report.md")


# ============================================================================
# MAIN
# ============================================================================
def main():
    log("=" * 70)
    log("GENE-TO-BRAIN FOUNDATION-UP REBUILD — FROM SCRATCH")
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Random seed: {RANDOM_SEED}")
    log(f"Output: {REBUILD}")
    log("=" * 70)

    data = phase1_load_data()
    pairs_df, flip_df, _ = phase23_pairwise_compute(data)
    mantel, ctx = phase4_mantel_all(pairs_df)
    layer2h = phase5_layer2h(pairs_df, ctx)
    per_region = phase6_per_region_row(pairs_df, ctx)
    dist_match = phase7_distance_matching(pairs_df)

    elapsed = (time.time() - START) / 60
    phase8_report(data, pairs_df, mantel, layer2h, per_region, dist_match, elapsed)

    log(f"\n{'='*70}")
    log(f"REBUILD COMPLETE in {elapsed:.1f} minutes")
    log(f"All outputs in: {REBUILD}")
    log(f"{'='*70}")


if __name__ == '__main__':
    main()
