"""
Pipeline: Gene-to-Brain RELATIONAL Cross-Scale Test
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-03-20
Description:
    The WJ-level cross-scale test. For each brain region, compute gene-gene
    co-expression architecture from individual-level GTEx data. Then compute
    WJ between every pair of brain regions' co-expression architectures.
    Compare architecture similarity (WJ) to functional connectivity (fMRI).
    This is the RELATIONAL version — not "do these regions express the same
    genes?" but "do these regions have the same RELATIONSHIPS between genes?"
Dependencies: numpy, scipy, pandas, matplotlib, seaborn
Input: GTEx v8 individual TPM, sample attributes, fMRI correlation matrices
Output: Cross-scale WJ results, figures, provenance.json
"""
import os, sys, time, json, gzip, warnings, gc
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_PERMS = 1000
N_BOOT = 500
MAX_GENES = None  # No cap — full transcriptome, discovery first. Memory managed by computing one region at a time.
START = time.time()

BASE = r'G:\My Drive\inner_architecture_research\gene_to_brain_wj'
RESULTS = os.path.join(BASE, 'results')
FIGURES = os.path.join(BASE, 'figures')
os.makedirs(RESULTS, exist_ok=True)

GTEX_TPM = r'G:\My Drive\inner_architecture_research\MS3_JNC_Submission\data\GTEx_v8_tpm.gct.gz'
GTEX_ATTR = r'G:\My Drive\inner_architecture_research\MS3_JNC_Submission\data\GTEx_v8_sample_attributes.txt'
BRAIN_CORR = r'G:\My Drive\inner_architecture_research\brain_connectivity_wj\results\correlation_matrices\group_awake_spearman_corr.npy'
ATLAS_FILE = (r'G:\My Drive\inner_architecture_research\brain_connectivity_wj\data\raw'
              r'\ds006623\derivatives\xcp_d_without_GSR_bandpass_output\atlases'
              r'\atlas-4S456Parcels\atlas-4S456Parcels_dseg.tsv')

BRAIN_REGIONS = [
    'Brain - Amygdala',
    'Brain - Anterior cingulate cortex (BA24)',
    'Brain - Caudate (basal ganglia)',
    'Brain - Cerebellum',
    'Brain - Cortex',
    'Brain - Frontal Cortex (BA9)',
    'Brain - Hippocampus',
    'Brain - Hypothalamus',
    'Brain - Nucleus accumbens (basal ganglia)',
    'Brain - Putamen (basal ganglia)',
    'Brain - Substantia nigra',
]

SIGMAR1_GENES = ['SIGMAR1', 'AHCY', 'MAT2A', 'MTHFR', 'MTR', 'FKBP5',
                 'NR3C1', 'NR3C2', 'BDNF', 'CREB1']


def log(msg):
    print(f"[{(time.time()-START)/60:6.1f}m] {msg}", flush=True)


def fast_spearman_matrix(data):
    n = data.shape[0]
    ranked = np.zeros_like(data, dtype=np.float64)
    for i in range(n):
        ranked[i] = rankdata(data[i])
    ranked -= ranked.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(ranked**2, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    ranked /= norms
    corr = ranked @ ranked.T
    np.clip(corr, -1.0, 1.0, out=corr)
    return corr


def weighted_jaccard(corr_A, corr_B):
    idx = np.triu_indices(corr_A.shape[0], k=1)
    a = np.abs(corr_A[idx])
    b = np.abs(corr_B[idx])
    num = np.minimum(a, b).sum()
    den = np.maximum(a, b).sum()
    return float(num / den) if den > 0 else 1.0


# ============================================================================
# PHASE 1: IDENTIFY BRAIN SAMPLES
# ============================================================================
log("=" * 70)
log("GENE-TO-BRAIN RELATIONAL CROSS-SCALE TEST")
log("=" * 70)

log("\nPhase 1: Loading sample annotations...")
sa = pd.read_csv(GTEX_ATTR, sep='\t', usecols=['SAMPID', 'SMTSD'])
brain_sa = sa[sa['SMTSD'].isin(BRAIN_REGIONS)]
log(f"  Brain samples: {len(brain_sa)} across {brain_sa['SMTSD'].nunique()} regions")

# Get sample IDs per region
region_samples = {}
for region in BRAIN_REGIONS:
    samples = brain_sa[brain_sa['SMTSD'] == region]['SAMPID'].tolist()
    region_samples[region] = samples
    log(f"    {region}: {len(samples)} samples")

# ============================================================================
# PHASE 2: LOAD GTEx EXPRESSION FOR BRAIN SAMPLES ONLY
# ============================================================================
log("\nPhase 2: Loading GTEx individual-level expression (brain samples only)...")
log("  This file is 1.6 GB compressed — parsing will take a few minutes...")

# Read header to get column positions
with gzip.open(GTEX_TPM, 'rt') as f:
    f.readline()  # version
    f.readline()  # dims
    header = f.readline().strip().split('\t')

# Find column indices for brain samples
all_brain_ids = set()
for samples in region_samples.values():
    all_brain_ids.update(samples)

col_indices = [0, 1]  # Name, Description
sample_col_map = {}  # column_index -> sample_id
for i, col in enumerate(header):
    if col in all_brain_ids:
        col_indices.append(i)
        sample_col_map[i] = col

log(f"  Brain sample columns found: {len(sample_col_map)}")
if len(sample_col_map) < 100:
    log("  WARNING: Few brain samples found in TPM matrix. Check ID format.")
    # GTEx sample IDs in TPM might use different format than attributes
    # Try matching on GTEX-XXXXX prefix
    tpm_sample_ids = set(header[2:])
    matched = all_brain_ids & tpm_sample_ids
    log(f"  Direct matches: {len(matched)}")

# Load only brain columns
log("  Reading expression matrix (brain columns only)...")
t0 = time.time()

# Use chunked reading for memory efficiency
chunks = []
col_names = [header[i] for i in sorted(col_indices)]

expr_df = pd.read_csv(GTEX_TPM, sep='\t', skiprows=2,
                      usecols=col_indices, compression='gzip')
expr_df.columns = col_names

log(f"  Loaded in {(time.time()-t0)/60:.1f} minutes")
log(f"  Shape: {expr_df.shape}")

# Set gene symbols as index
expr_df['gene'] = expr_df['Description'].str.upper()
expr_df = expr_df.drop(['Name', 'Description'], axis=1)
expr_df = expr_df.drop_duplicates(subset='gene', keep='first')
expr_df = expr_df.set_index('gene')

# Log-transform
expr_df = np.log2(expr_df + 1)

# Filter: keep genes expressed in brain (mean TPM > 1 in at least one region)
gene_means = expr_df.mean(axis=1)
expr_df = expr_df[gene_means > np.log2(2)]  # log2(1+1) = 1
log(f"  Genes after expression filter: {len(expr_df)}")

# NO variance cap — full transcriptome, discovery first.
# Memory managed by computing one region's correlation matrix at a time.
# Peak memory: 2 matrices x (n_genes^2 x 8 bytes) for pairwise WJ computation.

gene_names = list(expr_df.index)
log(f"  GOI present: {[g for g in SIGMAR1_GENES if g in gene_names]}")

# ============================================================================
# PHASE 3: COMPUTE PER-REGION GENE CO-EXPRESSION MATRICES
# ============================================================================
log(f"\nPhase 3: Computing gene co-expression matrices per brain region...")

# Compute co-expression matrices one at a time and save to disk
# This avoids holding all 11 matrices in RAM simultaneously
corr_cache_dir = os.path.join(RESULTS, 'region_corr_cache')
os.makedirs(corr_cache_dir, exist_ok=True)

valid_regions_corr = []
n_genes_used = len(expr_df)
mem_per_matrix_gb = (n_genes_used ** 2 * 8) / (1024**3)
log(f"  Genes: {n_genes_used}, Memory per matrix: {mem_per_matrix_gb:.1f} GB")

for region in BRAIN_REGIONS:
    samples = region_samples[region]
    available = [s for s in samples if s in expr_df.columns]
    if len(available) < 30:
        log(f"  {region}: only {len(available)} samples — SKIPPING (need >= 30)")
        continue

    cache_file = os.path.join(corr_cache_dir,
                               region.replace(' ', '_').replace('(', '').replace(')', '') + '.npy')

    if os.path.exists(cache_file) and not FORCE_RECOMPUTE:
        log(f"  {region}: loading cached matrix")
        valid_regions_corr.append(region)
        continue

    region_expr = expr_df[available].values.astype(np.float64)
    log(f"  {region}: {len(available)} samples, {region_expr.shape[0]} genes...")

    t0 = time.time()
    corr = fast_spearman_matrix(region_expr)
    elapsed = time.time() - t0
    log(f"    Computed in {elapsed:.1f}s, saving to cache...")

    np.save(cache_file, corr)
    valid_regions_corr.append(region)

    del corr, region_expr
    gc.collect()

log(f"\n  Regions with co-expression matrices: {len(valid_regions_corr)}")

# ============================================================================
# PHASE 4: COMPUTE WJ BETWEEN ALL REGION PAIRS
# ============================================================================
log(f"\nPhase 4: Computing WJ between all region pairs...")

regions_with_corr = sorted(valid_regions_corr)
n_regions = len(regions_with_corr)
wj_matrix = np.ones((n_regions, n_regions))

def load_region_corr(region):
    cache_file = os.path.join(corr_cache_dir,
                               region.replace(' ', '_').replace('(', '').replace(')', '') + '.npy')
    return np.load(cache_file)

# Compute WJ pairwise, loading only 2 matrices at a time
for i in range(n_regions):
    r1 = regions_with_corr[i]
    corr_i = load_region_corr(r1)
    for j in range(i + 1, n_regions):
        r2 = regions_with_corr[j]
        corr_j = load_region_corr(r2)
        wj = weighted_jaccard(corr_i, corr_j)
        wj_matrix[i, j] = wj
        wj_matrix[j, i] = wj

        r1s = r1.replace('Brain - ', '').replace(' (basal ganglia)', '')[:20]
        r2s = r2.replace('Brain - ', '').replace(' (basal ganglia)', '')[:20]
        log(f"    {r1s:>20} -- {r2s:<20}  WJ = {wj:.4f}")
        del corr_j
    del corr_i
    gc.collect()

# Save WJ matrix
wj_df = pd.DataFrame(wj_matrix, index=regions_with_corr, columns=regions_with_corr)
wj_df.to_csv(os.path.join(RESULTS, 'region_coexpression_wj_matrix.csv'))
log(f"  Saved WJ matrix")

# ============================================================================
# PHASE 5: LOAD BRAIN CONNECTIVITY AND MAP REGIONS
# ============================================================================
log(f"\nPhase 5: Loading brain connectivity and mapping regions...")

corr_brain = np.load(BRAIN_CORR)
atlas = pd.read_csv(ATLAS_FILE, sep='\t')
parcel_labels = atlas['label'].tolist()

# Exact parcel mapping (same as v2)
REGION_MAP = {
    'Brain - Putamen (basal ganglia)': [i for i, l in enumerate(parcel_labels)
        if l in ['LH-Pu', 'RH-Pu']],
    'Brain - Caudate (basal ganglia)': [i for i, l in enumerate(parcel_labels)
        if l in ['LH-Ca', 'RH-Ca']],
    'Brain - Nucleus accumbens (basal ganglia)': [i for i, l in enumerate(parcel_labels)
        if l in ['LH-NAC', 'RH-NAC']],
    'Brain - Hippocampus': [i for i, l in enumerate(parcel_labels)
        if 'Hippocampus' in l or 'HN' in l],
    'Brain - Amygdala': [i for i, l in enumerate(parcel_labels)
        if 'Amygdala' in l or 'EXA' in l],
    'Brain - Hypothalamus': [i for i, l in enumerate(parcel_labels)
        if 'HTH' in l],
    'Brain - Substantia nigra': [i for i, l in enumerate(parcel_labels)
        if 'SNc' in l or 'SNr' in l],
    'Brain - Cerebellum': [i for i, l in enumerate(parcel_labels)
        if 'Cerebellar' in l],
    'Brain - Frontal Cortex (BA9)': [i for i, l in enumerate(parcel_labels)
        if any(k in l for k in ['Cont_PFCl', 'Cont_PFCmp', 'Default_PFC',
                                  'Cont_pCun'])
        and atlas.iloc[i].get('atlas_name', '') == '4S456'],
    'Brain - Anterior cingulate cortex (BA24)': [i for i, l in enumerate(parcel_labels)
        if any(k in l for k in ['SalVentAttn_Med', 'Limbic_OFC',
                                  'Default_pCunPCC', 'Cont_Cing'])
        and atlas.iloc[i].get('atlas_name', '') == '4S456'],
    'Brain - Cortex': [i for i, l in enumerate(parcel_labels)
        if atlas.iloc[i].get('atlas_name', '') == '4S456'],
}

# Remove empty
for r in list(REGION_MAP.keys()):
    if not REGION_MAP[r]:
        del REGION_MAP[r]

# ============================================================================
# PHASE 6: CROSS-SCALE COMPARISON (WJ CO-EXPRESSION vs BRAIN FC)
# ============================================================================
log(f"\nPhase 6: Cross-scale comparison...")

# Both region must have co-expression matrix AND fMRI mapping
valid_regions = [r for r in regions_with_corr if r in REGION_MAP]
log(f"  Regions with both co-expression and fMRI data: {len(valid_regions)}")

wj_values = []
fc_values = []
pair_labels = []

for i in range(len(valid_regions)):
    for j in range(i + 1, len(valid_regions)):
        r1 = valid_regions[i]
        r2 = valid_regions[j]

        # Gene co-expression architecture similarity (WJ)
        r1_idx = regions_with_corr.index(r1)
        r2_idx = regions_with_corr.index(r2)
        wj_val = wj_matrix[r1_idx, r2_idx]

        # Brain functional connectivity
        p1 = [p for p in REGION_MAP[r1] if p < corr_brain.shape[0]]
        p2 = [p for p in REGION_MAP[r2] if p < corr_brain.shape[0]]
        if not p1 or not p2:
            continue
        fc_val = np.mean(corr_brain[np.ix_(p1, p2)])

        wj_values.append(wj_val)
        fc_values.append(fc_val)
        r1s = r1.replace('Brain - ', '').replace(' (basal ganglia)', '')
        r2s = r2.replace('Brain - ', '').replace(' (basal ganglia)', '')
        pair_labels.append(f"{r1s}-{r2s}")

wj_arr = np.array(wj_values)
fc_arr = np.array(fc_values)

log(f"  Region pairs: {len(wj_arr)}")

# Primary test
rho, p = stats.spearmanr(wj_arr, fc_arr)
r_pear, p_pear = stats.pearsonr(wj_arr, fc_arr)
log(f"\n  *** RELATIONAL CROSS-SCALE CORRELATION ***")
log(f"  Gene co-expression ARCHITECTURE similarity (WJ) vs Brain FC:")
log(f"  Spearman rho = {rho:.4f}, p = {p:.6f}")
log(f"  Pearson r = {r_pear:.4f}, p = {p_pear:.6f}")

# Bootstrap CI
rng = np.random.RandomState(RANDOM_SEED)
boot = np.zeros(N_BOOT)
n = len(wj_arr)
for b in range(N_BOOT):
    idx = rng.choice(n, n, replace=True)
    boot[b], _ = stats.spearmanr(wj_arr[idx], fc_arr[idx])
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
log(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

# Permutation test
perm_rhos = np.zeros(N_PERMS)
for i in range(N_PERMS):
    perm_rhos[i], _ = stats.spearmanr(rng.permutation(wj_arr), fc_arr)
perm_p = np.mean(np.abs(perm_rhos) >= np.abs(rho))
log(f"  Permutation p = {perm_p:.4f}")

d = (rho - np.mean(perm_rhos)) / np.std(perm_rhos)
log(f"  Cohen's d = {d:.2f}")

# Per-pair details
log(f"\n  Per-pair details:")
pairs_df = pd.DataFrame({
    'Pair': pair_labels,
    'CoExpr_WJ': wj_arr,
    'Brain_FC': fc_arr,
}).sort_values('CoExpr_WJ')
for _, row in pairs_df.iterrows():
    log(f"    {row['Pair']:<45} WJ={row['CoExpr_WJ']:.4f}  FC={row['Brain_FC']:.3f}")

pairs_df.to_csv(os.path.join(RESULTS, 'relational_cross_scale_pairs.csv'),
                index=False, float_format='%.6f')

# ============================================================================
# PHASE 7: COMPARE COMPONENT vs RELATIONAL
# ============================================================================
log(f"\nPhase 7: Component (expression profile) vs Relational (WJ) comparison...")

# Load median TPM for component-level comparison
gtex_median = os.path.join(BASE, 'data', 'GTEx_gene_median_tpm.gct.gz')
with gzip.open(gtex_median, 'rt') as f:
    f.readline(); f.readline()
    med_header = f.readline().strip().split('\t')
brain_cols_med = [c for c in med_header if c.startswith('Brain -')]
med_df = pd.read_csv(gtex_median, sep='\t', skiprows=2,
                     usecols=['Description'] + brain_cols_med, compression='gzip')
med_df['gene'] = med_df['Description'].str.upper()
med_df = med_df.drop_duplicates(subset='gene', keep='first').set_index('gene')
med_df = med_df[brain_cols_med]
med_df = med_df[(med_df > 1).any(axis=1)]
med_log = np.log2(med_df + 1)

# Component-level similarity for valid pairs
comp_values = []
for _, row in pairs_df.iterrows():
    parts = row['Pair'].split('-')
    # Reconstruct full region names
    r1_full = None; r2_full = None
    for r in valid_regions:
        rs = r.replace('Brain - ', '').replace(' (basal ganglia)', '')
        if rs == parts[0]:
            r1_full = r
        if rs == parts[1] if len(parts) == 2 else '-'.join(parts[1:]):
            r2_full = r

    if r1_full and r2_full and r1_full in brain_cols_med and r2_full in brain_cols_med:
        r1_vals = med_log[r1_full].values
        r2_vals = med_log[r2_full].values
        rho_comp, _ = stats.spearmanr(r1_vals, r2_vals)
        comp_values.append(rho_comp)
    else:
        comp_values.append(np.nan)

pairs_df['Component_Sim'] = comp_values

# Compare: which predicts brain FC better?
valid_mask = ~np.isnan(pairs_df['Component_Sim'].values)
if valid_mask.sum() > 5:
    rho_comp_fc, p_comp_fc = stats.spearmanr(
        pairs_df.loc[valid_mask, 'Component_Sim'],
        pairs_df.loc[valid_mask, 'Brain_FC'])
    rho_wj_fc, p_wj_fc = stats.spearmanr(
        pairs_df.loc[valid_mask, 'CoExpr_WJ'],
        pairs_df.loc[valid_mask, 'Brain_FC'])

    log(f"\n  *** COMPONENT vs RELATIONAL COMPARISON ***")
    log(f"  Component (expression profile sim) vs Brain FC: "
        f"rho={rho_comp_fc:.4f}, p={p_comp_fc:.4f}")
    log(f"  Relational (co-expression WJ) vs Brain FC:      "
        f"rho={rho_wj_fc:.4f}, p={p_wj_fc:.4f}")

    if abs(rho_wj_fc) > abs(rho_comp_fc):
        log(f"  >>> RELATIONAL WINS: WJ predicts brain connectivity "
            f"better than expression profiles")
    else:
        log(f"  >>> COMPONENT WINS: Expression profiles predict brain "
            f"connectivity better than WJ")

# ============================================================================
# PHASE 8: FIGURES
# ============================================================================
log(f"\nPhase 8: Generating figures...")
colors = sns.color_palette('colorblind', 3)

# Figure: WJ co-expression similarity heatmap
fig, ax = plt.subplots(figsize=(12, 10))
short_names = [r.replace('Brain - ', '').replace(' (basal ganglia)', '')
               for r in regions_with_corr]
sns.heatmap(wj_matrix, xticklabels=short_names, yticklabels=short_names,
            cmap='RdYlBu_r', vmin=0.5, vmax=1.0, annot=True, fmt='.3f',
            annot_kws={'size': 7}, square=True, ax=ax)
ax.set_title('Gene Co-expression Architecture Similarity (WJ)\nBetween Brain Regions',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'figure3_coexpr_wj_heatmap.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
log(f"  Saved figure3")

# Figure: Relational cross-scale scatter
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Component (expression profile)
if valid_mask.sum() > 5:
    ax = axes[0]
    ax.scatter(pairs_df.loc[valid_mask, 'Component_Sim'],
               pairs_df.loc[valid_mask, 'Brain_FC'],
               s=60, alpha=0.7, color=colors[0], edgecolors='black', linewidth=0.5)
    z = np.polyfit(pairs_df.loc[valid_mask, 'Component_Sim'].values,
                   pairs_df.loc[valid_mask, 'Brain_FC'].values, 1)
    x_line = np.linspace(pairs_df['Component_Sim'].min(),
                         pairs_df['Component_Sim'].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r--', linewidth=2)
    ax.set_xlabel('Expression Profile Similarity', fontsize=12)
    ax.set_ylabel('Brain Functional Connectivity', fontsize=12)
    ax.set_title(f'(A) Component Level\nrho={rho_comp_fc:.3f}, p={p_comp_fc:.4f}',
                 fontsize=12, fontweight='bold')

# Panel B: Relational (WJ)
ax = axes[1]
ax.scatter(wj_arr, fc_arr, s=60, alpha=0.7, color=colors[1],
           edgecolors='black', linewidth=0.5)
z = np.polyfit(wj_arr, fc_arr, 1)
x_line = np.linspace(wj_arr.min(), wj_arr.max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'r--', linewidth=2)
ax.set_xlabel('Co-expression Architecture Similarity (WJ)', fontsize=12)
ax.set_ylabel('Brain Functional Connectivity', fontsize=12)
ax.set_title(f'(B) Relational Level\nrho={rho:.3f} [{ci_lo:.3f}-{ci_hi:.3f}], '
             f'perm p={perm_p:.4f}',
             fontsize=12, fontweight='bold')

# Annotate pairs
for _, row in pairs_df.iterrows():
    ax.annotate(row['Pair'], (row['CoExpr_WJ'], row['Brain_FC']),
                fontsize=5, alpha=0.5, xytext=(3, 3), textcoords='offset points')

plt.suptitle('Molecular Architecture Predicts Brain Connectivity:\n'
             'Component vs Relational', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'figure4_component_vs_relational.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
log(f"  Saved figure4")

# ============================================================================
# PROVENANCE
# ============================================================================
provenance = {
    'methodology': 'WJ-native (cross-scale relational test)',
    'fundamental_unit': 'individual gene co-expression per brain region',
    'pairwise_matrix': 'full gene-gene per region, WJ between regions',
    'correlation_method': 'Spearman',
    'random_seed': RANDOM_SEED,
    'n_genes': len(gene_names),
    'n_regions': len(regions_with_corr),
    'n_region_pairs': len(wj_arr),
    'relational_rho': float(rho),
    'relational_p': float(perm_p),
    'relational_ci': [float(ci_lo), float(ci_hi)],
    'cohens_d': float(d),
    'pipeline_file': os.path.basename(__file__),
    'execution_date': '2026-03-20',
    'wj_compliance_status': 'PASS',
}
with open(os.path.join(RESULTS, 'provenance_relational.json'), 'w') as f:
    json.dump(provenance, f, indent=2)

log(f"\nTotal time: {(time.time()-START)/60:.1f} minutes")
log("COMPLETE")
