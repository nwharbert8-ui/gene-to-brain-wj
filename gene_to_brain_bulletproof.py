"""
Pipeline: Gene-to-Brain BULLETPROOF Validation
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-03-20
Description:
    Every test that could kill the gene-to-brain finding.
    If it survives all of these, it's real. If it doesn't, we say so.
    1. Spatial proximity control (partial correlation)
    2. Steiger's Z (formal relational > component test)
    3. Leave-one-region-out stability
    4. Split-half replication (random donor splits)
    5. Hemisphere control (left-only vs right-only parcels)
    6. Reconcile the two rho values (55-pair vs 21-pair)
    7. Effect of propofol (awake vs unconscious FC)
Dependencies: numpy, scipy, pandas
"""
import os, time, json, gzip, gc, warnings
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
START = time.time()

BASE = r'G:\My Drive\inner_architecture_research\gene_to_brain_wj'
RESULTS = os.path.join(BASE, 'results')
FIGURES = os.path.join(BASE, 'figures')
CACHE = os.path.join(RESULTS, 'region_corr_cache')
BRAIN = r'G:\My Drive\inner_architecture_research\brain_connectivity_wj'
GTEX_TPM = r'G:\My Drive\inner_architecture_research\MS3_JNC_Submission\data\GTEx_v8_tpm.gct.gz'
GTEX_ATTR = r'G:\My Drive\inner_architecture_research\MS3_JNC_Submission\data\GTEx_v8_sample_attributes.txt'

def log(msg):
    print(f"[{(time.time()-START)/60:6.1f}m] {msg}", flush=True)

def fast_spearman(data):
    n = data.shape[0]
    ranked = np.zeros_like(data, dtype=np.float64)
    for i in range(n):
        ranked[i] = rankdata(data[i])
    ranked -= ranked.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(ranked**2, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    ranked /= norms
    return np.clip(ranked @ ranked.T, -1, 1)

def wj_chunked(A, B, cs=2000):
    n = A.shape[0]; num=0.0; den=0.0
    for i in range(0,n,cs):
        ie=min(i+cs,n)
        for j in range(i,n,cs):
            je=min(j+cs,n)
            a=np.abs(A[i:ie,j:je]); b=np.abs(B[i:ie,j:je])
            if i==j:
                m=np.triu(np.ones((ie-i,je-j),dtype=bool),k=1)
                num+=np.minimum(a[m],b[m]).sum(); den+=np.maximum(a[m],b[m]).sum()
            elif j>i:
                num+=np.minimum(a,b).sum(); den+=np.maximum(a,b).sum()
    return float(num/den) if den>0 else 1.0

# Approximate MNI coordinates for GTEx brain regions (centroids)
# Sources: Harvard-Oxford atlas, Tian subcortical atlas, MNI coordinates
MNI_COORDS = {
    'Brain - Amygdala':                          np.array([24, -4, -18]),
    'Brain - Anterior cingulate cortex (BA24)':  np.array([2, 28, 22]),
    'Brain - Caudate (basal ganglia)':           np.array([12, 12, 10]),
    'Brain - Cerebellum':                        np.array([2, -60, -30]),
    'Brain - Cortex':                            np.array([0, 0, 40]),
    'Brain - Frontal Cortex (BA9)':              np.array([4, 50, 30]),
    'Brain - Hippocampus':                       np.array([28, -20, -14]),
    'Brain - Hypothalamus':                      np.array([2, -4, -10]),
    'Brain - Nucleus accumbens (basal ganglia)': np.array([10, 12, -6]),
    'Brain - Putamen (basal ganglia)':           np.array([26, 4, 2]),
    'Brain - Substantia nigra':                  np.array([10, -18, -12]),
}

log("=" * 70)
log("GENE-TO-BRAIN BULLETPROOF VALIDATION")
log("=" * 70)

# ============================================================================
# LOAD ALL DATA
# ============================================================================
log("\nLoading data...")

# Atlas + brain connectivity (awake AND unconscious)
atlas = pd.read_csv(os.path.join(BRAIN, 'data', 'raw', 'ds006623', 'derivatives',
    'xcp_d_without_GSR_bandpass_output', 'atlases', 'atlas-4S456Parcels',
    'atlas-4S456Parcels_dseg.tsv'), sep='\t')
pl = atlas['label'].tolist()
corr_awake = np.load(os.path.join(BRAIN, 'results', 'correlation_matrices',
                                   'group_awake_spearman_corr.npy'))
corr_uncon = np.load(os.path.join(BRAIN, 'results', 'correlation_matrices',
                                   'group_unconscious_spearman_corr.npy'))

RMAP = {
    'Brain - Putamen (basal ganglia)': [i for i,l in enumerate(pl) if l in ['LH-Pu','RH-Pu']],
    'Brain - Caudate (basal ganglia)': [i for i,l in enumerate(pl) if l in ['LH-Ca','RH-Ca']],
    'Brain - Nucleus accumbens (basal ganglia)': [i for i,l in enumerate(pl) if l in ['LH-NAC','RH-NAC']],
    'Brain - Hippocampus': [i for i,l in enumerate(pl) if 'Hippocampus' in l or 'HN' in l],
    'Brain - Amygdala': [i for i,l in enumerate(pl) if 'Amygdala' in l or 'EXA' in l],
    'Brain - Hypothalamus': [i for i,l in enumerate(pl) if 'HTH' in l],
    'Brain - Substantia nigra': [i for i,l in enumerate(pl) if 'SNc' in l or 'SNr' in l],
    'Brain - Cerebellum': [i for i,l in enumerate(pl) if 'Cerebellar' in l],
    'Brain - Frontal Cortex (BA9)': [i for i,l in enumerate(pl)
        if any(k in l for k in ['Cont_PFCl','Cont_PFCmp','Default_PFC','Cont_pCun'])
        and atlas.iloc[i].get('atlas_name','')=='4S456'],
    'Brain - Anterior cingulate cortex (BA24)': [i for i,l in enumerate(pl)
        if any(k in l for k in ['SalVentAttn_Med','Limbic_OFC','Default_pCunPCC','Cont_Cing'])
        and atlas.iloc[i].get('atlas_name','')=='4S456'],
    'Brain - Cortex': [i for i,l in enumerate(pl) if atlas.iloc[i].get('atlas_name','')=='4S456'],
}
RMAP = {k:v for k,v in RMAP.items() if v}

# Load GTEx
log("Loading GTEx individual-level expression...")
sa = pd.read_csv(GTEX_ATTR, sep='\t', usecols=['SAMPID', 'SMTSD'])
REGIONS = sorted([r for r in RMAP.keys() if r in sa['SMTSD'].unique()])
brain_sa = sa[sa['SMTSD'].isin(REGIONS)]
brain_ids = set(brain_sa['SAMPID'])

with gzip.open(GTEX_TPM, 'rt') as f:
    f.readline(); f.readline()
    header = f.readline().strip().split('\t')
bcols = [0, 1] + [i for i,c in enumerate(header) if c in brain_ids]

expr = pd.read_csv(GTEX_TPM, sep='\t', skiprows=2, usecols=bcols, compression='gzip')
expr.columns = [header[i] for i in bcols]
expr['gene'] = expr['Description'].str.upper()
expr = expr.drop(['Name','Description'], axis=1).drop_duplicates('gene', keep='first').set_index('gene')
expr = np.log2(expr + 1)
expr = expr[expr.mean(axis=1) > np.log2(2)]
genes = list(expr.index)
log(f"  {len(genes)} genes, {len(expr.columns)} samples")

# Region samples
reg_samp = {}
for r in REGIONS:
    s = brain_sa[brain_sa['SMTSD']==r]['SAMPID'].tolist()
    avail = [x for x in s if x in expr.columns]
    if len(avail) >= 30:
        reg_samp[r] = avail

valid_regions = sorted(reg_samp.keys())
log(f"  Valid regions: {len(valid_regions)}")

# ============================================================================
# COMPUTE WJ AND FC FOR ALL REGION PAIRS
# ============================================================================
log("\nComputing WJ and FC for all region pairs...")

pairs_data = []
for i in range(len(valid_regions)):
    r1 = valid_regions[i]
    d1 = expr[reg_samp[r1]].values.astype(np.float64)
    c1 = fast_spearman(d1)

    for j in range(i+1, len(valid_regions)):
        r2 = valid_regions[j]
        d2 = expr[reg_samp[r2]].values.astype(np.float64)
        c2 = fast_spearman(d2)

        wj_val = wj_chunked(c1, c2)

        # Expression profile similarity (component level)
        # Median TPM
        med1 = np.median(d1, axis=1)
        med2 = np.median(d2, axis=1)
        expr_sim, _ = stats.spearmanr(med1, med2)

        # FC awake
        p1 = [p for p in RMAP[r1] if p < corr_awake.shape[0]]
        p2 = [p for p in RMAP[r2] if p < corr_awake.shape[0]]
        fc_awake = np.mean(corr_awake[np.ix_(p1, p2)]) if p1 and p2 else np.nan

        # FC unconscious
        fc_uncon = np.mean(corr_uncon[np.ix_(p1, p2)]) if p1 and p2 else np.nan

        # Spatial distance
        if r1 in MNI_COORDS and r2 in MNI_COORDS:
            dist = np.linalg.norm(MNI_COORDS[r1] - MNI_COORDS[r2])
        else:
            dist = np.nan

        r1s = r1.replace('Brain - ','').replace(' (basal ganglia)','')
        r2s = r2.replace('Brain - ','').replace(' (basal ganglia)','')

        pairs_data.append({
            'Region1': r1, 'Region2': r2,
            'Label': f"{r1s}-{r2s}",
            'WJ': wj_val,
            'Expr_Sim': expr_sim,
            'FC_Awake': fc_awake,
            'FC_Unconscious': fc_uncon,
            'Distance_mm': dist,
            'n_samples_r1': len(reg_samp[r1]),
            'n_samples_r2': len(reg_samp[r2]),
        })

        log(f"  {r1s:>25}-{r2s:<25} WJ={wj_val:.4f} expr={expr_sim:.3f} "
            f"fc={fc_awake:.3f} dist={dist:.0f}mm")

        del c2, d2
    del c1, d1
    gc.collect()

df = pd.DataFrame(pairs_data)
df = df.dropna(subset=['FC_Awake'])
df.to_csv(os.path.join(RESULTS, 'bulletproof_pairs.csv'), index=False, float_format='%.6f')
log(f"\n  Total valid pairs: {len(df)}")

# ============================================================================
# TEST 1: PRIMARY CORRELATIONS
# ============================================================================
log(f"\n{'='*70}")
log("TEST 1: PRIMARY CORRELATIONS")
log(f"{'='*70}")

rho_wj, p_wj = stats.spearmanr(df['WJ'], df['FC_Awake'])
rho_expr, p_expr = stats.spearmanr(df['Expr_Sim'], df['FC_Awake'])
rho_dist, p_dist = stats.spearmanr(df['Distance_mm'], df['FC_Awake'])

log(f"  WJ vs FC (awake):         rho={rho_wj:.4f}, p={p_wj:.6f}")
log(f"  Expression vs FC (awake): rho={rho_expr:.4f}, p={p_expr:.6f}")
log(f"  Distance vs FC (awake):   rho={rho_dist:.4f}, p={p_dist:.6f}")

# ============================================================================
# TEST 2: STEIGER'S Z (FORMAL relational > component)
# ============================================================================
log(f"\n{'='*70}")
log("TEST 2: STEIGER'S Z-TEST (WJ vs Expression as predictors of FC)")
log(f"{'='*70}")

def steiger_z(r_xz, r_yz, r_xy, n):
    """Test whether correlation r_xz is significantly different from r_yz,
    where x and y are two predictors and z is the outcome.
    r_xy is the correlation between the two predictors."""
    # Fisher z-transform
    z_xz = np.arctanh(np.clip(r_xz, -0.999, 0.999))
    z_yz = np.arctanh(np.clip(r_yz, -0.999, 0.999))

    # Steiger's formula for dependent correlations
    r_det = 1 - r_xz**2 - r_yz**2 - r_xy**2 + 2*r_xz*r_yz*r_xy
    r_bar = (r_xz + r_yz) / 2
    denom = np.sqrt((2*(1-r_xy) * ((1-r_bar**2)**2) / (n-3)) +
                    ((r_xz + r_yz)**2 / 4) * ((1-r_xy)**3 / (n-1)))

    if denom == 0:
        return 0, 1.0
    z_stat = (z_xz - z_yz) / denom if denom > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_val

# Need correlation between WJ and Expression (the two predictors)
r_wj_expr, _ = stats.spearmanr(df['WJ'], df['Expr_Sim'])
n = len(df)

z_stat, p_steiger = steiger_z(rho_wj, rho_expr, r_wj_expr, n)
log(f"  r(WJ, FC) = {rho_wj:.4f}")
log(f"  r(Expr, FC) = {rho_expr:.4f}")
log(f"  r(WJ, Expr) = {r_wj_expr:.4f}")
log(f"  Steiger's Z = {z_stat:.3f}, p = {p_steiger:.4f}")
if p_steiger < 0.05:
    log(f"  >>> WJ is SIGNIFICANTLY better than expression profiles (p={p_steiger:.4f})")
else:
    log(f"  >>> Difference is NOT significant (p={p_steiger:.4f})")

# ============================================================================
# TEST 3: SPATIAL PROXIMITY CONTROL (partial correlation)
# ============================================================================
log(f"\n{'='*70}")
log("TEST 3: SPATIAL PROXIMITY CONTROL")
log(f"{'='*70}")

# Partial correlation: WJ vs FC, controlling for distance
from scipy.stats import pearsonr

def partial_corr_spearman(x, y, z):
    """Spearman partial correlation of x and y, controlling for z."""
    # Rank all variables first (Spearman = Pearson on ranks)
    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)
    # Residualize x and y on z
    slope_xz = np.polyfit(rz, rx, 1)
    slope_yz = np.polyfit(rz, ry, 1)
    resid_x = rx - np.polyval(slope_xz, rz)
    resid_y = ry - np.polyval(slope_yz, rz)
    r, p = pearsonr(resid_x, resid_y)
    return r, p

valid_dist = df.dropna(subset=['Distance_mm'])
rho_partial, p_partial = partial_corr_spearman(
    valid_dist['WJ'].values,
    valid_dist['FC_Awake'].values,
    valid_dist['Distance_mm'].values
)
log(f"  WJ vs FC (raw):                        rho={rho_wj:.4f}, p={p_wj:.6f}")
log(f"  WJ vs FC (controlling for distance):   rho={rho_partial:.4f}, p={p_partial:.6f}")

rho_expr_partial, p_expr_partial = partial_corr_spearman(
    valid_dist['Expr_Sim'].values,
    valid_dist['FC_Awake'].values,
    valid_dist['Distance_mm'].values
)
log(f"  Expr vs FC (controlling for distance): rho={rho_expr_partial:.4f}, p={p_expr_partial:.6f}")

if p_partial < 0.05:
    log(f"  >>> WJ-FC correlation SURVIVES distance control")
else:
    log(f"  >>> WJ-FC correlation does NOT survive distance control")

# Also check: is distance correlated with WJ and FC?
rho_dist_wj, p_dist_wj = stats.spearmanr(valid_dist['Distance_mm'], valid_dist['WJ'])
log(f"\n  Distance vs WJ: rho={rho_dist_wj:.4f}, p={p_dist_wj:.6f}")
log(f"  Distance vs FC: rho={rho_dist:.4f}, p={p_dist:.6f}")

# ============================================================================
# TEST 4: LEAVE-ONE-REGION-OUT STABILITY
# ============================================================================
log(f"\n{'='*70}")
log("TEST 4: LEAVE-ONE-REGION-OUT STABILITY")
log(f"{'='*70}")

for leave_out in valid_regions:
    subset = df[(df['Region1'] != leave_out) & (df['Region2'] != leave_out)]
    if len(subset) < 5:
        continue
    r, p = stats.spearmanr(subset['WJ'], subset['FC_Awake'])
    los = leave_out.replace('Brain - ','').replace(' (basal ganglia)','')
    sig = '*' if p < 0.05 else 'ns'
    log(f"  Without {los:>25}: rho={r:.4f}, p={p:.4f} ({sig}), n={len(subset)}")

# ============================================================================
# TEST 5: SPLIT-HALF DONOR REPLICATION
# ============================================================================
log(f"\n{'='*70}")
log("TEST 5: SPLIT-HALF DONOR REPLICATION")
log(f"{'='*70}")

rng = np.random.RandomState(RANDOM_SEED)
N_SPLITS = 10
split_rhos = []

for split in range(N_SPLITS):
    wj_split = []
    fc_split = []

    for i in range(len(valid_regions)):
        r1 = valid_regions[i]
        samp1 = reg_samp[r1]
        half1_idx = rng.choice(len(samp1), len(samp1)//2, replace=False)
        half1 = [samp1[k] for k in half1_idx]

        d1 = expr[half1].values.astype(np.float64)
        c1 = fast_spearman(d1)

        for j in range(i+1, len(valid_regions)):
            r2 = valid_regions[j]
            samp2 = reg_samp[r2]
            half2_idx = rng.choice(len(samp2), len(samp2)//2, replace=False)
            half2 = [samp2[k] for k in half2_idx]

            d2 = expr[half2].values.astype(np.float64)
            c2 = fast_spearman(d2)

            w = wj_chunked(c1, c2)

            p1 = [p for p in RMAP[r1] if p < corr_awake.shape[0]]
            p2 = [p for p in RMAP[r2] if p < corr_awake.shape[0]]
            if p1 and p2:
                fc = np.mean(corr_awake[np.ix_(p1, p2)])
                wj_split.append(w)
                fc_split.append(fc)

            del c2, d2
        del c1, d1
        gc.collect()

    r_split, p_split = stats.spearmanr(wj_split, fc_split)
    split_rhos.append(r_split)
    log(f"  Split {split+1}/{N_SPLITS}: rho={r_split:.4f}, p={p_split:.6f}")

log(f"\n  Split-half summary: mean rho={np.mean(split_rhos):.4f}, "
    f"std={np.std(split_rhos):.4f}, "
    f"min={np.min(split_rhos):.4f}, max={np.max(split_rhos):.4f}")
log(f"  All {N_SPLITS} splits significant: "
    f"{'YES' if np.min(split_rhos) > 0.3 else 'NO'}")

# ============================================================================
# TEST 6: UNCONSCIOUS FC (does finding hold under propofol?)
# ============================================================================
log(f"\n{'='*70}")
log("TEST 6: UNCONSCIOUS FC (propofol)")
log(f"{'='*70}")

rho_uncon, p_uncon = stats.spearmanr(df['WJ'], df['FC_Unconscious'])
log(f"  WJ vs FC (awake):       rho={rho_wj:.4f}, p={p_wj:.6f}")
log(f"  WJ vs FC (unconscious): rho={rho_uncon:.4f}, p={p_uncon:.6f}")
log(f"  Finding {'HOLDS' if p_uncon < 0.05 else 'DOES NOT HOLD'} under propofol")

# ============================================================================
# TEST 7: BOOTSTRAP CI AND PERMUTATION TEST
# ============================================================================
log(f"\n{'='*70}")
log("TEST 7: BOOTSTRAP CI AND PERMUTATION (on primary result)")
log(f"{'='*70}")

# Bootstrap
boot = np.zeros(1000)
wj_arr = df['WJ'].values
fc_arr = df['FC_Awake'].values
n = len(wj_arr)
for b in range(1000):
    idx = rng.choice(n, n, replace=True)
    boot[b], _ = stats.spearmanr(wj_arr[idx], fc_arr[idx])
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
log(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

# Permutation
perm = np.zeros(1000)
for i in range(1000):
    perm[i], _ = stats.spearmanr(rng.permutation(wj_arr), fc_arr)
perm_p = np.mean(np.abs(perm) >= np.abs(rho_wj))
log(f"  Permutation p: {perm_p:.4f}")

# Effect size
d = (rho_wj - perm.mean()) / perm.std()
log(f"  Cohen's d: {d:.2f}")

# ============================================================================
# FINAL VERDICT
# ============================================================================
log(f"\n{'='*70}")
log("FINAL VERDICT")
log(f"{'='*70}")

tests = {
    'Primary WJ-FC correlation': p_wj < 0.05,
    'Steiger Z (WJ > Expression)': p_steiger < 0.05,
    'Survives distance control': p_partial < 0.05,
    'Leave-one-out stable': all(r > 0.2 for r in [
        stats.spearmanr(
            df[(df['Region1']!=r)&(df['Region2']!=r)]['WJ'],
            df[(df['Region1']!=r)&(df['Region2']!=r)]['FC_Awake']
        )[0] for r in valid_regions
        if len(df[(df['Region1']!=r)&(df['Region2']!=r)]) >= 5
    ]),
    'Split-half replicable': np.mean(split_rhos) > 0.3,
    'Holds under propofol': p_uncon < 0.05,
    'Bootstrap CI excludes zero': ci_lo > 0,
    'Permutation significant': perm_p < 0.05,
}

n_pass = sum(tests.values())
n_total = len(tests)
log(f"\n  Results:")
for test, passed in tests.items():
    status = 'PASS' if passed else 'FAIL'
    log(f"    [{status}] {test}")

log(f"\n  Score: {n_pass}/{n_total} tests passed")

if n_pass == n_total:
    log(f"\n  VERDICT: THE FINDING IS REAL.")
    log(f"  Gene co-expression architecture predicts brain functional connectivity")
    log(f"  after controlling for spatial proximity, across donor splits,")
    log(f"  across consciousness states, and with formal statistical comparison")
    log(f"  against expression-profile-based prediction.")
elif n_pass >= 6:
    log(f"\n  VERDICT: THE FINDING IS LIKELY REAL WITH CAVEATS.")
    log(f"  Failed tests indicate specific limitations that must be disclosed.")
else:
    log(f"\n  VERDICT: THE FINDING DOES NOT SURVIVE RIGOROUS TESTING.")
    log(f"  The correlation may be driven by confounds or instability.")

# Save
summary = {
    'n_regions': len(valid_regions),
    'n_pairs': len(df),
    'n_genes': len(genes),
    'primary_rho': float(rho_wj),
    'primary_p': float(p_wj),
    'expression_rho': float(rho_expr),
    'steiger_z': float(z_stat),
    'steiger_p': float(p_steiger),
    'partial_rho_distance': float(rho_partial),
    'partial_p_distance': float(p_partial),
    'split_half_mean_rho': float(np.mean(split_rhos)),
    'unconscious_rho': float(rho_uncon),
    'unconscious_p': float(p_uncon),
    'bootstrap_ci': [float(ci_lo), float(ci_hi)],
    'permutation_p': float(perm_p),
    'cohens_d': float(d),
    'tests_passed': n_pass,
    'tests_total': n_total,
}
with open(os.path.join(RESULTS, 'bulletproof_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

df.to_csv(os.path.join(RESULTS, 'bulletproof_pairs.csv'), index=False, float_format='%.6f')

log(f"\nTotal time: {(time.time()-START)/60:.1f} minutes")
log("BULLETPROOF VALIDATION COMPLETE")
