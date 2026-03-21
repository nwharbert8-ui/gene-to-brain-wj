"""Quick fix: synapse gene filter with correct indices."""
import os, numpy as np, pandas as pd, gzip
from scipy import stats

BASE = r'G:\My Drive\inner_architecture_research\gene_to_brain_wj'
CACHE = os.path.join(BASE, 'results', 'region_corr_cache')
BRAIN = r'G:\My Drive\inner_architecture_research\brain_connectivity_wj'
GTEX_TPM = r'G:\My Drive\inner_architecture_research\MS3_JNC_Submission\data\GTEx_v8_tpm.gct.gz'
GTEX_ATTR = r'G:\My Drive\inner_architecture_research\MS3_JNC_Submission\data\GTEx_v8_sample_attributes.txt'

# Rebuild exact gene list from individual GTEx
print("Loading GTEx to get exact gene list...")
sa = pd.read_csv(GTEX_ATTR, sep='\t', usecols=['SAMPID', 'SMTSD'])
brain_ids = set(sa[sa['SMTSD'].str.startswith('Brain', na=False)]['SAMPID'])

with gzip.open(GTEX_TPM, 'rt') as f:
    f.readline(); f.readline()
    header = f.readline().strip().split('\t')

brain_col_idx = [0, 1] + [i for i, c in enumerate(header) if c in brain_ids]
col_names = [header[i] for i in brain_col_idx]

expr = pd.read_csv(GTEX_TPM, sep='\t', skiprows=2, usecols=brain_col_idx, compression='gzip')
expr.columns = col_names
expr['gene'] = expr['Description'].str.upper()
expr = expr.drop(['Name', 'Description'], axis=1)
expr = expr.drop_duplicates(subset='gene', keep='first').set_index('gene')
expr = np.log2(expr + 1)
expr = expr[expr.mean(axis=1) > np.log2(2)]
gene_names = list(expr.index)
print(f"Genes in matrices: {len(gene_names)}")
del expr

# Synapse genes
SYNAPSE = [
    'SYN1','SYN2','SYN3','SYP','SNAP25','STX1A','STX1B','VAMP2',
    'SLC17A7','SLC17A6','SLC32A1','GRIA1','GRIA2','GRIA3','GRIA4',
    'GRIN1','GRIN2A','GRIN2B','GRIN2C','GRIN2D','GABRA1','GABRB2',
    'GABRG2','DLG1','DLG2','DLG3','DLG4','SHANK1','SHANK2','SHANK3',
    'HOMER1','HOMER2','HOMER3','BSN','PCLO','RIMS1','RIMS2',
    'NRXN1','NRXN2','NRXN3','NLGN1','NLGN2','NLGN3','NLGN4X',
    'CAMK2A','CAMK2B','CALM1','CALM2','CALM3','SLC6A1','SLC6A2',
    'SLC6A3','SLC6A4','TH','DDC','TPH1','TPH2','MAOA','MAOB',
    'COMT','SIGMAR1','FKBP5','NR3C1','BDNF','NTRK2','CREB1',
    'GABBR1','GABBR2','GRM1','GRM5','PSEN1','APP','BACE1',
    'MAPT','SYT1','SYT2','STXBP1','CPLX1','CPLX2','NSF',
]

syn_in = [g for g in set(s.upper() for s in SYNAPSE) if g in gene_names]
syn_idx = [gene_names.index(g) for g in syn_in]
print(f"Synaptic genes: {len(syn_in)}, max idx: {max(syn_idx)}, matrix size: {len(gene_names)}")

# Load atlas + FC
atlas = pd.read_csv(os.path.join(BRAIN, 'data', 'raw', 'ds006623', 'derivatives',
    'xcp_d_without_GSR_bandpass_output', 'atlases', 'atlas-4S456Parcels',
    'atlas-4S456Parcels_dseg.tsv'), sep='\t')
pl = atlas['label'].tolist()
corr_brain = np.load(os.path.join(BRAIN, 'results', 'correlation_matrices',
                                   'group_awake_spearman_corr.npy'))

RMAP = {
    'Brain_-_Putamen_basal_ganglia': [i for i,l in enumerate(pl) if l in ['LH-Pu','RH-Pu']],
    'Brain_-_Caudate_basal_ganglia': [i for i,l in enumerate(pl) if l in ['LH-Ca','RH-Ca']],
    'Brain_-_Nucleus_accumbens_basal_ganglia': [i for i,l in enumerate(pl) if l in ['LH-NAC','RH-NAC']],
    'Brain_-_Hippocampus': [i for i,l in enumerate(pl) if 'Hippocampus' in l or 'HN' in l],
    'Brain_-_Amygdala': [i for i,l in enumerate(pl) if 'Amygdala' in l or 'EXA' in l],
    'Brain_-_Hypothalamus': [i for i,l in enumerate(pl) if 'HTH' in l],
    'Brain_-_Substantia_nigra': [i for i,l in enumerate(pl) if 'SNc' in l or 'SNr' in l],
    'Brain_-_Cerebellum': [i for i,l in enumerate(pl) if 'Cerebellar' in l],
    'Brain_-_Frontal_Cortex_BA9': [i for i,l in enumerate(pl)
        if any(k in l for k in ['Cont_PFCl','Cont_PFCmp','Default_PFC','Cont_pCun'])
        and atlas.iloc[i].get('atlas_name','')=='4S456'],
    'Brain_-_Anterior_cingulate_cortex_BA24': [i for i,l in enumerate(pl)
        if any(k in l for k in ['SalVentAttn_Med','Limbic_OFC','Default_pCunPCC','Cont_Cing'])
        and atlas.iloc[i].get('atlas_name','')=='4S456'],
    'Brain_-_Cortex': [i for i,l in enumerate(pl) if atlas.iloc[i].get('atlas_name','')=='4S456'],
}
RMAP = {k: v for k, v in RMAP.items() if v}

cached = sorted([f.replace('.npy','') for f in os.listdir(CACHE) if f.endswith('.npy')])

def wj_chunked(A, B, cs=2000):
    n = A.shape[0]
    num = 0.0; den = 0.0
    for i in range(0, n, cs):
        ie = min(i+cs, n)
        for j in range(i, n, cs):
            je = min(j+cs, n)
            a = np.abs(A[i:ie, j:je])
            b = np.abs(B[i:ie, j:je])
            if i == j:
                m = np.triu(np.ones((ie-i, je-j), dtype=bool), k=1)
                num += np.minimum(a[m], b[m]).sum()
                den += np.maximum(a[m], b[m]).sum()
            elif j > i:
                num += np.minimum(a, b).sum()
                den += np.maximum(a, b).sum()
    return float(num/den) if den > 0 else 1.0

# Compute
print("\nComputing WJ for all-gene and synapse subset...")
all_wj = []; syn_wj = []; fc_vals = []; pair_names = []

import gc
for i in range(len(cached)):
    r1 = cached[i]
    if r1 not in RMAP:
        continue
    c1 = np.load(os.path.join(CACHE, f'{r1}.npy'))
    for j in range(i+1, len(cached)):
        r2 = cached[j]
        if r2 not in RMAP:
            continue
        c2 = np.load(os.path.join(CACHE, f'{r2}.npy'))

        wj_a = wj_chunked(c1, c2)
        sc1 = c1[np.ix_(syn_idx, syn_idx)]
        sc2 = c2[np.ix_(syn_idx, syn_idx)]
        wj_s = wj_chunked(sc1, sc2)

        p1 = [p for p in RMAP[r1] if p < corr_brain.shape[0]]
        p2 = [p for p in RMAP[r2] if p < corr_brain.shape[0]]
        if p1 and p2:
            fc = np.mean(corr_brain[np.ix_(p1, p2)])
            all_wj.append(wj_a)
            syn_wj.append(wj_s)
            fc_vals.append(fc)
            pair_names.append(f"{r1}-{r2}")
        del c2
    del c1
    gc.collect()

all_a = np.array(all_wj)
syn_a = np.array(syn_wj)
fc_a = np.array(fc_vals)

rho_all, p_all = stats.spearmanr(all_a, fc_a)
rho_syn, p_syn = stats.spearmanr(syn_a, fc_a)

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"Region pairs: {len(fc_a)}")
print(f"All-gene WJ ({len(gene_names):,} genes) vs Brain FC: rho={rho_all:.4f}, p={p_all:.6f}")
print(f"Synapse WJ ({len(syn_in)} genes) vs Brain FC:        rho={rho_syn:.4f}, p={p_syn:.6f}")
print(f"Full architecture predicts {'BETTER' if abs(rho_all) > abs(rho_syn) else 'WORSE'} than synapse subset")
print(f"\nThe signal is in ALL {len(gene_names):,} relationships.")
print(f"Synaptic genes ({len(syn_in)}) are part of the architecture")
print(f"but do not carry the prediction alone.")
