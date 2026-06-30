# -*- coding: utf-8 -*-
"""
Pipeline: Gene-to-Brain cell-type composition confound check
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-06-30
Description:
    Tests whether the WJ-to-functional-connectivity correspondence (the manuscript
    headline) is explained by differences in cell-type composition across brain
    regions, the universal reviewer objection for bulk brain transcriptomics
    (Domain Confound Registry). Builds a per-region cell-type composition profile
    from canonical marker genes (neuron, astrocyte, oligodendrocyte, microglia,
    endothelial) in GTEx, forms a region-by-region cell-type-composition distance
    matrix, and runs partial Mantel of WJ vs FC controlling for cell-type distance.
    If the WJ-FC correspondence survives, it is not a cell-type artifact.

    The WJ, FC, and Euclidean-distance matrices are reconstructed directly from
    results/rebuild/pairwise_metrics.csv (sorted region order, matching the rebuild
    saved matrices), so only the marker genes are loaded from GTEx.
Dependencies: numpy, pandas, scipy
Input: results/rebuild/pairwise_metrics.csv ; GTEx_v8_tpm.gct.gz ; GTEx_v8_sample_attributes.txt
Output: results/celltype_confound/
"""
import os, sys, json, gzip, time
import numpy as np, pandas as pd
from scipy.stats import pearsonr, rankdata
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RANDOM_SEED = 42; FORCE_RECOMPUTE = True; N_PERM = 10000
np.random.seed(RANDOM_SEED)
BASE = r"G:/My Drive/inner_architecture_research/gene_to_brain_wj"
PAIRS = os.path.join(BASE, "results", "rebuild", "pairwise_metrics.csv")
GTEX_TPM = r"G:/My Drive/inner_architecture_research/MS3_JNC_Submission/data/GTEx_v8_tpm.gct.gz"
GTEX_ATTR = r"G:/My Drive/inner_architecture_research/MS3_JNC_Submission/data/GTEx_v8_sample_attributes.txt"
OUT = os.path.join(BASE, "results", "celltype_confound"); os.makedirs(OUT, exist_ok=True)

MARKERS = {
    "neuron":      ["RBFOX3","SYT1","SNAP25","SYN1","ENO2","GAD1","SLC17A7"],
    "astrocyte":   ["GFAP","AQP4","ALDH1L1","SLC1A3","SLC1A2"],
    "oligodendro": ["MBP","MOBP","PLP1","MOG","MAG"],
    "microglia":   ["AIF1","CX3CR1","PTPRC","CSF1R","C1QA"],
    "endothelial": ["CLDN5","FLT1","VWF","PECAM1"],
}
ALL_MARKERS = {g for v in MARKERS.values() for g in v}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def tri(m):
    iu = np.triu_indices(m.shape[0], k=1); return m[iu]

def mantel(A, B, nperm, rng):
    a, b = tri(A), tri(B); r = pearsonr(a, b)[0]; n = A.shape[0]; cnt = 0
    for _ in range(nperm):
        p = rng.permutation(n); rp = pearsonr(tri(A[np.ix_(p, p)]), b)[0]
        if abs(rp) >= abs(r): cnt += 1
    return float(r), float((cnt + 1) / (nperm + 1))

def resid(y, X):
    X1 = np.column_stack([np.ones(len(X)), X]); beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta

def partial_mantel(A, B, C, nperm, rng):
    # residualize upper-tri of A and B on C, then Mantel-permute residualized A-matrix
    a, b, c = tri(A), tri(B), tri(C)
    ra, rb = resid(a, c), resid(b, c); r = pearsonr(ra, rb)[0]; n = A.shape[0]
    # rebuild residualized A as a matrix for permutation
    from scipy.spatial.distance import squareform
    RA = squareform(ra, checks=False)
    cnt = 0
    for _ in range(nperm):
        p = rng.permutation(n); rp = pearsonr(tri(RA[np.ix_(p, p)]), rb)[0]
        if abs(rp) >= abs(r): cnt += 1
    return float(r), float((cnt + 1) / (nperm + 1))

def main():
    df = pd.read_csv(PAIRS)
    regions = sorted(set(df["Region1"]) | set(df["Region2"]))
    idx = {r: i for i, r in enumerate(regions)}; N = len(regions)
    log(f"{N} regions (sorted order matches rebuild matrices)")
    WJ = np.eye(N); FCa = np.eye(N); FCu = np.eye(N); DIST = np.zeros((N, N))
    for _, row in df.iterrows():
        i, j = idx[row["Region1"]], idx[row["Region2"]]
        for M, col in [(WJ, "wj_unsigned"), (FCa, "fc_awake"), (FCu, "fc_unconscious"), (DIST, "distance_mm")]:
            M[i, j] = M[j, i] = row[col]

    # ---- cell-type composition from GTEx markers ----
    log("loading GTEx sample attributes...")
    attr = pd.read_csv(GTEX_ATTR, sep="\t", usecols=["SAMPID", "SMTSD"])
    samp2reg = dict(zip(attr["SAMPID"], attr["SMTSD"]))
    log("streaming GTEx gct for marker genes (this is the slow step)...")
    marker_expr = {}  # gene -> {region: [values]}
    with gzip.open(GTEX_TPM, "rt") as f:
        f.readline(); f.readline()  # gct version + dims
        header = f.readline().rstrip("\n").split("\t")
        cols = header[2:]; col_regions = [samp2reg.get(c) for c in cols]
        keep_col = [k for k, r in enumerate(col_regions) if r in regions]
        kept_regions = [col_regions[k] for k in keep_col]
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sym = parts[1]
            if sym not in ALL_MARKERS: continue
            vals = np.array([float(parts[2 + k]) for k in keep_col])
            per = {}
            for r in regions:
                m = [v for v, rr in zip(vals, kept_regions) if rr == r]
                per[r] = float(np.mean(m)) if m else np.nan
            marker_expr[sym] = per
    log(f"  extracted {len(marker_expr)} of {len(ALL_MARKERS)} marker genes")

    # per-region cell-type vector (mean log2(TPM+1) of each cell type's markers)
    celltype_vec = np.full((N, len(MARKERS)), np.nan)
    for ci, (ct, genes) in enumerate(MARKERS.items()):
        present = [g for g in genes if g in marker_expr]
        for r in regions:
            vals = [np.log2(marker_expr[g][r] + 1) for g in present if not np.isnan(marker_expr[g][r])]
            celltype_vec[idx[r], ci] = np.mean(vals) if vals else np.nan
    # z-score each cell-type column, then Euclidean distance between regions
    cz = (celltype_vec - np.nanmean(celltype_vec, 0)) / np.nanstd(celltype_vec, 0)
    CELL = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            CELL[i, j] = np.sqrt(np.nansum((cz[i] - cz[j]) ** 2))

    rng = np.random.RandomState(RANDOM_SEED)
    res = {}
    r, p = mantel(WJ, FCu, N_PERM, rng); res["WJ_vs_FCuncon"] = {"mantel_r": r, "p": p}
    r, p = mantel(WJ, FCa, N_PERM, rng); res["WJ_vs_FCawake"] = {"mantel_r": r, "p": p}
    r, p = mantel(CELL, WJ, N_PERM, rng); res["celltype_vs_WJ"] = {"mantel_r": r, "p": p}
    r, p = mantel(CELL, FCu, N_PERM, rng); res["celltype_vs_FCuncon"] = {"mantel_r": r, "p": p}
    r, p = partial_mantel(WJ, FCu, CELL, N_PERM, rng); res["WJ_vs_FCuncon_PARTIAL_celltype"] = {"partial_mantel_r": r, "p": p}
    # control for both distance and cell-type
    a, b = tri(WJ), tri(FCu); C = np.column_stack([tri(DIST), tri(CELL)])
    ra, rb = resid(a, C), resid(b, C); rboth = pearsonr(ra, rb)[0]
    from scipy.spatial.distance import squareform
    RA = squareform(ra, checks=False); cnt = 0
    for _ in range(N_PERM):
        pmt = rng.permutation(N)
        if abs(pearsonr(tri(RA[np.ix_(pmt, pmt)]), rb)[0]) >= abs(rboth): cnt += 1
    res["WJ_vs_FCuncon_PARTIAL_distance_AND_celltype"] = {"partial_r": float(rboth), "p": float((cnt + 1) / (N_PERM + 1))}

    survives = res["WJ_vs_FCuncon_PARTIAL_celltype"]["p"] < 0.05
    verdict = ("WJ-FC correspondence SURVIVES cell-type adjustment (not a cell-type artifact)"
               if survives else
               "WJ-FC correspondence does NOT survive cell-type adjustment (cell-type composition is a confound)")
    out = {"regions": regions, "markers_found": sorted(marker_expr.keys()),
           "results": res, "verdict": verdict, "random_seed": RANDOM_SEED,
           "n_perm": N_PERM, "execution_date": "2026-06-30"}
    json.dump(out, open(os.path.join(OUT, "celltype_confound.json"), "w"), indent=2)
    pd.DataFrame(celltype_vec, index=regions, columns=list(MARKERS.keys())).to_csv(
        os.path.join(OUT, "per_region_celltype_scores.csv"))
    log("\n=== RESULTS ===")
    for k, v in res.items(): log(f"  {k}: {v}")
    log(f"\nVERDICT: {verdict}")
    log(f"-> {OUT}")

if __name__ == "__main__": main()
