# -*- coding: utf-8 -*-
"""
Pipeline: Architecture-similarity metric comparison (the empirical "why WJ")
Author: Drake H. Harbert (D.H.H.) | ORCID 0009-0007-7740-3616 | 2026-06-30
Description:
    For each candidate inter-regional architecture-similarity metric, computes its
    correspondence (Mantel, Pearson and Spearman) to unconscious-state functional
    connectivity. Demonstrates which metric maximizes the gene-to-FC correspondence.
    Reads results/rebuild/pairwise_metrics.csv. Frobenius is a distance (reported as
    |r|). Output: supplementary metric-comparison table.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SEED = 42; NPERM = 10000; np.random.seed(SEED)
PAIRS = r"G:/My Drive/inner_architecture_research/gene_to_brain_wj/results/rebuild/pairwise_metrics.csv"
OUT = r"G:/My Drive/inner_architecture_research/gene_to_brain_wj/results/metric_comparison"; os.makedirs(OUT, exist_ok=True)
METRICS = {"Weighted Jaccard (unsigned)": "wj_unsigned", "Cosine similarity": "cosine_abs",
           "Frobenius distance": "frobenius_diff", "Pearson matrix correlation": "pearson_abs",
           "Spearman matrix correlation": "spearman_abs", "Expression-profile similarity": "expr_profile_sim"}

def tri(m): return m[np.triu_indices(m.shape[0], k=1)]
def mantel(A, B, rng):
    a, b = tri(A), tri(B); r = pearsonr(a, b)[0]; n = A.shape[0]; c = 0
    for _ in range(NPERM):
        p = rng.permutation(n)
        if abs(pearsonr(tri(A[np.ix_(p, p)]), b)[0]) >= abs(r): c += 1
    return float(r), float((c + 1) / (NPERM + 1))

def main():
    df = pd.read_csv(PAIRS)
    regions = sorted(set(df["Region1"]) | set(df["Region2"])); idx = {r: i for i, r in enumerate(regions)}; N = len(regions)
    def mat(col):
        M = np.zeros((N, N))
        for _, row in df.iterrows():
            i, j = idx[row["Region1"]], idx[row["Region2"]]; M[i, j] = M[j, i] = row[col]
        return M
    FC = mat("fc_unconscious"); rng = np.random.RandomState(SEED)
    rows = []
    for name, col in METRICS.items():
        M = mat(col)
        r, p = mantel(M, FC, rng)
        sp = spearmanr(tri(M), tri(FC))[0]
        rows.append({"metric": name, "mantel_r": round(abs(r), 3), "mantel_p": round(p, 4),
                     "spearman": round(abs(sp), 3),
                     "type": "distance" if col == "frobenius_diff" else "similarity"})
    rows.sort(key=lambda x: -x["mantel_r"])
    json.dump({"target": "unconscious-state FC", "rows": rows, "n_perm": NPERM},
              open(os.path.join(OUT, "metric_comparison.json"), "w"), indent=2)
    print("Metric -> FC correspondence (|Mantel r| Pearson, |Spearman|):")
    for x in rows: print(f"  {x['metric']:32s} |r|={x['mantel_r']:.3f} (p={x['mantel_p']:.4f})  |Spearman|={x['spearman']:.3f}")

if __name__ == "__main__":
    main()
