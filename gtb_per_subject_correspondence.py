# -*- coding: utf-8 -*-
"""
Pipeline: Per-subject, all-three-condition WJ-to-FC correspondence (exploratory)
Author: Drake H. Harbert (D.H.H.) | ORCID 0009-0007-7740-3616
Affiliation: Inner Architecture LLC, Canton, OH | Date: 2026-06-30
Description:
    Measures, for every subject and every condition (awake, unconscious, recovery),
    how well gene co-expression architecture (WJ) corresponds to that individual's
    functional connectivity, via Mantel. Reports the full per-subject distribution
    per condition, tests each against zero, and relates per-subject correspondence
    to any available dosing/demographic variable. No hypothesis is presupposed; all
    available data (every subject, all three states) is measured and reported.
    Per-subject FC is aggregated from the 456-parcel matrices to the 11 GTEx regions
    using the rebuild's exact region->parcel map and block-mean aggregation.
Dependencies: numpy, pandas, scipy ; imports gtb_rebuild_from_scratch (region map)
Output: results/per_subject/
"""
import os, sys, glob, json, re
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gtb_rebuild_from_scratch as gtb  # reuse build_region_parcel_map

SEED = 42; NPERM = 5000; np.random.seed(SEED)
BRAIN = r"G:/My Drive/inner_architecture_research/brain_connectivity_wj"
CM = os.path.join(BRAIN, "results", "correlation_matrices")
ATLAS = os.path.join(BRAIN, "data", "raw", "ds006623", "derivatives",
                     "xcp_d_without_GSR_bandpass_output", "atlases",
                     "atlas-4S456Parcels", "atlas-4S456Parcels_dseg.tsv")
PAIRS = os.path.join(HERE, "results", "rebuild", "pairwise_metrics.csv")
DEMO = os.path.join(BRAIN, "results", "wj", "manuscript_supplements", "per_subject_demographics_wj.csv")
OUT = os.path.join(HERE, "results", "per_subject"); os.makedirs(OUT, exist_ok=True)
CONDITIONS = ["awake", "unconscious", "recovery"]

def tri(m): return m[np.triu_indices(m.shape[0], k=1)]
def mantel(A, B, rng):
    a, b = tri(A), tri(B)
    if np.std(a) == 0 or np.std(b) == 0: return np.nan, np.nan
    r = pearsonr(a, b)[0]; n = A.shape[0]; c = 0
    for _ in range(NPERM):
        p = rng.permutation(n)
        if abs(pearsonr(tri(A[np.ix_(p, p)]), b)[0]) >= abs(r): c += 1
    return float(r), float((c + 1) / (NPERM + 1))

def build_rmap():
    atlas = pd.read_csv(ATLAS, sep="\t"); pl = atlas["label"].tolist()
    rmap = gtb.build_region_parcel_map(pl)
    an = atlas["atlas_name"].astype(str).tolist() if "atlas_name" in atlas.columns else ["4S456"] * len(pl)
    def pick(keys):
        return [i for i, l in enumerate(pl) if any(k in l for k in keys) and an[i] == "4S456"]
    rmap["Brain - Frontal Cortex (BA9)"] = pick(["Cont_PFCl", "Cont_PFCmp", "Default_PFC", "Cont_pCun"])
    rmap["Brain - Anterior cingulate cortex (BA24)"] = pick(["SalVentAttn_Med", "Limbic_OFC", "Default_pCunPCC", "Cont_Cing"])
    rmap["Brain - Cortex"] = [i for i, l in enumerate(pl) if an[i] == "4S456"]
    return {k: v for k, v in rmap.items() if v}

def region_fc(parcel_corr, rmap, regions):
    N = len(regions); M = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            p1 = [p for p in rmap[regions[i]] if p < parcel_corr.shape[0]]
            p2 = [p for p in rmap[regions[j]] if p < parcel_corr.shape[0]]
            M[i, j] = M[j, i] = float(np.mean(parcel_corr[np.ix_(p1, p2)])) if p1 and p2 else np.nan
    return M

def main():
    df = pd.read_csv(PAIRS)
    rmap = build_rmap()
    regions = [r for r in sorted(set(df["Region1"]) | set(df["Region2"])) if r in rmap]
    idx = {r: i for i, r in enumerate(regions)}; N = len(regions)
    print(f"regions used (in both WJ and parcel map): {N}")
    WJ = np.eye(N)
    for _, row in df.iterrows():
        if row["Region1"] in idx and row["Region2"] in idx:
            i, j = idx[row["Region1"]], idx[row["Region2"]]; WJ[i, j] = WJ[j, i] = row["wj_unsigned"]

    subs = sorted({re.match(r"(sub-\d+)_awake", os.path.basename(p)).group(1)
                   for p in glob.glob(os.path.join(CM, "sub-*_awake_spearman_corr.npy"))})
    rng = np.random.RandomState(SEED); rows = []
    for s in subs:
        rec = {"subject": s}
        ok = True
        for cond in CONDITIONS:
            fp = os.path.join(CM, f"{s}_{cond}_spearman_corr.npy")
            if not os.path.exists(fp): ok = False; break
            fc = region_fc(np.load(fp), rmap, regions)
            r, p = mantel(WJ, fc, rng)
            rec[f"mantel_r_{cond}"] = r; rec[f"mantel_p_{cond}"] = p
        if ok: rows.append(rec)
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, "per_subject_mantel.csv"), index=False)
    print(f"\nsubjects with all 3 conditions: {len(res)}")

    summary = {"n_subjects": int(len(res)), "regions": regions, "n_perm": NPERM, "seed": SEED}
    for cond in CONDITIONS:
        vals = res[f"mantel_r_{cond}"].dropna().values
        w_p = float(wilcoxon(vals)[1]) if len(vals) > 5 else None
        summary[cond] = {"mean_r": round(float(np.mean(vals)), 4), "median_r": round(float(np.median(vals)), 4),
                         "sd": round(float(np.std(vals)), 4), "frac_positive": round(float(np.mean(vals > 0)), 3),
                         "wilcoxon_p_vs_0": round(w_p, 5) if w_p is not None else None}
        print(f"  {cond:12s} mean r={summary[cond]['mean_r']:+.3f}  frac_pos={summary[cond]['frac_positive']}  W p={summary[cond]['wilcoxon_p_vs_0']}")

    # dose / demographic relationship, if available
    summary["dose_relationship"] = {}
    if os.path.exists(DEMO):
        demo = pd.read_csv(DEMO)
        subcol = next((c for c in demo.columns if demo[c].astype(str).str.contains("sub-").any()), None)
        if subcol:
            demo["subject"] = demo[subcol].astype(str).str.extract(r"(sub-\d+)")[0]
            m = res.merge(demo, on="subject", how="inner")
            numcols = [c for c in demo.columns if pd.api.types.is_numeric_dtype(demo[c]) and demo[c].notna().sum() > 5]
            for cond in CONDITIONS:
                for nc in numcols:
                    sub = m[[f"mantel_r_{cond}", nc]].dropna()
                    if len(sub) > 5 and sub[nc].std() > 0:
                        rho, p = spearmanr(sub[f"mantel_r_{cond}"], sub[nc])
                        if p < 0.10:
                            summary["dose_relationship"][f"{cond}__{nc}"] = {"spearman": round(float(rho), 3), "p": round(float(p), 4), "n": int(len(sub))}
            print("  demographic columns available:", numcols)
    json.dump(summary, open(os.path.join(OUT, "per_subject_summary.json"), "w"), indent=2)
    print(f"\n-> {OUT}")

if __name__ == "__main__":
    main()
