# -*- coding: utf-8 -*-
"""
Pipeline: Per-region (Layer 2F) gene-to-FC correspondence across the three states
Author: Drake H. Harbert (D.H.H.) | ORCID 0009-0007-7740-3616
Affiliation: Inner Architecture LLC, Canton, OH | Date: 2026-06-30
Description:
    For each brain region, computes how well that region's gene co-expression
    coupling profile (its WJ row to the other 10 regions) corresponds to that
    region's functional-connectivity profile, per subject and per state. Aggregates
    across the 24 subjects. Reports, per region: mean row-correspondence in each
    state, and the within-subject paired state gaps (unconscious vs awake, etc.) so
    we can see which regions carry the overall correspondence and which drive the
    state-dependent elevation. Layer 2F decomposition crossed with the state axis.
Dependencies: numpy, pandas, scipy ; imports gtb_rebuild_from_scratch
Output: results/per_region_state/
"""
import os, sys, glob, re, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr, wilcoxon
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gtb_rebuild_from_scratch as gtb
SEED = 42; np.random.seed(SEED)
BRAIN = r"G:/My Drive/inner_architecture_research/brain_connectivity_wj"
CM = os.path.join(BRAIN, "results", "correlation_matrices")
ATLAS = os.path.join(BRAIN, "data", "raw", "ds006623", "derivatives",
                     "xcp_d_without_GSR_bandpass_output", "atlases",
                     "atlas-4S456Parcels", "atlas-4S456Parcels_dseg.tsv")
PAIRS = os.path.join(HERE, "results", "rebuild", "pairwise_metrics.csv")
OUT = os.path.join(HERE, "results", "per_region_state"); os.makedirs(OUT, exist_ok=True)
CONDS = ["awake", "unconscious", "recovery"]

def build_rmap():
    atlas = pd.read_csv(ATLAS, sep="\t"); pl = atlas["label"].tolist()
    an = atlas["atlas_name"].astype(str).tolist() if "atlas_name" in atlas.columns else ["4S456"] * len(pl)
    rmap = gtb.build_region_parcel_map(pl)
    pick = lambda ks: [i for i, l in enumerate(pl) if any(k in l for k in ks) and an[i] == "4S456"]
    rmap["Brain - Frontal Cortex (BA9)"] = pick(["Cont_PFCl", "Cont_PFCmp", "Default_PFC", "Cont_pCun"])
    rmap["Brain - Anterior cingulate cortex (BA24)"] = pick(["SalVentAttn_Med", "Limbic_OFC", "Default_pCunPCC", "Cont_Cing"])
    rmap["Brain - Cortex"] = [i for i, l in enumerate(pl) if an[i] == "4S456"]
    return {k: v for k, v in rmap.items() if v}
def region_fc(pc, rmap, regions):
    N = len(regions); M = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            p1 = [p for p in rmap[regions[i]] if p < pc.shape[0]]; p2 = [p for p in rmap[regions[j]] if p < pc.shape[0]]
            M[i, j] = M[j, i] = float(np.mean(pc[np.ix_(p1, p2)])) if p1 and p2 else np.nan
    return M

def main():
    df = pd.read_csv(PAIRS); rmap = build_rmap()
    regions = [r for r in sorted(set(df["Region1"]) | set(df["Region2"])) if r in rmap]
    idx = {r: i for i, r in enumerate(regions)}; N = len(regions)
    WJ = np.eye(N)
    for _, row in df.iterrows():
        if row["Region1"] in idx and row["Region2"] in idx:
            i, j = idx[row["Region1"]], idx[row["Region2"]]; WJ[i, j] = WJ[j, i] = row["wj_unsigned"]
    subs = sorted({re.match(r"(sub-\d+)_awake", os.path.basename(p)).group(1)
                   for p in glob.glob(os.path.join(CM, "sub-*_awake_spearman_corr.npy"))})
    # per-region row correspondence: rowcorr[cond][region] = list over subjects
    store = {c: {r: [] for r in regions} for c in CONDS}
    used = 0
    for s in subs:
        fcs = {}
        ok = True
        for c in CONDS:
            fp = os.path.join(CM, f"{s}_{c}_spearman_corr.npy")
            if not os.path.exists(fp): ok = False; break
            fcs[c] = region_fc(np.load(fp), rmap, regions)
        if not ok: continue
        used += 1
        for c in CONDS:
            FC = fcs[c]
            for ri, r in enumerate(regions):
                others = [k for k in range(N) if k != ri]
                wjrow = WJ[ri, others]; fcrow = FC[ri, others]
                if np.std(wjrow) > 0 and np.std(fcrow) > 0:
                    store[c][r].append(pearsonr(wjrow, fcrow)[0])
    rows = []
    for r in regions:
        a = np.array(store["awake"][r]); u = np.array(store["unconscious"][r]); rec = np.array(store["recovery"][r])
        n = min(len(a), len(u), len(rec))
        a, u, rec = a[:n], u[:n], rec[:n]
        ua_p = float(wilcoxon(u, a)[1]) if n > 5 else np.nan
        rows.append({"region": r,
                     "rowcorr_awake": round(float(np.mean(a)), 4),
                     "rowcorr_unconscious": round(float(np.mean(u)), 4),
                     "rowcorr_recovery": round(float(np.mean(rec)), 4),
                     "unc_minus_awake": round(float(np.mean(u - a)), 4),
                     "unc_minus_awake_p": round(ua_p, 4),
                     "mean_across_states": round(float(np.mean([np.mean(a), np.mean(u), np.mean(rec)])), 4)})
    res = pd.DataFrame(rows).sort_values("mean_across_states", ascending=False)
    res.to_csv(os.path.join(OUT, "per_region_state.csv"), index=False)
    json.dump({"n_subjects": used, "regions": regions, "table": rows,
               "seed": SEED, "execution_date": "2026-06-30"},
              open(os.path.join(OUT, "per_region_state.json"), "w"), indent=2)
    print(f"subjects used: {used}\n")
    print(res.to_string(index=False))
    print("\nTOP carriers (mean across states):", ", ".join(res.head(4)["region"].str.replace("Brain - ", "")))
    drv = res.sort_values("unc_minus_awake", ascending=False)
    print("TOP state-effect drivers (unc-awake):",
          ", ".join((drv.head(4)["region"].str.replace("Brain - ", "") + f" ({drv.head(4)['unc_minus_awake'].iloc[0]:+.2f}...)" if False else drv.head(4)["region"].str.replace("Brain - ", ""))))

if __name__ == "__main__":
    main()
