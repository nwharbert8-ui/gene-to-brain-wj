# -*- coding: utf-8 -*-
"""
Pipeline: Layer 2H pairings on the per-subject, three-state correspondence
Author: Drake H. Harbert (D.H.H.) | ORCID 0009-0007-7740-3616
Affiliation: Inner Architecture LLC, Canton, OH | Date: 2026-06-30
Description:
    Two principled pairings on the gene-to-FC correspondence, using all subjects and
    all three states.
    (1) State-treatment pairing (within-subject): paired gaps between awake,
        unconscious, recovery correspondence per subject, with paired Wilcoxon and
        bootstrap CIs; the gap localizes how brain state modulates the molecular
        constraint, and whether recovery returns to baseline.
    (2) Local-global pairing (Type 5): per-subject correspondence vs the single
        group-matrix correspondence (group FC for each of the three states added,
        completing the group trajectory).
Dependencies: numpy, pandas, scipy ; imports gtb_rebuild_from_scratch
Output: results/state_pairings/
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr, wilcoxon
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gtb_rebuild_from_scratch as gtb

SEED = 42; NPERM = 10000; NBOOT = 1000; np.random.seed(SEED)
BRAIN = r"G:/My Drive/inner_architecture_research/brain_connectivity_wj"
CM = os.path.join(BRAIN, "results", "correlation_matrices")
ATLAS = os.path.join(BRAIN, "data", "raw", "ds006623", "derivatives",
                     "xcp_d_without_GSR_bandpass_output", "atlases",
                     "atlas-4S456Parcels", "atlas-4S456Parcels_dseg.tsv")
PAIRS = os.path.join(HERE, "results", "rebuild", "pairwise_metrics.csv")
PSUB = os.path.join(HERE, "results", "per_subject", "per_subject_mantel.csv")
OUT = os.path.join(HERE, "results", "state_pairings"); os.makedirs(OUT, exist_ok=True)
CONDS = ["awake", "unconscious", "recovery"]

def tri(m): return m[np.triu_indices(m.shape[0], k=1)]
def mantel(A, B, rng):
    a, b = tri(A), tri(B); r = pearsonr(a, b)[0]; n = A.shape[0]; c = 0
    for _ in range(NPERM):
        p = rng.permutation(n)
        if abs(pearsonr(tri(A[np.ix_(p, p)]), b)[0]) >= abs(r): c += 1
    return float(r), float((c + 1) / (NPERM + 1))
def boot_ci(x, rng):
    x = np.asarray(x); bs = [np.mean(x[rng.randint(0, len(x), len(x))]) for _ in range(NBOOT)]
    return round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)
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
    rng = np.random.RandomState(SEED)
    df = pd.read_csv(PAIRS); rmap = build_rmap()
    regions = [r for r in sorted(set(df["Region1"]) | set(df["Region2"])) if r in rmap]
    idx = {r: i for i, r in enumerate(regions)}; N = len(regions)
    WJ = np.eye(N)
    for _, row in df.iterrows():
        if row["Region1"] in idx and row["Region2"] in idx:
            i, j = idx[row["Region1"]], idx[row["Region2"]]; WJ[i, j] = WJ[j, i] = row["wj_unsigned"]

    out = {"regions": regions, "n_perm": NPERM}

    # ---- (2) GROUP trajectory (all three states) + local-global gap ----
    group = {}
    for cond in CONDS:
        gp = os.path.join(CM, f"group_{cond}_spearman_corr.npy")
        r, p = mantel(WJ, region_fc(np.load(gp), rmap, regions), rng)
        group[cond] = {"mantel_r": round(r, 4), "p": round(p, 4)}
    out["group_trajectory"] = group

    ps = pd.read_csv(PSUB)
    out["per_subject_vs_group"] = {}
    for cond in CONDS:
        sub = ps[f"mantel_r_{cond}"].dropna().values
        out["per_subject_vs_group"][cond] = {
            "per_subject_mean": round(float(np.mean(sub)), 4),
            "per_subject_ci": boot_ci(sub, rng),
            "group_single_matrix": group[cond]["mantel_r"],
            "group_p": group[cond]["p"],
            "note": "group p is single-matrix Mantel (weak inference); per-subject CI is over independent subjects"}

    # ---- (1) STATE-TREATMENT pairing: within-subject paired gaps ----
    a, u, r_ = ps["mantel_r_awake"].values, ps["mantel_r_unconscious"].values, ps["mantel_r_recovery"].values
    def paired(x, y, name):
        d = x - y; w = wilcoxon(x, y)
        return {"gap_mean": round(float(np.mean(d)), 4), "gap_ci": boot_ci(d, rng),
                "wilcoxon_p": round(float(w[1]), 5), "frac_x_gt_y": round(float(np.mean(x > y)), 3), "n": int(len(d))}
    out["state_treatment_pairing"] = {
        "unconscious_minus_awake": paired(u, a, "u-a"),
        "unconscious_minus_recovery": paired(u, r_, "u-r"),
        "recovery_minus_awake": paired(r_, a, "r-a"),
    }
    # trajectory shape: is unconscious the within-subject peak; does recovery return toward awake
    out["trajectory_shape"] = {
        "frac_unconscious_is_max": round(float(np.mean((u >= a) & (u >= r_))), 3),
        "frac_recovery_below_unconscious": round(float(np.mean(r_ < u)), 3),
        "recovery_vs_awake_ns": "see recovery_minus_awake wilcoxon_p (large p = returns to baseline)",
    }
    json.dump(out, open(os.path.join(OUT, "state_pairings.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
