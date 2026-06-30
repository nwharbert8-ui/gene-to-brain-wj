# -*- coding: utf-8 -*-
"""
Pipeline: Gene-to-Brain head-motion confound check
Author: Drake H. Harbert (D.H.H.) | ORCID 0009-0007-7740-3616
Affiliation: Inner Architecture LLC, Canton, OH | Date: 2026-06-30
Description:
    Verifies that head motion (framewise displacement) does not differ between the
    awake and unconscious conditions whose group FC matrices drive the WJ-FC result,
    so a reviewer cannot attribute the connectivity difference to motion. XCP-D
    already regressed motion; this is the explicit per-condition verification.
    Condition->run map (from brain_connectivity_wj_pipeline.py):
        awake       = task-rest_run-1
        unconscious = task-imagery_run-2
    Per subject, mean framewise displacement (FD) is taken from each run's XCP-D
    motion.tsv; a paired Wilcoxon tests awake vs unconscious across subjects.
Dependencies: numpy, pandas, scipy
Output: results/headmotion_check/
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, ttest_rel
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

XCP = r"G:/My Drive/inner_architecture_research/brain_connectivity_wj/data/raw/ds006623/derivatives/xcp_d_without_GSR_bandpass_output"
OUT = r"G:/My Drive/inner_architecture_research/gene_to_brain_wj/results/headmotion_check"
os.makedirs(OUT, exist_ok=True)
AWAKE_RUN = "task-rest_run-1"; UNCON_RUN = "task-imagery_run-2"

def fd_mean(path):
    df = pd.read_csv(path, sep="\t")
    col = next((c for c in df.columns if c.lower() in
                ("framewise_displacement", "fd", "framewisedisplacement")), None)
    if col is None:
        col = next((c for c in df.columns if "framewise" in c.lower() or c.lower() == "fd"), None)
    if col is None:
        return None, None
    fd = pd.to_numeric(df[col], errors="coerce").dropna().values
    return float(np.mean(fd)), float(np.mean(fd > 0.5))  # mean FD, fraction high-motion vols

def main():
    subs = sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                   for p in glob.glob(os.path.join(XCP, "sub-*", "func", "*_motion.tsv"))})
    rows = []
    for s in subs:
        aw = os.path.join(XCP, s, "func", f"{s}_{AWAKE_RUN}_motion.tsv")
        un = os.path.join(XCP, s, "func", f"{s}_{UNCON_RUN}_motion.tsv")
        if not (os.path.exists(aw) and os.path.exists(un)):
            continue
        am, ah = fd_mean(aw); um, uh = fd_mean(un)
        if am is None or um is None:
            continue
        rows.append({"subject": s, "fd_awake": am, "fd_unconscious": um,
                     "hi_frac_awake": ah, "hi_frac_unconscious": uh})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "per_subject_fd.csv"), index=False)
    a = df["fd_awake"].values; u = df["fd_unconscious"].values
    w_stat, w_p = wilcoxon(a, u)
    t_stat, t_p = ttest_rel(a, u)
    res = {
        "n_subjects": int(len(df)),
        "condition_map": {"awake": AWAKE_RUN, "unconscious": UNCON_RUN},
        "mean_FD_awake": round(float(a.mean()), 4),
        "mean_FD_unconscious": round(float(u.mean()), 4),
        "median_FD_awake": round(float(np.median(a)), 4),
        "median_FD_unconscious": round(float(np.median(u)), 4),
        "wilcoxon_p": round(float(w_p), 4),
        "paired_t_p": round(float(t_p), 4),
        "verdict": ("FD does NOT differ between conditions (motion not a confound)"
                    if w_p > 0.05 else
                    "FD DIFFERS between conditions (motion is a potential confound; report and consider FD covariate)"),
        "design_note": "awake=task-rest, unconscious=task-imagery: the contrast mixes drug state with task; disclose in Limitations.",
        "execution_date": "2026-06-30",
    }
    json.dump(res, open(os.path.join(OUT, "headmotion_check.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
