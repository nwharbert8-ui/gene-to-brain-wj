# Gene-to-Brain Cross-Scale WJ Analysis

**Gene Co-expression Architecture Predicts Cross-Regional Brain Functional Connectivity Independent of Spatial Proximity**

Drake H. Harbert | Inner Architecture LLC | ORCID: 0009-0007-7740-3616

## Key Finding

Gene co-expression **architecture** similarity (weighted Jaccard index on full-transcriptome pairwise correlation matrices) predicts brain functional connectivity where expression **profile** similarity does not.

| Measure | rho | p |
|---------|-----|---|
| Co-expression architecture (WJ) | 0.464 | <0.001 |
| Expression profiles | 0.091 | 0.509 |
| Steiger's Z (WJ > Expression) | 5.11 | <0.001 |

The finding survives:
- Cortical-subcortical pairs after distance control (rho=0.635, p=0.002)
- Propofol-induced unconsciousness after distance control (rho=0.297, p=0.027)
- 10/10 split-half donor replications (mean rho=0.450)
- Leave-one-region-out jackknife (all 11 significant)
- Pearson vs Spearman comparison (agreement rho=0.986)

## Data Sources

- **Gene expression:** GTEx v8 individual-level TPM (16,273 genes, 2,268 brain samples, 11 regions)
- **Functional connectivity:** OpenNeuro ds006623 (Michigan Human Anesthesia fMRI, n=24, 4S456 parcellation)
- **Protein interactions:** STRING v12 (human, 4.8M scored interactions)

## Pipeline Files

| File | Description |
|------|-------------|
| `gene_to_brain_relational.py` | Primary analysis: per-region co-expression matrices, WJ between regions, comparison to fMRI FC |
| `gene_to_brain_v2_quick.py` | Expression profile (component-level) analysis with proper atlas mapping |
| `gene_to_brain_bulletproof.py` | Full validation suite: 8 tests including distance control, Steiger's Z, split-half, propofol |
| `spatial_deep_dive.py` | Detailed spatial proximity analysis: cross-type pairs, propofol modulation, residuals |
| `synapse_filter_fix.py` | Synaptic gene subset comparison (69 genes vs full transcriptome) |

## Reproduction

```bash
# Requirements
pip install numpy scipy pandas matplotlib seaborn python-docx

# Data (auto-downloaded by pipelines)
# GTEx v8: https://gtexportal.org (individual-level TPM + sample attributes)
# OpenNeuro ds006623: https://openneuro.org/datasets/ds006623
# STRING v12: https://string-db.org

# Run primary analysis
python gene_to_brain_relational.py

# Run validation
python gene_to_brain_bulletproof.py
```

All pipelines use `RANDOM_SEED = 42` and save results to Google Drive paths (configurable in each script's CONFIG section).

## Methodology

Weighted Jaccard (WJ) measures the similarity of two correlation matrices:

```
WJ(A, B) = Σ min(|r_ij^A|, |r_ij^B|) / Σ max(|r_ij^A|, |r_ij^B|)
```

Applied here: for each brain region, compute gene-gene Spearman correlation across GTEx donors. Compare architectures between region pairs using WJ. Correlate with fMRI functional connectivity.

## License

MIT

## Citation

Harbert, D.H. (2026). Gene co-expression architecture predicts cross-regional brain functional connectivity independent of spatial proximity. *NeuroImage* (in preparation).
