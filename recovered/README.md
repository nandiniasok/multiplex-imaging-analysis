# Recovered analyses: v7.6 and v2

After the August 2026 crash, the analysis workstation (`/home/hp/projects/multiplex_imaging_analysis`)
and two of its files could not be recovered:

- `experiments/exp-04-03-2026/stage6_subclustering/final_adata.h5ad`
- `config_18.06.26.yaml` (only the reconstructed copy exists)

This folder makes v7.6 runnable again **without re-running any clustering or sub-clustering**.
The cluster labels were annotated by hand and are only ever read from the saved CSVs.

## v7.6_recovered/

| File | What it is |
|---|---|
| `rebuild_final_adata.py` | **New.** Rebuilds `final_adata.h5ad` from the raw segmentation CSVs. No PCA, neighbours or Leiden. |
| `build_v7.6.py`, `Nandhu_final_v7.6.ipynb` | Original v7.6 code, with the same code in both. The only change is that `PROJ` can be set with `MRD_PROJ`. |
| `make_spatial_overlays.py`, `make_summary_pdf.py` | Original code. The same `MRD_PROJ` change is the only edit. |
| `verify_v7.6.py` | **New.** Compares a rerun against the original v7.6 results saved on 2026-08-09. |

### What final_adata.h5ad contained

This was confirmed against the saved v2 ROI CSVs:

- **X**: `double_z_norm` of the **25 markers in the repo `config.yaml`**. These are not the 24 in
  `config_18.06.26.yaml`. The normalisation was computed **separately for each machine**
  (`03_initial_clustering`), then the two machines were concatenated (`06_subclustering`).
- **obs**: metadata, the 63 unselected raw features, and labels. There are 233,949 cells:
  155,883 from Machine A and 78,066 from Machine B.

### Run

```bash
# Optional; these are the defaults:
export MRD_RAW="/mnt/e/MRD analysis_Feb 2026/CSVs for initial clustering"
export MRD_REC="/mnt/e/MRD analysis Apr_2026/30 May 2026_Claude new results/new 18.06.26"
export MRD_PROJ="$MRD_REC/proj_recovered"   # results are written under here

cd recovered/v7.6_recovered
python rebuild_final_adata.py   # final_adata.h5ad + stages the config and the v7.5 CSV
python build_v7.6.py            # or run Nandhu_final_v7.6.ipynb
python make_spatial_overlays.py
python make_summary_pdf.py
python verify_v7.6.py
```

Tested environment: Python 3.12, scanpy 1.12.4, anndata 0.13.4, scikit-learn 1.9.1, numpy 2.5.3,
**pandas 2.3.3**. Use pandas 2.x, because these scripts were written before pandas 3.

### Verification (2026-09-28, repeated 2026-09-30 with the code in this folder)

- The raw CSVs matched the values in the v2 ROI CSVs for all 205,681 cells and 66 columns.
  The largest difference was 1e-13, which is CSV rounding.
- `v76_label`, `v76_coarse` and `v76_label_int` were **identical for 233,949 of 233,949 cells**.
- All 49 `roi_assignments_final` files and `recovery_counts.csv` were identical.
- `final_label_integer_key.csv` and `recovery_quantification.csv` were identical in content.
- The heatmap matrices differed by at most 9e-16.

Note: the saved originals of `final_label_integer_key.csv` and `recovery_quantification.csv` were
later re-saved by Excel. They are `;`-separated, and in `recovery_quantification.csv` the decimal
point is lost (30.5 became 305). Use the regenerated copies.

## v2_recovered/

This is the original v2 code (`generate_v2.py`, `generate_nandhu_final_v2.py`) plus its small outputs.
Its large per-ROI output (`roi_assignments_final/`) is kept on the E: drive, not in git.
The only change is that the results root can be set with `MRD_PROJ`.

**v2 has not been rerun.** It reads `results_soumya/all rois all markers final/`
(`csv/*_best_means.csv` and `roi_assignments/*_converted.csv`), which was also lost.
v7.6 does not need v2 to be rerun: it uses v2 only through the `v2_final_label_integer` column
in v7.5's `cell_assignments.csv`, and that column was verified identical.
