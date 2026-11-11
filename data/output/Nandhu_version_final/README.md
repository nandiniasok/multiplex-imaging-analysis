# Nandhu_version_final (built 2026-11-11 from Nandhu_final_v7.6)

No clustering or sub-clustering was run. The v7.6 labels are read from
`Nandhu_final_v7.6/csv/cell_assignments.csv`, and only the changes below are applied.
`build_Nandhu_version_final.py` and `Nandhu_version_final.ipynb` contain the same code.
The notebook was executed and its outputs are saved.

## 1. Dendritic-cell naming
The 2,116 v2 "Dendritic cell" cells (v2 label 14) are found in v7.6 as follows:
- 1,860 are in "HLA-DR-hi APC (CD1c+)". v7.5 created that name by renaming "Immature myeloid CD11c+".
- 212 are in Erythroid and 44 are in MSC. These were left unchanged.

v7.5 renamed two clusters, and both renames are now reversed. No cells move:

| v7.6 label | final label | cells |
|---|---|---|
| HLA-DR-hi APC (CD1c+) | **Immature myeloid CD11c+** | 1,975 (1,978 minus 3 v2 tumour cells) |
| Dendritic cells (CD11c+HLA-DR+) | **HLA-DR-hi APC (CD1c+)** | 6,770 (6,782 minus 12 v2 tumour cells) |

Marker check (mean z):
- The new "HLA-DR-hi APC (CD1c+)" cluster has CD11c 1.75, HLA-DR 1.68 and **CD1c −0.34**, so it is CD1c-negative.
- "Immature myeloid CD11c+" has CD11c 0.62, HLA-DR 2.12 and CD1c 0.82.

## 2. Tumour cells moved back into NPM1_blasts (16,222 → 43,493)
| source | cells |
|---|---|
| (a) all v2 Tumor + Tumor/Erythroid cells (v2 labels 18, 19, 20, 22, 23), whatever their v7.6 label | 33,694 (27,271 were outside NPM1_blasts) |
| (b) cells in the two Unassigned-Leiden clusters that are NPM1⁺ CD43⁺ CD147⁺ (z>0.5) | 1 |
| (c) Unassigned/Other or Artifact cells that are NPM1_mut⁺ (per-machine z>0.5). This is the same rule `Spatial_analysis_Aug 9_New` used. | 5,814 |

The cause of the June → August loss: v2 "Proliferating leukemic erythroid" (v2 label 23) became
**Mixed progenitors** in v7.3–v7.6. 13,092 of the 13,290 Mixed progenitors were v2 tumour cells.
Using v7.6 NPM1_blasts alone as the tumour set therefore missed them, and also missed the
v2 tumour cells that had moved to Erythroid (2,935), MSC (1,297), Unassigned (2,429), Mast cells (637) and others.
See `csv/tumour_recovery_by_source.csv`.

## 3. Everything else is unchanged
Apart from the tumour cells above, no cell changed label. `csv/cluster_size_v76_vs_final.csv` shows how
many cells each cluster kept. Mixed progenitors drops to 198 cells, Erythroid keeps 88.1% and MSC keeps 85.8%.
Every other named cluster keeps at least 95.7%.

## Spatial check (`spatial_check/`)
The Aug 9_New method (`spatial_analysis_v76_recovered_tumour.py`) was rerun unchanged except that
tumour = the final NPM1_blasts. It ran on 49/49 ROIs.
- **Proximity composition:** Endothelial is higher in non-relapse at <15, 15–30, 30–60 and 60–100 µm (BH<0.05).
  Macrophages are higher in relapse at 100+ µm (BH 0.008), the same as the June result.
- **Immune proliferation at contact:** lower in relapse (BH p = 0.040). With the Aug 9_New tumour set it was not significant (BH 0.12).
