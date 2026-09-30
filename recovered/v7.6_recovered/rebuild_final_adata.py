"""
Rebuild experiments/exp-04-03-2026/stage6_subclustering/final_adata.h5ad from the raw
segmentation feature CSVs, WITHOUT any clustering (no PCA / neighbours / Leiden).

The original final_adata.h5ad was lost with the /home/hp workstation. This reproduces exactly
what the pipeline put into it:
  X   = double_z_norm(25 selected markers from config.yaml), computed PER MACHINE
        (03_initial_clustering, one run per machine), then the two machines concatenated
        (06_subclustering, ad.concat join="outer").
  obs = Cell Id, Nuc X, Nuc Y Inv, region_num, unique_region, machine, 63 unselected raw features,
        + cluster labels copied by (unique_region, Cell Id) from the saved v2 ROI CSVs.

Labels are never recomputed: they are looked up from the saved CSVs.

Also stages the other two inputs build_v7.6.py reads, so MRD_PROJ is a complete stand-in for
/home/hp/projects/multiplex_imaging_analysis:
  config_18.06.26.yaml and results_soumya/Nandhu_final_v7.5/csv/cell_assignments.csv

Paths (override with environment variables):
  MRD_RAW   raw per-ROI feature CSVs, with "Machine A" / "Machine B" subfolders
  MRD_REC   recovered results folder ("new 18.06.26")
  MRD_PROJ  output project root that the v7.6 scripts then read from
"""
import os, sys, glob, shutil
from pathlib import Path
import pandas as pd, anndata as ad, yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline.io import combine_feature_files, discover_feature_files
from pipeline.preprocessing import clean_column_names, select_features, build_adata

RAW  = os.environ.get("MRD_RAW",  "/mnt/e/MRD analysis_Feb 2026/CSVs for initial clustering")
REC  = os.environ.get("MRD_REC",  "/mnt/e/MRD analysis Apr_2026/30 May 2026_Claude new results/new 18.06.26")
PROJ = os.environ.get("MRD_PROJ", f"{REC}/proj_recovered")
OUT  = f"{PROJ}/experiments/exp-04-03-2026/stage6_subclustering/final_adata.h5ad"
MARKERS = yaml.safe_load(open(REPO / "config.yaml"))["selected_markers"]
META = ["Cell Id", "Nuc X", "Nuc Y Inv", "region_num", "unique_region"]
assert len(MARKERS) == 25, "final_adata.X used the 25 markers in the repo config.yaml"

parts = []
for machine, sub in [("machine_A", "Machine A"), ("machine_B", "Machine B")]:
    # unique_region was "Machine A/Slides1_ROI_01" -> the walked root was the parent of "Machine A"
    files, _ = discover_feature_files(f"{RAW}/{sub}")
    files   = [os.path.join(sub, os.path.basename(f)) for f in files]
    regions = [os.path.join(sub, os.path.splitext(os.path.basename(f))[0]) for f in files]
    df = clean_column_names(combine_feature_files(RAW, files, regions))
    sel, unsel, meta = select_features(df, MARKERS, META)
    a = build_adata(sel, meta, MARKERS)
    for c in unsel.columns:
        a.obs[c] = unsel[c].values
    a.obs["machine"] = machine
    print(f"{machine}: {a.n_obs} cells, {len(files)} ROIs, X={a.n_vars}, unselected={unsel.shape[1]}")
    parts.append(a)

adata = ad.concat(parts, join="outer")
adata.obs_names_make_unique()
assert adata.n_obs == 233949, f"expected 233,949 cells, got {adata.n_obs}"

# ---- attach saved labels (lookup only) ----
lab = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{REC}/Nandhu_final_v2/roi_assignments_final/*.csv"))])
lab = lab.rename(columns={"initial_cluster": "leiden", "pooled_group": "cluster_group"})
LCOLS = ["leiden", "cluster_label", "cluster_group", "cluster_labelled", "sub_cluster", "final_label", "final_label_integer"]
lab["Cell Id"] = lab["Cell Id"].astype(str)
lab = lab.set_index(["unique_region", "Cell Id"])[LCOLS]
key = pd.MultiIndex.from_arrays([adata.obs["unique_region"].values, adata.obs["Cell Id"].astype(str).values])
for c in LCOLS:
    adata.obs[c] = lab[c].reindex(key).values
print(f"labels attached for {adata.obs['final_label'].notna().sum()} / {adata.n_obs} cells (v2 ROI CSVs cover 205,681)")
for c in ["cluster_label", "cluster_group", "cluster_labelled", "final_label", "leiden", "sub_cluster"]:
    adata.obs[c] = adata.obs[c].astype(str)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
adata.write_h5ad(OUT)
print("wrote", OUT, adata.shape)

# ---- stage the other v7.6 inputs ----
shutil.copy(f"{REC}/config_18.06.26.yaml", PROJ)
os.makedirs(f"{PROJ}/results_soumya/Nandhu_final_v7.5/csv", exist_ok=True)
shutil.copy(f"{REC}/Nandhu_final_v7.5/csv/cell_assignments.csv", f"{PROJ}/results_soumya/Nandhu_final_v7.5/csv/")
print("staged config_18.06.26.yaml and Nandhu_final_v7.5/csv/cell_assignments.csv under", PROJ)
