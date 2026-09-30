#!/usr/bin/env python3
"""
Nandhu_final_v2 — pools cells by INITIAL cluster_label (not broad pooled_group).

Root cause of v1 discrepancies:
  3,267 "Immature B cells"  → ended up in Mixed Lymphoid  (not B cells)
  3,803 "Dendritic cells"   → ended up in Mixed Lymphoid  (not Myeloid)
  5,858 "Fibroblasts"       → mixed into MSC pool
  Mast cells                → 0 cells from Machine B
  Monocytes                 → 0 cells from Machine B
  Machine B lymphoid cells  → no "Lymphoid" initial cluster at all

Fix: use cluster_label as the pool, standardising naming across machines.
For each cell, look up its sub_cluster z-scores from the existing best_means.csv,
then apply label rules on those z-scores.

Priority overrides (cross-group, user requirements):
  1. Mast cell  : sub_cluster Tryptase z-score > 1.0 in any myeloid-type group
  2. Macrophage : sub_cluster CD163>0 AND CD68>0    in any myeloid-type group

Confirmed-correct labels kept unchanged:
  CD4+ T helper, Megakaryocyte, Endothelial, Mature B,
  Plasmablast/Plasma cell, Early/Immature B,
  Proliferating AML blast, Quiescent AML blast
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from pathlib import Path
from collections import defaultdict

RESULTS  = Path(os.environ.get("MRD_PROJ", "/home/hp/projects/multiplex_imaging_analysis")) / "results_soumya"
SRC_BASE = RESULTS / "all rois all markers final"
SRC_CSV  = SRC_BASE / "csv"
SRC_ROI  = SRC_BASE / "roi_assignments"
OUT      = RESULTS / "Nandhu_final_v2"

for d in [OUT / "heatmaps_final", OUT / "roi_assignments_final"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Excluded markers (same as v1) ──────────────────────────────────────────────
EXCLUDED = {
    "AKT Pan REA676","c myc REAL810","CD104 REA236","CD117 REA787",
    "CD181 REA958","CD182 REA208","CD183 REAL756","CD196 REA190",
    "CD202b REA198","CD204 REA460","CD23 REAL1096","CD295 REA361",
    "Podoplanin REA446","CD52 REA164","CD2 REA1130","CD247 REAL1135",
    "EmptyPE1","EZH2 REA907","FAK pS910 REA407","IFN gamma REAL788",
    "IRF 7 REA521","JNK2 REA1153","Nucleophosmin REAL823","PKCalpha REA156",
    "RRM2 REAL1010","Synaptophysin REA1121","WT1 REA925","ZAP70 REA814",
}

# ── Cluster_label → new_group mapping ────────────────────────────────────────
POOL_MAP = {
    "Cd4+ T cell":              "CD4 T",
    "CD4+ T cells":             "CD4 T",
    "CD8 T cell":               "CD8 T",
    "CD8+ T cells":             "CD8 T",
    "Mature B cells":           "Mature B",
    "Immature B cells":         "Immature B",
    "Early B":                  "Immature B",
    "Plasma cells":             "Plasma cell",
    "Granluocyte":              "Granulocyte",
    "Granulocyte":              "Granulocyte",
    "Granulocytes":             "Granulocyte",
    "Granulocytes/Mast cells":  "Granulocyte",
    "Macrophage":               "Macrophage",
    "Monocytes":                "Monocyte",
    "Monocytes/CD4+ T":         "Monocyte",
    "Mast cells":               "Mast cell",
    "Dendritic cells":          "Dendritic cell",
    "Myeloid":                  "Myeloid misc",
    "Myeloid/Lymphoid":         "Myeloid misc",
    "Tumor":                    "Tumor",
    "NPM1+/Unknown":            "Tumor",
    "NPM1-positive/Erythroid":  "Leukemic erythroid",
    "Erythroid/Tumor":          "Leukemic erythroid",
    "Erythroid":                "Erythroid",
    "MSC":                      "MSC",
    "Fibroblasts":              "MSC",
    "Megakaryocytes":           "Megakaryocyte",
    "Endothelial":              "Endothelial",
    "Lymphoid":                 "Lymphoid",
    "Mixed Lymphoid":           "Lymphoid",
    "Unknown":                  "DROP",
    "Unknown/Tumor":            "DROP",
    "Artifact":                 "DROP",
    "Myeloid/Erythroid":        "DROP",
}

# Pooled_group → best_means filename
PG_FILE = {
    "T cells":"T_cells","B cells":"B_cells","Mixed Lymphoid":"Mixed_Lymphoid",
    "Myeloid":"Myeloid","Tumor":"Tumor","Erythroid":"Erythroid",
    "Tumor/Erythroid":"Tumor-Erythroid","MSC":"MSC",
    "Megakaryocytes":"Megakaryocytes","Endothelial":"Endothelial","Unknown":"Unknown",
}

# Groups eligible for priority overrides
MAST_ELIGIBLE    = {"Granulocyte","Macrophage","Monocyte","Dendritic cell",
                    "Mast cell","Lymphoid","Myeloid misc"}
MACRO_ELIGIBLE   = {"Granulocyte","Macrophage","Monocyte","Lymphoid","Myeloid misc"}
PROTECTED        = {"CD4 T","CD8 T","Mature B","Plasma cell","Immature B",
                    "Tumor","Leukemic erythroid","Erythroid",
                    "Megakaryocyte","Endothelial"}

# ── All final labels (flat list, defines integers) ────────────────────────────
ALL_LABELS = [
    "CD4+ T helper",               # 0
    "CD8+ Cytotoxic T",            # 1
    "Regulatory T (Treg)",         # 2
    "CD4+ PD1+ T",                 # 3
    "Mature B",                    # 4
    "Plasmablast/Plasma cell",     # 5
    "Early/Immature B",            # 6
    "NK cell",                     # 7
    "Memory T lymphoid",           # 8
    "HLA-DR+ lymphoid",            # 9
    "Mixed B/T lymphoid",          # 10
    "Classical monocyte",          # 11
    "Patrolling monocyte",         # 12
    "Macrophage (M2-like)",        # 13
    "Dendritic cell",              # 14
    "Mast cell",                   # 15
    "Neutrophil/Granulocyte",      # 16
    "Proliferating myeloid",       # 17
    "Proliferating AML blast",     # 18
    "Quiescent AML blast",         # 19
    "AML monocytic differentiated",# 20
    "Mature erythroblast",         # 21
    "Leukemic erythroid blast",    # 22
    "Proliferating leukemic erythroid", # 23
    "Bone marrow MSC",             # 24
    "CD271+ MSC",                  # 25
    "Megakaryocyte",               # 26
    "Endothelial",                 # 27
]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(ALL_LABELS)}

GROUP_COLORS = {
    "CD4 T":"#4C72B0","CD8 T":"#1A6EA5","Lymphoid":"#8172B2",
    "Macrophage":"#C44E52","Granulocyte":"#E08080","Monocyte":"#D4614E",
    "Dendritic cell":"#A03030","Mast cell":"#803020",
    "Tumor":"#CCB974","Leukemic erythroid":"#DD8452","Erythroid":"#64B5CD",
    "Mature B":"#55A868","Plasma cell":"#3D9A55","Immature B":"#68C47A",
    "MSC":"#937860","Megakaryocyte":"#DA8BC3","Endothelial":"#8C8C8C",
    "Myeloid misc":"#BB7744",
}


# ── Step 1: Load z-score lookup from best_means.csv ──────────────────────────
print("Step 1: Building z-score lookup from best_means.csv …")

zscore_lookup = {}   # (pooled_group, sub_cluster) -> {marker: z_score}
all_markers_ordered = None

for pg, fname in PG_FILE.items():
    path = SRC_CSV / f"{fname}_best_means.csv"
    if not path.exists():
        continue
    df = pd.read_csv(path)
    marker_cols = [c for c in df.columns if c not in {"group","best_resolution","sub_cluster"}]
    if all_markers_ordered is None:
        all_markers_ordered = marker_cols
    for _, row in df.iterrows():
        sc = int(row["sub_cluster"])
        zscore_lookup[(pg, sc)] = {m: float(row[m]) for m in marker_cols if m in row.index}

remaining_markers = [m for m in all_markers_ordered if m not in EXCLUDED]
print(f"  Lookup entries: {len(zscore_lookup)}")
print(f"  Remaining markers for heatmaps: {len(remaining_markers)}")


# ── Label rule function ───────────────────────────────────────────────────────
def get_label(new_group, m):
    """Apply label rules given new_group and sub_cluster z-score dict m."""

    def v(marker):
        return m.get(marker, 0.0)

    # ── PRIORITY RULES ────────────────────────────────────────────────────────
    if new_group in MAST_ELIGIBLE:
        if v("Mast Cell Tryptase REAL798") > 1.0:
            return "Mast cell"

    if new_group in MACRO_ELIGIBLE:
        if v("CD163 REA1309") > 0 and v("CD68 REAL1346") > 0:
            return "Macrophage (M2-like)"

    # ── GROUP-SPECIFIC RULES ──────────────────────────────────────────────────
    if new_group == "CD4 T":
        # Treg: strictly FoxP3-high
        if v("FoxP3 REA1253") > 0.5 and v("CD3 REAL1097") > -0.25:
            return "Regulatory T (Treg)"
        # CD4+PD1+: PD-1 score must beat CD4 score
        if v("CD279 PD1 3 1 3") > 0.3 and v("CD4 REA1307") > -0.5:
            pd1_s = 4*v("CD279 PD1 3 1 3") + 2*v("CD4 REA1307") + v("CD3 REAL1097")
            cd4_s = 2*v("CD3 REAL1097") + 3*v("CD4 REA1307")
            if pd1_s > cd4_s:
                return "CD4+ PD1+ T"
        if v("CD3 REAL1097") > -0.25 and v("CD4 REA1307") > 0:
            return "CD4+ T helper"
        if v("CD4 REA1307") > 0:
            return "CD4+ T helper"
        return None

    if new_group == "CD8 T":
        if v("CD3 REAL1097") > -0.25 and v("CD8a REA1024") > 0:
            return "CD8+ Cytotoxic T"
        if v("CD8a REA1024") > 0:
            return "CD8+ Cytotoxic T"
        return None

    if new_group == "Mature B":
        return "Mature B" if v("CD20 REAL1069") > 0 else None

    if new_group == "Plasma cell":
        return "Plasmablast/Plasma cell" if v("CD38 REAL719") > 0 else None

    if new_group == "Immature B":
        if v("CD79a REA1168") > 0 or v("CD38 REAL719") > 0:
            return "Early/Immature B"
        return None

    if new_group == "Granulocyte":
        # Mast cell handled above
        return "Neutrophil/Granulocyte"  # all granulocytes that are not mast cells

    if new_group == "Macrophage":
        # Double-positive handled by priority above
        if v("CD163 REA1309") > 0 or v("CD68 REAL1346") > 0:
            return "Macrophage (M2-like)"
        return "Macrophage (M2-like)"  # default: all initial macrophages

    if new_group == "Monocyte":
        # Proliferating myeloid: Ki67-high and CD163+
        if v("Ki 67 REAL1047") > 0.5 and v("CD163 REA1309") > 0:
            return "Proliferating myeloid"
        # Patrolling: CD16+CD11b+ CD14-low
        if v("CD16 REA1324") > 0 and v("CD11b REA1321") > 0 and v("CD14 REA1314") < -0.2:
            return "Patrolling monocyte"
        # Classical: CD14+
        if v("CD14 REA1314") > 0:
            return "Classical monocyte"
        return "Classical monocyte"  # default for monocyte group

    if new_group == "Dendritic cell":
        if v("CD11c REA1310") > 0 or v("HLA DR REAL550") > 0:
            return "Dendritic cell"
        return "Dendritic cell"  # all initial DCs

    if new_group == "Mast cell":
        return "Mast cell"  # all initial mast cells are mast cells

    if new_group == "Tumor":
        if v("Ki 67 REAL1047") > 0 and v("Alexa647 anti rabbit1") > 0:
            return "Proliferating AML blast"
        if v("Alexa647 anti rabbit1") > 0:
            if v("HLA DR REAL550") > 0.5:
                # Check monocytic score vs quiescent
                mono_s = (4*v("HLA DR REAL550") + 2*v("Alexa647 anti rabbit1")
                          + v("CD43 REA833") + v("CD11c REA1310"))
                quiet_s = 3*v("Alexa647 anti rabbit1") - 2*v("Ki 67 REAL1047")
                if mono_s > quiet_s:
                    return "AML monocytic differentiated"
            return "Quiescent AML blast"
        return None

    if new_group == "Leukemic erythroid":
        if v("Ki 67 REAL1047") > 0:
            return "Proliferating leukemic erythroid"
        if v("CD235a REA175") > 0 or v("Alexa647 anti rabbit1") > 0:
            return "Leukemic erythroid blast"
        return "Leukemic erythroid blast"  # default

    if new_group == "Erythroid":
        return "Mature erythroblast" if v("CD235a REA175") > 0.3 else None

    if new_group == "MSC":
        if v("CD271 REAL709") > 0.5 and v("CD73 REAL1172") > 0:
            return "CD271+ MSC"
        if v("CD73 REAL1172") > 0.5:
            return "Bone marrow MSC"
        return None

    if new_group == "Megakaryocyte":
        return "Megakaryocyte" if v("CD31 REA1312") > 1.0 else None

    if new_group == "Endothelial":
        return "Endothelial" if v("CD34 REAL1217") > 0 else None

    if new_group == "Lymphoid":
        # CD8+ T cells that ended up in lymphoid group
        if v("CD8a REA1024") > 0 and v("CD3 REAL1097") > -0.25:
            return "CD8+ Cytotoxic T"
        # NK: CD16+CD56+ no Tryptase
        if (v("CD16 REA1324") > 0 and v("CD56 REAL1142") > 0
                and v("Mast Cell Tryptase REAL798") < 0.3):
            return "NK cell"
        # Memory T: CD45RO+CD3+
        if v("CD45RO REA611") > 0 and v("CD3 REAL1097") > -0.25:
            return "Memory T lymphoid"
        # Activated HLA-DR+ lymphoid
        if v("HLA DR REAL550") > 1.0:
            return "HLA-DR+ lymphoid"
        # Mixed lymphoid
        if v("CD20 REAL1069") > 0 or v("CD3 REAL1097") > 0:
            return "Mixed B/T lymphoid"
        return None

    if new_group == "Myeloid misc":
        # Apply full myeloid rules to unclassified myeloid cells
        # Priority overrides (Mast cell, Macrophage) already applied above
        if v("CD11c REA1310") > 0 and v("HLA DR REAL550") > 0:
            return "Dendritic cell"
        if v("Ki 67 REAL1047") > 0.5 and v("CD163 REA1309") > 0:
            return "Proliferating myeloid"
        if v("CD16 REA1324") > 0 and v("CD11b REA1321") > 0 and v("CD14 REA1314") < -0.2:
            return "Patrolling monocyte"
        if v("CD14 REA1314") > 0:
            return "Classical monocyte"
        if v("CD15 VIMC6") > 0:
            return "Neutrophil/Granulocyte"
        return None

    return None


# ── Step 2: Process all ROI files ─────────────────────────────────────────────
print("Step 2: Assigning final labels and writing roi_assignments_final/ …")

# Also accumulate data for heatmaps
# heatmap_accum[(new_group, final_label)][marker] = (sum_zscores, count)
heatmap_accum = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

total_kept = total_dropped = 0

for roi_path in sorted(SRC_ROI.glob("*_converted.csv")):
    df = pd.read_csv(roi_path, low_memory=False)

    labels_out    = []
    integers_out  = []
    rows_to_keep  = []

    for idx, row in df.iterrows():
        cl  = str(row.get("cluster_label", ""))
        pg  = str(row.get("pooled_group", ""))
        sc  = int(row.get("sub_cluster", -1))
        ng  = POOL_MAP.get(cl, "DROP")

        if ng == "DROP":
            labels_out.append(None)
            integers_out.append(None)
            rows_to_keep.append(False)
            continue

        # Look up sub_cluster z-scores
        means = zscore_lookup.get((pg, sc), {})
        if not means:
            labels_out.append(None)
            integers_out.append(None)
            rows_to_keep.append(False)
            continue

        final = get_label(ng, means)

        if final is not None:
            labels_out.append(final)
            integers_out.append(LABEL_TO_INT[final])
            rows_to_keep.append(True)
            # Accumulate for heatmaps
            key = (ng, final)
            for m in remaining_markers:
                if m in means:
                    acc = heatmap_accum[key][m]
                    acc[0] += means[m]
                    acc[1] += 1
        else:
            labels_out.append(None)
            integers_out.append(None)
            rows_to_keep.append(False)

    mask = np.array(rows_to_keep, dtype=bool)
    out  = df.iloc[np.where(mask)[0]].copy()
    out["final_label"]         = [l for l, k in zip(labels_out, rows_to_keep) if k]
    out["final_label_integer"] = [i for i, k in zip(integers_out, rows_to_keep) if k]

    out.to_csv(OUT / "roi_assignments_final" / roi_path.name, index=False)
    kept    = mask.sum()
    dropped = (~mask).sum()
    total_kept    += kept
    total_dropped += dropped
    print(f"  {roi_path.name}: kept {kept:5d}  dropped {dropped:5d}")

print(f"\n  Total: kept {total_kept:,}  dropped {total_dropped:,}")


# ── Step 3: final_label_integer_key.csv ──────────────────────────────────────
print("\nStep 3: Writing final_label_integer_key.csv …")
seen = {}
for f in sorted((OUT/"roi_assignments_final").glob("*_converted.csv")):
    df = pd.read_csv(f, usecols=["final_label_integer","final_label"], low_memory=False)
    for _, row in df.drop_duplicates().iterrows():
        k = int(row["final_label_integer"])
        if k not in seen:
            seen[k] = row["final_label"]
key_df = pd.DataFrame(
    [(k, v) for k, v in sorted(seen.items())],
    columns=["final_label_integer","cell_type"]
)
key_df.to_csv(OUT / "final_label_integer_key.csv", index=False)
print(key_df.to_string(index=False))


# ── Step 4: final_label_mapping.csv ──────────────────────────────────────────
print("\nStep 4: Writing final_label_mapping.csv …")
map_rows = []
for i, lbl in enumerate(ALL_LABELS):
    n_cells = sum(
        heatmap_accum[(ng, lbl)][m][1]
        for ng in set(ng for ng, fl in heatmap_accum if fl == lbl)
        for m in remaining_markers
        if heatmap_accum[(ng, lbl)][m][1] > 0
    )
    map_rows.append({"final_label_integer": i, "final_label": lbl,
                     "has_data": any(
                         heatmap_accum[(ng, lbl)]
                         for ng in set(ng for ng, fl in heatmap_accum if fl == lbl))})
pd.DataFrame(map_rows).to_csv(OUT / "final_label_mapping.csv", index=False)
print(f"  Saved ({len(map_rows)} labels)")


# ── Step 5: markers_used.csv ──────────────────────────────────────────────────
pd.DataFrame({"marker": remaining_markers}).to_csv(OUT / "markers_used.csv", index=False)


# ── Heatmap helper ────────────────────────────────────────────────────────────
def shorten(name):
    aliases = {
        "Alexa647 anti rabbit1":"Alexa647-anti-rabbit1","Ki 67 REAL1047":"Ki67",
        "Mast Cell Tryptase REAL798":"MastTryptase","HLA DR REAL550":"HLA-DR",
        "PAX 5 REA140":"PAX5","FoxP3 REA1253":"FoxP3",
        "NPM1mut primaryRab400 polyclonal":"NPM1mut_poly",
        "Arginase 1 REAL1137":"Arginase1","Bcl 2 REA872":"BCL-2",
        "Galectin 9 REA435":"Galectin-9","FcepsilonRIalpha REA758":"FceRI-alpha",
        "beta Catenin REA480":"betaCatenin","TIM 3 REAL818":"TIM-3",
        "CD279 PD1 3 1 3":"PD-1","PI3K p85 alpha REAL589":"PI3K-p85",
        "JNK1 REAL1128":"JNK1","BATF REA486":"BATF","PCNA REA858":"PCNA",
        "Actin REAL650":"Actin","p53 REA1132":"p53","CD45 5B1":"CD45",
        "CD45RO REA611":"CD45RO","CD45RA REAL164":"CD45RA","CD57 REA769":"CD57",
    }
    return aliases.get(name, name.split(" ")[0])

def make_heatmap(agg_df, title, path, fig_w=None, fig_h=None):
    nl, nm = agg_df.shape[1], agg_df.shape[0]
    fw = fig_w or max(4, nl*2.0)
    fh = fig_h or max(8, nm*0.28)
    fig, ax = plt.subplots(figsize=(fw, fh))
    sns.heatmap(agg_df, cmap="RdBu_r", center=0, vmin=-2, vmax=2,
                linewidths=0.3, linecolor="white", ax=ax,
                cbar_kws={"label":"Mean Expression (z-score)","shrink":0.6})
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("Subcluster", fontsize=9)
    ax.set_ylabel("Marker", fontsize=9)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Step 6: Build per-(new_group, final_label) mean z-score matrix ────────────
print("\nStep 5: Building mean expression matrices for heatmaps …")

def build_mean_df(keys):
    """keys: list of (new_group, final_label) to include"""
    rows = {}
    for key in keys:
        ng, fl = key
        acc = heatmap_accum[key]
        if not acc:
            continue
        row = {}
        for m in remaining_markers:
            s, n = acc.get(m, [0.0, 0])
            row[shorten(m)] = s/n if n > 0 else 0.0
        rows[fl] = row
    if not rows:
        return None
    return pd.DataFrame(rows).T   # labels × markers → .T → markers × labels


# Per-group heatmaps
print("\nStep 6: Per-group heatmaps …")
groups_with_data = sorted(set(ng for ng, fl in heatmap_accum))
group_agg_for_combined = []

for ng in groups_with_data:
    keys_here = [(ng, fl) for ng2, fl in heatmap_accum if ng2 == ng]
    # Order labels by their integer
    keys_here_sorted = sorted(set(keys_here), key=lambda k: LABEL_TO_INT.get(k[1], 99))

    agg = build_mean_df(keys_here_sorted)
    if agg is None or agg.empty:
        continue

    agg = agg.T   # markers × labels
    title = f"{ng}  |  {agg.shape[1]} subclusters  |  {agg.shape[0]} markers"
    safe_ng = ng.replace(" ", "_").replace("/","_")
    make_heatmap(agg, title, OUT/"heatmaps_final"/f"{safe_ng}_heatmap.png")
    print(f"  Saved  {safe_ng}_heatmap.png  ({agg.shape[1]} labels × {agg.shape[0]} markers)")
    group_agg_for_combined.append((ng, agg))


# ── Step 7: Combined heatmap ──────────────────────────────────────────────────
print("\nStep 7: Combined heatmap …")
if group_agg_for_combined:
    combined = pd.concat([agg for _, agg in group_agg_for_combined], axis=1)
    nl, nm = combined.shape[1], combined.shape[0]
    fw = max(18, nl*1.5)
    fh = max(18, nm*0.28)
    fig, ax = plt.subplots(figsize=(fw, fh))
    sns.heatmap(combined, cmap="RdBu_r", center=0, vmin=-2, vmax=2,
                linewidths=0.25, linecolor="white", ax=ax,
                cbar_kws={"label":"Mean Expression (z-score)","shrink":0.5,"pad":0.02})
    x_pos = 0
    for ng, agg in group_agg_for_combined:
        n = agg.shape[1]
        if x_pos > 0:
            ax.axvline(x=x_pos, color="black", linewidth=1.5, zorder=5)
        rect = mpatches.FancyBboxPatch(
            (x_pos,-2.2), n, 1.6, boxstyle="square,pad=0",
            facecolor=GROUP_COLORS.get(ng,"#AAAAAA"),
            transform=ax.transData, clip_on=False, zorder=10)
        ax.add_patch(rect)
        ax.text(x_pos+n/2,-1.4, ng, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                transform=ax.transData, clip_on=False, zorder=11)
        x_pos += n
    ax.set_title(f"Nandhu_final_v2 — {nl} labels × {nm} markers  (z-score)",
                 fontsize=11, fontweight="bold", pad=18)
    ax.set_xlabel("")
    ax.set_ylabel("Marker", fontsize=9)
    plt.xticks(rotation=40, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT/"heatmaps_final"/"all_groups_combined_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  all_groups_combined_heatmap.png  ({nl} × {nm})")

n_hm  = len(list((OUT/"heatmaps_final").glob("*.png")))
n_roi = len(list((OUT/"roi_assignments_final").glob("*.csv")))
print(f"""
✓ All done — Nandhu_final_v2
  Approach          : cluster_label-based pooling (not pooled_group)
  Remaining markers : {len(remaining_markers)} / 87
  Cells kept        : {total_kept:,}
  Cells dropped     : {total_dropped:,}
  Final labels      : {len(key_df)}
  Heatmaps          : {n_hm} PNG files
  ROI CSVs          : {n_roi}
  Output            : {OUT}
""")
