# %% [markdown]
# # Nandhu_version_final — corrections on v7.6
# Starts from the saved v7.6 labels (`Nandhu_final_v7.6/csv/cell_assignments.csv`). **No clustering is run.**
# Only the changes below are made; every other v7.6 cell keeps its label.
#
# 1. **Dendritic-cell naming restored.** v2 "Dendritic cell" (v2 label 14) = the v7.4 cluster
#    "Immature myeloid CD11c+", which v7.5 renamed "HLA-DR-hi APC (CD1c+)" (1,860 of the 2,116 v2 DCs are there).
#    The two v7.5 renames are swapped back:
#    * v7.6 "HLA-DR-hi APC (CD1c+)"           → **Immature myeloid CD11c+**
#    * v7.6 "Dendritic cells (CD11c+HLA-DR+)" → **HLA-DR-hi APC (CD1c+)**
#    These are renames only. No cells move.
# 2. **All tumour cells go back into NPM1_blasts.** Tumour is defined the way the June spatial analysis and the
#    significant Aug 9_New analysis defined it:
#    * (a) every v2 **Tumor** cell (v2 labels 18 Proliferating AML blast, 19 Quiescent AML blast,
#      20 AML monocytic differentiated) and every v2 **Tumor/Erythroid** cell (22 Leukemic erythroid blast,
#      23 Proliferating leukemic erythroid), whatever its v7.6 label;
#    * (b) cells in the two kept Unassigned-Leiden clusters that are **NPM1⁺ and CD43⁺ and CD147⁺**
#      (z > 0.5, the same z used for every v7.x marker gate);
#    * (c) cells in Unassigned/Other or Artifact that are **NPM1_mut⁺** (Alexa647 per-machine z > 0.5).
#      This is the exact extra-tumour rule in `Spatial_analysis_Aug 9_New/spatial_analysis_v76_recovered_tumour.py`.
# 3. Nothing else changes.

# %%
import sys, types
_stub=types.ModuleType("umap.parametric_umap"); _stub.ParametricUMAP=None; _stub.load_ParametricUMAP=None
sys.modules["umap.parametric_umap"]=_stub
import pandas as pd, numpy as np, os, scanpy as sc, yaml, warnings
warnings.filterwarnings("ignore"); pd.options.mode.chained_assignment=None
REC=os.environ.get("MRD_REC","/mnt/e/MRD analysis Apr_2026/30 May 2026_Claude new results/new 18.06.26")
PROJ=os.environ.get("MRD_PROJ",f"{REC}/proj_recovered")   # where rebuild_final_adata.py wrote final_adata.h5ad
ADATA=f"{PROJ}/experiments/exp-04-03-2026/stage6_subclustering/final_adata.h5ad"
V76=f"{REC}/Nandhu_final_v7.6"
OUT=os.environ.get("MRD_OUT","/mnt/e/multiplex-imaging-analysis/data/output/Nandhu_version_final")
KEY=["unique_region","Cell Id"]
for d in ["csv","heatmaps_final","roi_assignments_final"]: os.makedirs(f"{OUT}/{d}",exist_ok=True)

m=pd.read_csv(f"{V76}/csv/cell_assignments.csv",dtype={"Cell Id":str}).set_index(KEY); UNI=m.index
v76=m["v76_label"].astype(str); final=v76.copy(); v2int=m["v2_final_label_integer"]
COARSE=dict(zip(m["v76_label"],m["v76_coarse"]))
print("v7.6 cells:",len(m))

# expression exactly as in build_v7.6.py (X = 25 config markers + raw obs features; global z over all cells)
ad=sc.read_h5ad(ADATA)
o=ad.obs.copy(); o["Cell Id"]=o["Cell Id"].astype(str)
MI=pd.MultiIndex.from_arrays([o["unique_region"].values,o["Cell Id"].values],names=KEY)
Xdf=pd.DataFrame(np.asarray(ad.X),columns=list(ad.var_names)); Xdf.index=MI
META={"Cell Id","Nuc X","Nuc Y Inv","region_num","unique_region","leiden","cluster_label","cluster_group",
      "cluster_labelled","machine","sub_cluster","final_label","final_label_integer","EmptyPE1"}
Odf=o.set_index(MI)[[c for c in o.columns if c not in META]]
EXPR=pd.concat([Xdf,Odf],axis=1); EXPR=EXPR[~EXPR.index.duplicated()].reindex(UNI).apply(pd.to_numeric,errors="coerce").dropna(axis=1,how="all")
Z=(EXPR-EXPR.mean())/EXPR.std(ddof=0)
def zz(n): return Z[n] if n in Z.columns else pd.Series(np.nan,index=UNI)
CFG=[x for x in yaml.safe_load(open(f"{REC}/config_18.06.26.yaml"))["selected_markers"] if x in Z.columns]
machine=o.set_index(MI)["machine"].reindex(UNI)

# %% [markdown]
# ## 1. Dendritic-cell naming: swap the two v7.5 renames back
# %%
OLD_DC,OLD_APC="Dendritic cells (CD11c+HLA-DR+)","HLA-DR-hi APC (CD1c+)"
IMM,APC="Immature myeloid CD11c+","HLA-DR-hi APC (CD1c+)"
print("v2 Dendritic cells (v2=14) by v7.6 label:"); print(v76[v2int==14].value_counts().to_string())
mDC,mAPC=(v76==OLD_DC),(v76==OLD_APC)
final[mAPC]=IMM; final[mDC]=APC
COARSE[IMM]="Myeloid"; COARSE[APC]="Myeloid"
print(f"  {OLD_APC!r} -> {IMM!r}: {int(mAPC.sum())} cells")
print(f"  {OLD_DC!r} -> {APC!r}: {int(mDC.sum())} cells")
for lbl in [IMM,APC]:
    s=final==lbl; print(f"  {lbl}: mean z CD11c={zz('CD11c REA1310')[s].mean():.2f} HLA-DR={zz('HLA DR REAL550')[s].mean():.2f} CD1c={zz('CD1c REAL1005')[s].mean():.2f}")

# %% [markdown]
# ## 2. Put every tumour cell back into NPM1_blasts
# %%
TUMOUR_V2={18:"Tumor: Proliferating AML blast",19:"Tumor: Quiescent AML blast",20:"Tumor: AML monocytic differentiated",
           22:"Tumor/Erythroid: Leukemic erythroid blast",23:"Tumor/Erythroid: Proliferating leukemic erythroid"}
LEI=["Unassigned-Leiden: CD68+CD3+Tryptase+","Unassigned-Leiden: Ki67+CD147+CD45RA+"]
src_a=v2int.isin(list(TUMOUR_V2))
triple=(zz("Alexa647 anti rabbit1")>0.5)&(zz("CD43 REA833")>0.5)&(zz("CD147 REA282")>0.5)
src_b=(~src_a)&v76.isin(LEI)&triple
alexa=EXPR["Alexa647 anti rabbit1"]
z_npm_machine=alexa.groupby(machine).transform(lambda s:(s-s.mean())/(s.std()+1e-9))   # as in Aug 9_New
src_c=(~src_a)&(~src_b)&v76.isin(["Unassigned/Other","Artifact"])&(z_npm_machine>0.5)
to_npm=src_a|src_b|src_c
reason=pd.Series("",index=UNI)
reason[src_a]="(a) v2 "+v2int[src_a].astype(int).map(TUMOUR_V2)
reason[src_b]="(b) Leiden NPM1+CD43+CD147+"; reason[src_c]="(c) Unassigned/Artifact NPM1_mut+"
before_npm=int((final=="NPM1_blasts").sum())
final[to_npm]="NPM1_blasts"; COARSE["NPM1_blasts"]="Leukemia"
print(f"(a) v2 tumour cells: {int(src_a.sum())} | (b) Leiden NPM1+CD43+CD147+: {int(src_b.sum())} | (c) Unassigned/Artifact NPM1_mut+: {int(src_c.sum())}")
print(f"NPM1_blasts {before_npm} -> {int((final=='NPM1_blasts').sum())}")
for l in LEI: print(f"  {l}: NPM1+ {int(((v76==l)&(zz('Alexa647 anti rabbit1')>0.5)).sum())}, NPM1+CD43+CD147+ {int(((v76==l)&triple).sum())}")

moved=to_npm&(v76!="NPM1_blasts")
tr=pd.crosstab(v76[moved].rename("v7.6 label (before)"),reason[moved].rename("reason")); tr["total_moved"]=tr.sum(axis=1)
tr=tr.sort_values("total_moved",ascending=False); tr.loc["TOTAL"]=tr.sum()
tr.to_csv(f"{OUT}/csv/tumour_recovery_by_source.csv"); print(tr.to_string())

# %% [markdown]
# ## Check: what changed relative to v7.6
# %%
chg=pd.DataFrame({"v76_label":v76.values,"final_label":final.values})
chg=chg.groupby(["v76_label","final_label"]).size().rename("n_cells").reset_index()
chg["changed"]=np.where(chg.v76_label==chg.final_label,"unchanged",
                        np.where(chg.final_label.isin([IMM,APC])&chg.v76_label.isin([OLD_DC,OLD_APC]),"renamed","moved"))
chg.to_csv(f"{OUT}/csv/changes_vs_v7.6.csv",index=False)
cmp=pd.DataFrame({"v7.6_n":v76.value_counts()}).join(pd.DataFrame({"final_n":final.value_counts()}),how="outer").fillna(0).astype(int)
cmp["lost_to_NPM1_blasts"]=v76[moved].value_counts().reindex(cmp.index).fillna(0).astype(int)
cmp["%_kept"]=np.where(cmp["v7.6_n"]>0,(100*(1-cmp.lost_to_NPM1_blasts/cmp["v7.6_n"].clip(lower=1))).round(1),np.nan)
cmp.to_csv(f"{OUT}/csv/cluster_size_v76_vs_final.csv"); print(cmp.to_string())

# %% [markdown]
# ## Write master, key, per-ROI, recovery quantification, heatmaps
# %%
master=pd.DataFrame(index=UNI); master["v2_final_label_integer"]=v2int
master["v76_label"]=v76.values; master["final_label"]=final.values
master["final_coarse"]=[COARSE.get(x,"Other") for x in final.values]; master["tumour_recovery_reason"]=reason.values
oMI=o.set_index(MI)
for c in ["machine","Nuc X","Nuc Y Inv"]: master[c]=oMI[c].reindex(UNI).values
lo=master.groupby("final_label")["final_coarse"].first().reset_index()
crank={c:i for i,c in enumerate(["Lymphoid","Myeloid","Erythroid","Leukemia","Immature","Stromal","Megakaryocyte","Unassigned-derived","Other"])}
lo["r"]=lo["final_coarse"].map(lambda c:crank.get(c,99)); lo=lo.sort_values(["r","final_label"]).reset_index(drop=True)
lo["final_label_int"]=range(len(lo)); master["final_label_int"]=master["final_label"].map(dict(zip(lo["final_label"],lo["final_label_int"])))
master=master.reset_index(); master.to_csv(f"{OUT}/csv/cell_assignments.csv",index=False)
key=lo.copy(); key["n_cells"]=key["final_label"].map(final.value_counts()); key=key[["final_label_int","final_label","final_coarse","n_cells"]]
key.to_csv(f"{OUT}/csv/final_label_integer_key.csv",index=False)
print("total:",len(master),"| sum:",key["n_cells"].sum()); print(key.to_string(index=False))
for reg,g in master.groupby("unique_region"):
    g.to_csv(f"{OUT}/roi_assignments_final/{reg.replace('/','_').replace(' ','_')}_final.csv",index=False)
roi=master.assign(roi=master.unique_region).pivot_table(index="roi",columns="final_label",values="Cell Id",aggfunc="count",fill_value=0)
roi.insert(0,"total",roi.sum(axis=1)); roi.to_csv(f"{OUT}/csv/roi_celltype_counts.csv")

DEFN={
 "CD4+ T":(["CD3 REAL1097","CD4 REA1307"],"all"),
 "CD8+ T":(["CD3 REAL1097","CD8a REA1024"],"all"),
 "Plasma cells":(["CD38 REAL719"],"all"),
 "HLA-DR-hi APC (CD1c+)":(["HLA DR REAL550","CD1c REAL1005"],"all"),
 "Immature myeloid CD11c+":(["CD11c REA1310"],"all"),
 "Granulocyte":(["CD15 VIMC6","Arginase 1 REAL1137"],"any"),
 "Macrophages":(["CD68 REAL1346","CD163 REA1309"],"all"),
 "Mast cells":(["Mast Cell Tryptase REAL798"],"all"),
 "Monocytes":(["CD14 REA1314"],"all"),
 "Erythroid":(["CD235a REA175"],"all"),
 "NPM1_blasts":(["Alexa647 anti rabbit1"],"all"),
 "Mixed progenitors":(["CD34 REAL1217"],"all"),
 "Endothelial":(["CD34 REAL1217","CD31 REA1312"],"all"),
 "MSC":(["CD271 REAL709"],"all"),
 "Megakaryocyte":(["CD31 REA1312"],"all"),
 "Unassigned-Leiden: CD68+CD3+Tryptase+":(["CD68 REAL1346","CD3 REAL1097","Mast Cell Tryptase REAL798"],"all"),
 "Unassigned-Leiden: Ki67+CD147+CD45RA+":(["Ki 67 REAL1047","CD147 REA282","CD45RA REAL164"],"all"),
}
def posmask(markers,gate):
    ok=[(zz(mk)>0.5) for mk in markers if mk in Z.columns]
    r=ok[0]
    for x in ok[1:]: r=(r&x) if gate=="all" else (r|x)
    return r
rows=[]
for lbl,(markers,gate) in DEFN.items():
    pm=posmask(markers,gate); incl=(final==lbl)
    stain=int(pm.sum()); size=int(incl.sum()); rec=int((pm&incl).sum())
    rows.append({"subcluster":lbl,"defining_markers":" & ".join(markers),"gate":gate,"marker+_total(staining)":stain,
                 "cluster_size":size,"marker+_in_cluster":rec,"recovery_%":round(100*rec/stain,1) if stain else 0.0,
                 "purity_%":round(100*rec/size,1) if size else 0.0})
rq=pd.DataFrame(rows); rq.to_csv(f"{OUT}/csv/recovery_quantification.csv",index=False)
print(rq.to_string(index=False))

# %%
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
RENAME={"Alexa647 anti rabbit1":"Alexa647 anti rabbit1_NPM1_mut"}
def heatmap(markers,fname,title):
    mk=[x for x in markers if x in Z.columns]
    show=[l for l in key["final_label"] if l not in ("Unassigned/Other","Artifact")]
    M=pd.DataFrame({l:Z.loc[final[final==l].index,mk].mean() for l in show}).rename(index=RENAME)
    plt.figure(figsize=(max(12,0.5*M.shape[1]+4),max(8,0.26*M.shape[0]+2)))
    sns.heatmap(M,cmap="RdBu_r",center=0,vmin=-2,vmax=2,cbar_kws={"label":"mean z-score","shrink":0.5},linewidths=0.3,linecolor="#ddd")
    plt.title(title,fontsize=13,weight="bold"); plt.xlabel("cell type (Nandhu_version_final)"); plt.ylabel("marker")
    plt.xticks(rotation=90,fontsize=8); plt.yticks(fontsize=7); plt.tight_layout()
    plt.savefig(f"{OUT}/heatmaps_final/{fname}",dpi=200,bbox_inches="tight"); plt.close(); M.to_csv(f"{OUT}/heatmaps_final/{fname.replace('.png','_matrix.csv')}")
    print(" ",fname,M.shape)
heatmap(CFG,"all_groups_combined_heatmap_selected_markers.png",f"all groups combined heatmap — selected markers ({len(CFG)}) | Nandhu_version_final")
heatmap(list(Z.columns),"all_groups_combined_heatmap_all_markers.png",f"all groups combined heatmap — all markers ({Z.shape[1]}) | Nandhu_version_final")
print("DONE.")
