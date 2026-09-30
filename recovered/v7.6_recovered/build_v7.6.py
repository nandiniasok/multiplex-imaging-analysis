# %% [markdown]
# # Nandhu_final_v7.6 — corrections on v7.5
# 1. **CD8+ T** left as-is: ROI09 CD8+T cells have genuine raw CD3/CD8a (not background), so the "empty-space"
#    look is real sparse T cells — not correctable by marker gating without deleting true T cells.
# 2. **Erythroid / MSC** rebuilt by marker: **free-pool + dominant-marker**. A cell → Erythroid if it is in the
#    free pool (Unassigned/Artifact/Ery/MSC) with CD235a z>0.5, OR in a protected cluster where **CD235a is its
#    single strongest lineage marker** (z>0.5). Same for MSC with CD271. This recovers the missing bright
#    CD235a+/CD271+ cells while moving only 0–12% of any protected cluster (mostly CD235a-dominant erythroid
#    progenitors). The two kept Leiden clusters (below) are not touched.
# 3. **Mature myeloid CD11b+CD15+ removed** → Unassigned.
# 4. **Kept only Leiden 19 (CD68+CD3+Tryptase+) & 25 (Ki67+CD147+CD45RA+)**; all other Unassigned-Leiden → Unassigned.
# Plus: **recovery_quantification.csv** — for every named subcluster, its defining marker+ staining count vs how
# many the cluster recovered (recovery %) and purity %.

# %%
import sys, types
_stub=types.ModuleType("umap.parametric_umap"); _stub.ParametricUMAP=None; _stub.load_ParametricUMAP=None
sys.modules["umap.parametric_umap"]=_stub
import pandas as pd, numpy as np, os, scanpy as sc, yaml, warnings
warnings.filterwarnings("ignore"); pd.options.mode.chained_assignment=None
PROJ=os.environ.get("MRD_PROJ", "/home/hp/projects/multiplex_imaging_analysis"); RS=f"{PROJ}/results_soumya"
V75=f"{RS}/Nandhu_final_v7.5"; OUT=f"{RS}/Nandhu_final_v7.6"; KEY=["unique_region","Cell Id"]
for d in ["csv","heatmaps_final","roi_assignments_final","spatial_overlays"]: os.makedirs(f"{OUT}/{d}",exist_ok=True)
def keyed(df): df=df.copy(); df["Cell Id"]=df["Cell Id"].astype(str); return df.set_index(KEY)

m=keyed(pd.read_csv(f"{V75}/csv/cell_assignments.csv")); UNI=m.index
final=m["v75_label"].astype(str).copy(); v2int=m["v2_final_label_integer"]
COARSE=dict(zip(m["v75_label"],m["v75_coarse"]))

ad=sc.read_h5ad(f"{PROJ}/experiments/exp-04-03-2026/stage6_subclustering/final_adata.h5ad")
o=ad.obs.copy(); o["Cell Id"]=o["Cell Id"].astype(str)
MI=pd.MultiIndex.from_arrays([o["unique_region"].values,o["Cell Id"].values],names=KEY)
Xdf=pd.DataFrame(np.asarray(ad.X),columns=list(ad.var_names)); Xdf.index=MI
META={"Cell Id","Nuc X","Nuc Y Inv","region_num","unique_region","leiden","cluster_label","cluster_group",
      "cluster_labelled","machine","sub_cluster","final_label","final_label_integer","EmptyPE1"}
Odf=o.set_index(MI)[[c for c in o.columns if c not in META]]
EXPR=pd.concat([Xdf,Odf],axis=1); EXPR=EXPR[~EXPR.index.duplicated()].reindex(UNI).apply(pd.to_numeric,errors="coerce").dropna(axis=1,how="all")
Z=(EXPR-EXPR.mean())/EXPR.std(ddof=0)
def zz(n): return Z[n] if n in Z.columns else pd.Series(np.nan,index=UNI)
CFG=[x for x in yaml.safe_load(open(f"{PROJ}/config_18.06.26.yaml"))["selected_markers"] if x in Z.columns]

KEEP_LEI=["Unassigned-Leiden: CD68+CD3+Tryptase+","Unassigned-Leiden: Ki67+CD147+CD45RA+"]

print("### #3 remove Mature myeloid CD11b+CD15+ -> Unassigned ###")
n=int((final=="Mature myeloid CD11b+CD15+").sum()); final[final=="Mature myeloid CD11b+CD15+"]="Unassigned/Other"; print("  moved",n)

print("### #4 keep only Leiden 19 & 25; others -> Unassigned ###")
lei_all=[l for l in final.unique() if str(l).startswith("Unassigned-Leiden:")]
removed=0
for l in lei_all:
    if l not in KEEP_LEI:
        removed+=int((final==l).sum()); final[final==l]="Unassigned/Other"
print(f"  removed {len(lei_all)-len(KEEP_LEI)} Leiden clusters ({removed} cells); kept {KEEP_LEI}")

# %% [markdown]
# ## #2 Erythroid / MSC — free-pool + dominant-marker
# %%
LIN=["CD3 REAL1097","CD8a REA1024","CD4 REA1307","CD14 REA1314","CD11b REA1321","CD15 VIMC6","CD16 REA1324",
 "CD11c REA1310","CD34 REAL1217","CD31 REA1312","CD271 REAL709","CD38 REAL719","CD20 REAL1069","CD56 REAL1142",
 "HLA DR REAL550","Alexa647 anti rabbit1","Arginase 1 REAL1137","CD68 REAL1346","CD163 REA1309",
 "Mast Cell Tryptase REAL798","CD235a REA175"]
LIN=[x for x in LIN if x in Z.columns]
dom=Z[LIN].idxmax(axis=1); dommax=Z[LIN].max(axis=1)
zery=zz("CD235a REA175"); zmsc=zz("CD271 REAL709")
PROT=["CD4+ T","CD8+ T","Plasma cells","Dendritic cells (CD11c+HLA-DR+)","Granulocyte","HLA-DR-hi APC (CD1c+)",
 "Macrophages","Mast cells","Monocytes","NPM1_blasts","Mixed progenitors","Endothelial","Megakaryocyte"]
inF=final.isin(["Unassigned/Other","Artifact","Erythroid","MSC"]); inP=final.isin(PROT)
ery_cond=(inF&(zery>0.5)) | (inP&(dom=="CD235a REA175")&(dommax>0.5))
msc_cond=(inF&(zmsc>0.5)) | (inP&(dom=="CD271 REAL709")&(dommax>0.5))
before=final.copy(); new=final.copy()
both=ery_cond&msc_cond
new[ery_cond&~msc_cond]="Erythroid"; new[msc_cond&~ery_cond]="MSC"
new[both]=np.where(zery[both].values>=zmsc[both].values,"Erythroid","MSC")
was_em=final.isin(["Erythroid","MSC"]); new[was_em&~ery_cond&~msc_cond]="Unassigned/Other"
final=new; COARSE["Erythroid"]="Erythroid"; COARSE["MSC"]="Stromal"
print("### #2 Erythroid/MSC rebuilt (free-pool + dominant-marker) ###")
print(f"  Erythroid {int((final=='Erythroid').sum())} | MSC {int((final=='MSC').sum())}")
moved=before[(before!=final)&before.isin(PROT)]
print("  cells moved OUT of protected clusters:", {k:int(v) for k,v in moved.value_counts().items()})

# %% [markdown]
# ## Write master, key, per-ROI, heatmaps
# %%
master=pd.DataFrame(index=UNI); master["v2_final_label_integer"]=v2int
master["v76_label"]=final.values; master["v76_coarse"]=[COARSE.get(x,"Other") for x in final.values]
oMI=o.set_index(MI)
for c in ["machine","Nuc X","Nuc Y Inv"]: master[c]=oMI[c].reindex(UNI).values
lo=master.groupby("v76_label")["v76_coarse"].first().reset_index()
crank={c:i for i,c in enumerate(["Lymphoid","Myeloid","Erythroid","Leukemia","Immature","Stromal","Megakaryocyte","Unassigned-derived","Other"])}
lo["r"]=lo["v76_coarse"].map(lambda c:crank.get(c,99)); lo=lo.sort_values(["r","v76_label"]).reset_index(drop=True)
lo["v76_label_int"]=range(len(lo)); master["v76_label_int"]=master["v76_label"].map(dict(zip(lo["v76_label"],lo["v76_label_int"])))
master=master.reset_index(); master.to_csv(f"{OUT}/csv/cell_assignments.csv",index=False)
key=lo.copy(); key["n_cells"]=key["v76_label"].map(final.value_counts()); key=key[["v76_label_int","v76_label","v76_coarse","n_cells"]]
key.to_csv(f"{OUT}/csv/final_label_integer_key.csv",index=False)
print("total:",len(master),"| sum:",key["n_cells"].sum()); print(key.to_string(index=False))
for reg,g in master.groupby("unique_region"):
    g.to_csv(f"{OUT}/roi_assignments_final/{reg.replace('/','_').replace(' ','_')}_v7_6.csv",index=False)

# %% [markdown]
# ## Recovery-quantification CSV: cluster vs true marker staining (top markers)
# %%
DEFN={  # cluster -> (markers, gate 'all'/'any')
 "CD4+ T":(["CD3 REAL1097","CD4 REA1307"],"all"),
 "CD8+ T":(["CD3 REAL1097","CD8a REA1024"],"all"),
 "Plasma cells":(["CD38 REAL719"],"all"),
 "Dendritic cells (CD11c+HLA-DR+)":(["CD11c REA1310","HLA DR REAL550"],"all"),
 "Granulocyte":(["CD15 VIMC6","Arginase 1 REAL1137"],"any"),
 "HLA-DR-hi APC (CD1c+)":(["HLA DR REAL550","CD1c REAL1005"],"all"),
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
    if not ok: return pd.Series(False,index=UNI)
    r=ok[0]
    for x in ok[1:]: r=(r&x) if gate=="all" else (r|x)
    return r
rows=[]
for lbl,(markers,gate) in DEFN.items():
    pm=posmask(markers,gate); incl=(final==lbl)
    stain=int(pm.sum()); size=int(incl.sum()); rec=int((pm&incl).sum())
    rows.append({"subcluster":lbl,"defining_markers":" & ".join(markers),
                 "gate":gate,"marker+_total(staining)":stain,"cluster_size":size,"marker+_in_cluster":rec,
                 "recovery_%":round(100*rec/stain,1) if stain else 0.0,
                 "purity_%":round(100*rec/size,1) if size else 0.0})
rq=pd.DataFrame(rows); rq.to_csv(f"{OUT}/csv/recovery_quantification.csv",index=False)
print("\n### recovery_quantification.csv ###"); print(rq.to_string(index=False))

# %% [markdown]
# ## Heatmaps (Alexa647 renamed)
# %%
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
RENAME={"Alexa647 anti rabbit1":"Alexa647 anti rabbit1_NPM1_mut"}
def heatmap(markers,fname,title):
    mk=[x for x in markers if x in Z.columns]
    show=[l for l in key["v76_label"] if l not in ("Unassigned/Other","Artifact")]
    M=pd.DataFrame({l:Z.loc[final[final==l].index,mk].mean() for l in show}).rename(index=RENAME)
    plt.figure(figsize=(max(12,0.5*M.shape[1]+4),max(8,0.26*M.shape[0]+2)))
    sns.heatmap(M,cmap="RdBu_r",center=0,vmin=-2,vmax=2,cbar_kws={"label":"mean z-score","shrink":0.5},linewidths=0.3,linecolor="#ddd")
    plt.title(title,fontsize=13,weight="bold"); plt.xlabel("cell type (v7.6)"); plt.ylabel("marker")
    plt.xticks(rotation=90,fontsize=8); plt.yticks(fontsize=7); plt.tight_layout()
    plt.savefig(f"{OUT}/heatmaps_final/{fname}",dpi=200,bbox_inches="tight"); plt.close(); M.to_csv(f"{OUT}/heatmaps_final/{fname.replace('.png','_matrix.csv')}")
    print(" ",fname,M.shape)
heatmap(CFG,"all_groups_combined_heatmap_selected_markers.png",f"all groups combined heatmap — selected markers ({len(CFG)}) | Nandhu_final_v7.6")
heatmap(list(Z.columns),"all_groups_combined_heatmap_all_markers.png",f"all groups combined heatmap — all markers ({Z.shape[1]}) | Nandhu_final_v7.6")
print("DONE.")
