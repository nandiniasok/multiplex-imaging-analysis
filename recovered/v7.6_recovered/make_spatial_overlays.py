"""
Nandhu_final_v7.6 spatial overlays + recovery cross-verification.
Part A: one map per ROI (all 49), coloured by v76_label.
Part B: per-ROI recovery panels for Mast cells / Macrophages / Erythroid —
        3 rows (populations) x 3 cols (raw marker intensity | v2 cells | v7.6 cells),
        directly showing that v7.6 recovers the bright cells v2 missed.
Part C: recovery_counts.csv — per-ROI n(v2) vs n(v7.6) for the 3 populations.
"""
import warnings; warnings.filterwarnings("ignore")
import os
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.cm as cm

PROJ=Path(os.environ.get("MRD_PROJ", "/home/hp/projects/multiplex_imaging_analysis"))
OUT =PROJ/"results_soumya"/"Nandhu_final_v7.6"
ODIR=OUT/"spatial_overlays"; ODIR.mkdir(exist_ok=True)
XCOL,YCOL="Nuc X","Nuc Y Inv"

master=pd.read_csv(OUT/"csv"/"cell_assignments.csv")
master["Cell Id"]=master["Cell Id"].astype(str)
key=pd.read_csv(OUT/"csv"/"final_label_integer_key.csv")

# ---- marker intensity from the master adata (consistent panel) ----
ad=sc.read_h5ad(PROJ/"experiments/exp-04-03-2026/stage6_subclustering/final_adata.h5ad")
o=ad.obs.copy(); o["Cell Id"]=o["Cell Id"].astype(str)
def col(name):
    if name in ad.var_names: return np.asarray(ad[:,name].X).ravel()
    return pd.to_numeric(o[name],errors="coerce").values
exprs={"Tryptase":col("Mast Cell Tryptase REAL798"),
       "CD68":col("CD68 REAL1346"),"CD163":col("CD163 REA1309"),
       "CD235a":col("CD235a REA175")}
key_df=pd.DataFrame({"unique_region":o["unique_region"].values,"Cell Id":o["Cell Id"].values,**exprs})
master=master.merge(key_df,on=["unique_region","Cell Id"],how="left")

# ---- stable colour map by coarse family ----
labels=list(key["v76_label"])
FAM=dict(zip(key["v76_label"],key["v76_coarse"]))
fam_cmap={"Lymphoid":cm.Blues,"Myeloid":cm.Oranges,"Erythroid":cm.Purples,
          "Leukemia":cm.Reds,"Immature":cm.Greens,"Stromal":cm.YlGn,
          "Megakaryocyte":cm.spring,"Unassigned-derived":cm.autumn,"Other":None}
COLOR={}
for fam,cmp in fam_cmap.items():
    ls=[l for l in labels if FAM.get(l)==fam and l!="Unassigned/Other"]
    for l,s in zip(ls,np.linspace(0.45,0.9,max(len(ls),1))): COLOR[l]=cmp(s) if cmp else "#999"
COLOR["Unassigned/Other"]="#e5e5e5"; COLOR["Artifact"]="#bbbbbb"
for l in labels: COLOR.setdefault(l,"#777777")
ORDER=[l for l in labels if l!="Unassigned/Other"]+["Unassigned/Other"]

def r01(v):
    v=np.asarray(v,float); lo,hi=np.nanpercentile(v,1),np.nanpercentile(v,99)
    return np.clip((v-lo)/(hi-lo+1e-9),0,1)

# ================= Part A: 49 v73 maps =================
print("Part A: per-ROI v7.6 maps")
for reg,df in master.groupby("unique_region"):
    x,y=df[XCOL].values,df[YCOL].values; n=len(df); ps=max(1.0,min(6.0,4000.0/np.sqrt(max(n,1))/2))
    present=[l for l in ORDER if l in set(df["v76_label"])]
    fig,ax=plt.subplots(figsize=(10,8))
    for lbl in present:
        m=(df["v76_label"]==lbl).values; other=(lbl=="Unassigned/Other")
        ax.scatter(x[m],y[m],s=ps*(0.6 if other else 1.5),c=[COLOR[lbl]],linewidths=0,
                   rasterized=True,alpha=0.3 if other else 0.95,label=f"{lbl} ({int(m.sum()):,})")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"v7.6 cell-type map — {reg} ({n:,} cells)",fontsize=11,fontweight="bold")
    ax.legend(loc="center left",bbox_to_anchor=(1.01,0.5),fontsize=6,markerscale=2,frameon=False)
    fig.tight_layout(); fig.savefig(ODIR/(reg.replace('/','_').replace(' ','_')+"_map.png"),
                                    dpi=110,bbox_inches="tight"); plt.close(fig)
print("  done 49 maps")

# ================= Part B + C: recovery panels + counts =================
POPS=[("Mast cells",   ["Tryptase"],        15,"#d62728"),
      ("Macrophages",  ["CD68","CD163"],    13,"#8c564b"),
      ("Erythroid",    ["CD235a"],          21,"#7b3fa0")]
rows=[]
print("Part B: recovery panels per ROI")
for reg,df in master.groupby("unique_region"):
    x,y=df[XCOL].values,df[YCOL].values; n=len(df); ps=max(1.0,min(6.0,4000.0/np.sqrt(max(n,1))/2))
    fig,axes=plt.subplots(3,3,figsize=(15,15))
    rec={"unique_region":reg,"n_cells":n}
    for r,(pop,mk,v2int,colr) in enumerate(POPS):
        inten=np.nanmean([r01(df[c].values) for c in mk],axis=0)
        v2m=(df["v2_final_label_integer"]==v2int).values
        v3m=(df["v76_label"]==pop).values
        rec[f"{pop}_v2"]=int(v2m.sum()); rec[f"{pop}_v74"]=int(v3m.sum())
        mklab="+".join(mk)
        ax=axes[r,0]; sc0=ax.scatter(x,y,c=inten,s=ps,cmap="magma",vmin=0,vmax=1,linewidths=0,rasterized=True)
        ax.set_title(f"{pop}: raw {mklab}",fontsize=10); plt.colorbar(sc0,ax=ax,shrink=0.6)
        ax=axes[r,1]; ax.scatter(x,y,s=ps,c="#dddddd",linewidths=0,rasterized=True)
        ax.scatter(x[v2m],y[v2m],s=ps*1.7,c=colr,linewidths=0,rasterized=True)
        ax.set_title(f"v2 {pop} = {int(v2m.sum()):,}",fontsize=10)
        ax=axes[r,2]; ax.scatter(x,y,c=inten,s=ps,cmap="Greys",vmin=0,vmax=1.3,linewidths=0,rasterized=True)
        ax.scatter(x[v3m],y[v3m],s=ps*1.7,c=colr,linewidths=0,rasterized=True,label=f"v7.6 {pop}")
        ax.set_title(f"v7.6 {pop} = {int(v3m.sum()):,}  (Δ={int(v3m.sum())-int(v2m.sum()):+,})",fontsize=10)
        ax.legend(loc="upper right",fontsize=8,markerscale=2,framealpha=0.8)
        for c in range(3): axes[r,c].set_aspect("equal"); axes[r,c].invert_yaxis(); axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
    fig.suptitle(f"v7.6 recovery — {reg}\ncols: raw marker | v2 cells | v7.6 cells (grey background = raw marker)",
                 fontsize=13,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(ODIR/(reg.replace('/','_').replace(' ','_')+"_recovery.png"),dpi=115,bbox_inches="tight"); plt.close(fig)
    rows.append(rec)
rc=pd.DataFrame(rows)
for pop,_,_,_ in POPS: rc[f"{pop}_delta"]=rc[f"{pop}_v74"]-rc[f"{pop}_v2"]
rc.to_csv(OUT/"csv"/"recovery_counts.csv",index=False)
print("  wrote recovery_counts.csv")
tot={p:(rc[f'{p}_v2'].sum(),rc[f'{p}_v74'].sum()) for p,_,_,_ in POPS}
print("  totals v2 -> v7.6:",{k:f'{a:,}->{b:,} (+{b-a:,})' for k,(a,b) in tot.items()})
print("DONE.")
