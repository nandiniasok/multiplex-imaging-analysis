"""PDF summary of Nandhu_final_v7.6."""
import os
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak, HRFlowable)

V=Path(os.environ.get("MRD_PROJ", "/home/hp/projects/multiplex_imaging_analysis"))/"results_soumya"/"Nandhu_final_v7.6"
HM=V/"heatmaps_final"; SP=V/"spatial_overlays"; OUT=V/"Nandhu_final_v7.6_summary.pdf"
master=pd.read_csv(V/"csv"/"cell_assignments.csv"); key=pd.read_csv(V/"csv"/"final_label_integer_key.csv")
rq=pd.read_csv(V/"csv"/"recovery_quantification.csv")
N=len(master); nA=int((master.machine=="machine_A").sum()); nB=int((master.machine=="machine_B").sum()); nROI=master.unique_region.nunique()
ss=getSampleStyleSheet()
H1=ParagraphStyle("H1",parent=ss["Heading1"],fontSize=17,spaceAfter=6,textColor=colors.HexColor("#1a3c6e"))
H2=ParagraphStyle("H2",parent=ss["Heading2"],fontSize=12.5,spaceBefore=10,spaceAfter=4,textColor=colors.HexColor("#22558b"))
BODY=ParagraphStyle("BODY",parent=ss["BodyText"],fontSize=9.6,leading=13.5,spaceAfter=5)
BULLET=ParagraphStyle("BULLET",parent=BODY,leftIndent=12,bulletIndent=2)
SMALL=ParagraphStyle("SMALL",parent=BODY,fontSize=8.3,leading=11,textColor=colors.HexColor("#444"))
CAP=ParagraphStyle("CAP",parent=SMALL,alignment=1)
def P(t,s=BODY): return Paragraph(t,s)
def B(t): return Paragraph("• "+t,BULLET)
def rule(): return HRFlowable(width="100%",thickness=0.6,color=colors.HexColor("#bbb"),spaceBefore=4,spaceAfter=8)
def tbl(data,colw,fs=8):
    t=Table(data,colWidths=colw,repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),fs),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#ccc")),("TOPPADDING",(0,0),(-1,-1),2.5),("BOTTOMPADDING",(0,0),(-1,-1),2.5),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#22558b")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")])); return t
def img(path,w=17*cm,max_h=22*cm):
    from PIL import Image as PImage
    iw,ih=PImage.open(path).size; W,H=w,w*ih/iw
    if H>max_h: H,W=max_h,max_h*iw/ih
    return Image(str(path),width=W,height=H)
st=[]
st.append(P("Nandhu_final_v7.6 — corrections on v7.5",H1))
st.append(P(f"AML bone-marrow multiplex imaging · machines A + B · {nROI} ROIs · {N:,} cells (A {nA:,} + B {nB:,}).",SMALL))
st.append(rule())
st.append(P("Corrections",H2))
st.append(B("<b>1. CD8+ T — left unchanged.</b> The ROI09 CD8+T cells have genuine raw CD3 (~1.3) and CD8a (~1.6) "
            "and above-median total signal (0% below the background floor) — they are real sparse T cells, not "
            "empty-space artifacts, so marker gating cannot remove them without deleting true T cells."))
st.append(B("<b>2. Erythroid &amp; MSC — rebuilt by marker (free-pool + dominant-marker).</b> A cell joins "
            "Erythroid if it is in the free pool (Unassigned/Artifact/Ery/MSC) with CD235a z&gt;0.5, OR sits in "
            "another cluster where <b>CD235a is its single strongest lineage marker</b> (z&gt;0.5); same for MSC with "
            "CD271. This recovered the missing bright CD235a+/CD271+ cells: <b>Erythroid 17,803 → 24,731</b>, "
            "<b>MSC 3,011 → 9,144</b>. Cause of the v7.5 gap: CD235a/CD271 are broadly detected in dense marrow "
            "(spillover/contamination), and v7.5 built these clusters from source-membership lists, not the marker."))
st.append(B("<b>3. Mature myeloid CD11b+CD15+ removed</b> → Unassigned (4,896 cells)."))
st.append(B("<b>4. Kept only Leiden clusters 19 (CD68+CD3+Tryptase+) &amp; 25 (Ki67+CD147+CD45RA+);</b> the other 12 "
            "Unassigned-Leiden clusters (17,587 cells) returned to Unassigned."))
moved=master.merge(pd.read_csv(Path(str(V).replace('v7.6','v7.5'))/"csv"/"cell_assignments.csv")[["unique_region","Cell Id","v75_label"]],on=["unique_region","Cell Id"],how="left")
mv=moved[(moved.v75_label.isin(["CD4+ T","CD8+ T","Plasma cells","Dendritic cells (CD11c+HLA-DR+)","Granulocyte","HLA-DR-hi APC (CD1c+)","Macrophages","Mast cells","Monocytes","NPM1_blasts","Mixed progenitors","Endothelial","Megakaryocyte"]))&(moved.v75_label!=moved.v76_label)]
st.append(P(f"#2 moved {len(mv):,} cells out of protected clusters (into Erythroid/MSC) — each 0–15% of a cluster, "
            "mostly CD235a-dominant erythroid progenitors from Mixed progenitors and Granulocyte. Every protected "
            "cluster kept ≥85% of its cells; the two kept Leiden clusters were not touched. Alexa647 anti rabbit1 "
            "is shown as Alexa647 anti rabbit1_NPM1_mut on the heatmaps.",SMALL))

st.append(P("Recovery quantification — cluster vs true marker staining",H2))
st.append(P("For each named cluster: how many cells in the whole dataset are positive for its defining marker(s) "
            "(z&gt;0.5, 'staining'); the cluster size; how many of the cluster's cells are marker+ ; "
            "<b>recovery%</b> = marker+ in cluster ÷ total marker+ (what fraction of the stained cells we captured); "
            "<b>purity%</b> = marker+ in cluster ÷ cluster size.",SMALL))
rows=[["subcluster","defining markers","gate","staining+","size","recovery%","purity%"]]
for _,r in rq.iterrows():
    rows.append([str(r["subcluster"]).replace("Unassigned-Leiden: ","Leiden: "),
                 str(r["defining_markers"]).replace(" REAL"," ").replace(" REA"," ").replace(" VIMC6","").replace(" anti rabbit1","")[:30],
                 str(r["gate"]),f"{int(r['marker+_total(staining)']):,}",f"{int(r['cluster_size']):,}",
                 f"{r['recovery_%']}",f"{r['purity_%']}"])
st.append(tbl(rows,[4.4*cm,4.2*cm,1*cm,2*cm,1.7*cm,2*cm,1.7*cm],fs=7.2))
st.append(P("Recovery% is modest for broadly-detected markers (CD235a, CD271, CD38, CD31, Tryptase, NPM1mut, CD34) "
            "because those stain many cells across lineages; purity% shows how specific each cluster is.",SMALL))

st.append(PageBreak())
st.append(P("Final roster",H2))
rows=[["int","cell type","coarse","n"]]+[[str(r.v76_label_int),r.v76_label,r.v76_coarse,f"{int(r.n_cells):,}"] for r in key.itertuples()]
st.append(tbl(rows,[1*cm,10.5*cm,3.2*cm,2*cm],fs=7.8))
st.append(PageBreak())
st.append(P("Combined heatmap — selected markers",H2)); st.append(img(HM/"all_groups_combined_heatmap_selected_markers.png"))
st.append(PageBreak())
st.append(P("Combined heatmap — all markers",H2)); st.append(img(HM/"all_groups_combined_heatmap_all_markers.png"))
st.append(PageBreak())
st.append(P("Folder contents",H2))
for t in ["<b>csv/</b> cell_assignments.csv, final_label_integer_key.csv, <b>recovery_quantification.csv</b>, recovery_counts.csv.",
          "<b>heatmaps_final/</b> selected + all-marker heatmaps (+matrices).",
          "<b>roi_assignments_final/</b> 49 per-ROI CSVs · <b>spatial_overlays/</b> 49 maps + 49 recovery panels.",
          "<b>Nandhu_final_v7.6.ipynb</b>, build_v7.6.py, make_spatial_overlays.py, make_summary_pdf.py."]:
    st.append(B(t))
SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=1.5*cm,rightMargin=1.5*cm,topMargin=1.4*cm,bottomMargin=1.4*cm).build(st)
print("wrote",OUT)
