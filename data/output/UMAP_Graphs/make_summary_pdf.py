"""Summary PDF for UMAP_Graphs. Reads the CSVs and figures written by umap_graphs.py."""
import os, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak
from PIL import Image as PImage
D=os.path.dirname(os.path.abspath(__file__))
ss=getSampleStyleSheet()
H1=ParagraphStyle("H1",parent=ss["Heading1"],fontSize=17,textColor=colors.HexColor("#1c3f6e"),spaceAfter=4)
H2=ParagraphStyle("H2",parent=ss["Heading2"],fontSize=12.5,textColor=colors.HexColor("#1c5cab"),spaceBefore=10,spaceAfter=4)
B=ParagraphStyle("B",parent=ss["BodyText"],fontSize=9.4,leading=13.2,spaceAfter=5)
BL=ParagraphStyle("BL",parent=B,leftIndent=12)
SM=ParagraphStyle("SM",parent=B,fontSize=8,leading=10.5,textColor=colors.HexColor("#52514e"))
def P(t,s=B): return Paragraph(t,s)
def bul(t): return Paragraph("• "+t,BL)
def img(name,w=17.5*cm,maxh=21*cm):
    p=f"{D}/{name}"; iw,ih=PImage.open(p).size; W,H=w,w*ih/iw
    if H>maxh: H,W=maxh,maxh*iw/ih
    return Image(p,width=W,height=H)
def tbl(rows,colw,fs=7.6):
    t=Table(rows,colWidths=colw,repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),fs),("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#c3c2b7")),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1c5cab")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)])); return t
def pf(x): return "<0.001" if x<0.001 else f"{x:.3f}"
roi=pd.read_csv(f"{D}/celltype_composition_group_stats.csv").set_index("cell_type")
pat=pd.read_csv(f"{D}/celltype_composition_patient_level_stats.csv").set_index("cell_type")
pc=pd.read_csv(f"{D}/celltype_composition_per_patient_percent.csv")
rows=[["Cell type","NR % (ROI)","R % (ROI)","ROI BH p","NR % (pt)","R % (pt)","patient p","patient BH"]]
for t in pat.index:
    a,b=roi.loc[t],pat.loc[t]
    rows.append([t,f"{a['nonrelapse_mean_%']:.1f}",f"{a['relapse_mean_%']:.1f}",pf(a.p_adj),f"{b['nonrelapse_mean_%']:.1f}",f"{b['relapse_mean_%']:.1f}",pf(b.p),pf(b.p_adj)])
prow=[["Patient","Group","Stratification","Acquisition","ROIs","Cells"]]+[[r.patient,r.group,r.stratification,r.acquisition,int(r.n_ROIs),f"{int(r.n_cells):,}"] for _,r in pc.iterrows()]

st=[P("UMAP projections &amp; cell-type composition",H1),
    P("Nandhu_version_final · 233,949 cells · 49 ROIs · 11 patients (6 non-relapse, 5 relapse) · 2026-11-11",SM),
    P("Method",H2),
    bul("<b>Input.</b> The 25 selected markers in final_adata.X, double z-normalised per machine. These are the same values the cell types "
        "were annotated from. <b>No clustering was run.</b> Every cell keeps its Nandhu_version_final label."),
    bul("<b>Batch correction for the picture only.</b> On the uncorrected UMAP, Machine A and Machine B cells fall in largely separate "
        "halves (fig 4, top). Machine is partly tied to outcome (relapse ROIs are 22 Machine A + 4 Machine B), so an uncorrected UMAP "
        "would partly show machine instead of biology. The 15 principal components were therefore Harmony-corrected for machine "
        "before UMAP (umap-learn, 15 neighbours, min_dist 0.3, seed 0). After correction the two machines overlap (fig 4, bottom). "
        "Both embeddings are saved in umap_coordinates.csv."),
    bul("<b>Composition statistics.</b> Each cell type's % of all cells is compared per ROI (23 vs 26) and per patient "
        "(each patient = mean of its ROIs, 6 vs 5), using Mann–Whitney with BH across cell types."),
    P("Key observations",H2),
    bul("<b>The cell types form coherent, separate regions</b> on the corrected UMAP (fig 1): granulocytes, erythroid, endothelium, T cells, "
        "plasma cells and mast cells each occupy their own area, and the marker plots (fig 5) line up with the labels. "
        "NPM1 blasts appear in two regions, consistent with heterogeneous blasts: the recovered tumour includes the former "
        "'leukemic erythroid' and immature/monocytic blast groups of v2."),
    bul("<b>Relapse cells pile up in the blast regions; non-relapse cells in normal-haematopoiesis regions</b> (figs 2–3): granulocytic, erythroid, "
        "endothelial and mast-cell areas. The differences are in which regions are populated, not in where the cell types sit."),
    bul("<b>Two composition differences hold at both ROI and patient level:</b> NPM1 blasts (about 9% vs 27%; patient BH 0.041) and "
        "endothelial cells (about 9% vs 2%; patient BH 0.041). Everything else, including CD4/CD8 T cells, macrophages and monocytes, is not "
        "different per patient (figs 7, 9)."),
    bul("<b>Large differences between patients</b> (fig 8). 'Immature myeloid CD11c+' is concentrated in two non-relapse patients (NR-P3 15.8%, "
        "NR-P4 9.0%). R-P4 (2 ROIs) is dominated by plasma cells. NR-P6 is CD8+ T rich. These patient-specific features explain "
        "why several ROI-level differences do not survive per patient."),
    P("Patients",H2),tbl(prow,[1.6*cm,2.2*cm,6.4*cm,3.4*cm,1.2*cm,2*cm]),PageBreak(),
    img("fig1_umap_celltype.png",17.5*cm,15*cm),PageBreak(),
    img("fig2_umap_relapse_vs_nonrelapse.png",17.5*cm,7*cm),img("fig3_umap_density_difference.png",12*cm,10.5*cm),PageBreak(),
    img("fig4_umap_machine_check.png",15*cm,14*cm),PageBreak(),img("fig5_umap_markers.png",17.5*cm,14*cm),PageBreak(),
    img("fig10_umap_per_patient.png",17.5*cm,14*cm),PageBreak(),
    P("Cell-type composition",H2),img("fig8_composition_per_patient.png",17.5*cm,8.5*cm),img("fig6_composition_per_ROI.png",17.5*cm,8*cm),PageBreak(),
    tbl(rows,[5.6*cm,1.6*cm,1.6*cm,1.6*cm,1.6*cm,1.6*cm,1.7*cm,1.7*cm]),PageBreak(),
    img("fig9_composition_patient_level.png",17.5*cm,14*cm),PageBreak(),img("fig7_composition_group_boxplots.png",17.5*cm,14*cm),
    P("Interpretation",H2),
    P("At the whole-tissue level, relapse and non-relapse MRD marrow differ mainly in two things. Relapse marrow carries about three times more NPM1-mutant blasts, "
      "and non-relapse marrow has about four times more endothelium. The phenotypes of the other cell types are largely shared. Relapse is "
      "therefore marked by the expansion of the blast compartment and the loss of the vascular niche, rather than by new cell states. "
      "This agrees with the CN and proximity analyses. The UMAP also shows large differences between individual patients, so with "
      "6 vs 5 patients only effects as large as these two reach significance. Smaller shifts need a larger cohort."),
    P("Files",H2),
    P("<b>umap_graphs.py / UMAP_Graphs.ipynb</b> is the full analysis (identical code). <b>umap_coordinates.csv</b> has per-cell coordinates "
      "(Harmony-corrected and uncorrected). The composition CSVs cover per ROI, per patient, and group statistics. "
      "The patient mapping is <b>../roi_patient_mapping.csv</b>.",SM)]
SimpleDocTemplate(f"{D}/UMAP_Graphs_summary.pdf",pagesize=A4,leftMargin=1.5*cm,rightMargin=1.5*cm,topMargin=1.5*cm,bottomMargin=1.5*cm,
                  title="UMAP projections and cell-type composition — Nandhu_version_final").build(st)
print("wrote PDF")
