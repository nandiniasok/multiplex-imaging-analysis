"""Summary PDF for CN_final. Reads the CSVs and figures written by build_CN_final.py."""
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
MONO=ParagraphStyle("MONO",parent=SM,fontName="Courier",fontSize=7.6,leading=9.6,textColor=colors.black)
def P(t,s=B): return Paragraph(t,s)
def bul(t): return Paragraph("• "+t,BL)
def img(p,w=17.5*cm,maxh=21*cm):
    iw,ih=PImage.open(p).size; W,H=w,w*ih/iw
    if H>maxh: H,W=maxh,maxh*iw/ih
    return Image(p,width=W,height=H)
def tbl(rows,colw,fs=7.6):
    t=Table(rows,colWidths=colw,repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),fs),("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#c3c2b7")),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1c5cab")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)])); return t
def pf(x): return "<0.001" if x<0.001 else f"{x:.3f}"
def cn_table(k):
    s=pd.read_csv(f"{D}/k{k}/CN_group_stats.csv"); pl=pd.read_csv(f"{D}/k{k}/CN_group_stats_patient_level.csv").set_index("CN")
    nm=pd.read_csv(f"{D}/k{k}/CN_names.csv").set_index("CN")
    rows=[["CN","Name (top enriched types)","cells","NR %","R %","ROI BH p","patient p","patient BH"]]
    for _,r in s.iterrows():
        q=pl.loc[r.CN]; rows.append([r.CN,r["name"],f"{int(nm.loc[r.CN,'n_cells']):,}",f"{r['nonrelapse_%']:.1f}",f"{r['relapse_%']:.1f}",pf(r.p_adj),pf(q.p_patient),pf(q.p_adj)])
    return tbl(rows,[1.1*cm,6.2*cm,1.6*cm,1.3*cm,1.3*cm,1.7*cm,1.7*cm,1.7*cm])

st=[P("Cellular neighbourhood (CN) analysis",H1),
    P("Nandhu_version_final cell types · Nolan-lab method (Schürch et al., Cell 2020) · k = 10 and 20 · 10 CNs · "
      "49 ROIs from 11 patients (6 non-relapse, 5 relapse) · 2026-11-11",SM),
    P("Method",H2),
    bul("<b>Windows.</b> For each cell, the window is its k nearest cells in the same ROI, including itself. Both k = 10 "
        "(about the immediate 1–2 cell-layer neighbourhood) and k = 20 (a wider niche) were run."),
    bul("<b>Neighbourhoods.</b> The cell-type fractions in each window are clustered with MiniBatchKMeans into 10 CNs (random_state 0), "
        "following the NeighborhoodCoordination repository."),
    bul("<b>Cell types used.</b> 17 annotated types from Nandhu_version_final; tumour = NPM1_blasts. Unassigned/Other and Artifact "
        "(15,462 cells) are excluded, as you chose. CNs are numbered by size (CN0 = largest). Each CN is named after its two most "
        "enriched cell types among those making up at least 5% of it."),
    bul("<b>Statistics.</b> CN abundance is the % of a ROI's cells. Non-relapse and relapse are compared per ROI (23 vs 26, Mann–Whitney, BH) "
        "and per patient (each patient = mean of its ROIs, 6 vs 5). The patient-level test is the stricter one. "
        "With 6 vs 5 patients, the smallest possible p is 0.0043."),
    P("Key findings",H2),
    bul("<b>A tumour-dominated neighbourhood marks relapse.</b> At k = 10, CN0 (NPM1 blasts + Granulocyte) covers 30.3% of relapse "
        "versus 7.1% of non-relapse tissue (ROI BH 0.001). At k = 20 the tumour niche splits into two CNs: CN1 (NPM1 blasts + "
        "Macrophages) and CN2 (NPM1 blasts + Granulocyte). Both are expanded in relapse (ROI BH 0.004 and 0.002)."),
    bul("<b>A vascular neighbourhood marks non-relapse.</b> The endothelial CN (CN7 at both k) covers 13.2% vs 1.5% of tissue (k = 10) and "
        "17.2% vs 1.4% (k = 20), ROI BH ≤ 0.0003. This is the CN counterpart of the endothelial result in the proximity analysis."),
    bul("<b>Healthy-haematopoiesis niches lean towards non-relapse but do not reach significance.</b> These are Granulocyte (CN3 at k = 10, CN0 at k = 20) and "
        "Erythroid (CN4 at k = 10, CN3 at k = 20), with raw p 0.03–0.06."),
    bul("<b>Patient-level confirmation.</b> With patients as the unit (6 vs 5), the blast + granulocyte CN (CN0 at k = 10, CN2 at k = 20) and the "
        "endothelial CN (CN7 at both k) remain significant (patient BH 0.022 each). The k = 20 blast + macrophage CN is borderline "
        "(patient p 0.052). Nothing else differs per patient."),
    P("k = 10",H2),cn_table(10),PageBreak(),
    img(f"{D}/k10/fig1_CN_enrichment_heatmap.png",16*cm,13*cm),img(f"{D}/k10/fig4_CN_abundance_group.png",17.5*cm,7.5*cm),PageBreak(),
    img(f"{D}/k10/fig2_celltype_within_CN.png",17.5*cm,9*cm),img(f"{D}/k10/fig3_CN_frequency_per_ROI.png",17.5*cm,8*cm),PageBreak(),
    P("k = 20",H2),cn_table(20),
    img(f"{D}/k20/fig1_CN_enrichment_heatmap.png",16*cm,13*cm),PageBreak(),
    img(f"{D}/k20/fig4_CN_abundance_group.png",17.5*cm,7.5*cm),img(f"{D}/k20/fig2_celltype_within_CN.png",17.5*cm,9*cm),PageBreak(),
    img(f"{D}/k20/fig3_CN_frequency_per_ROI.png",17.5*cm,8*cm),
    P("Overall cell-type composition by group",H2),img(f"{D}/fig_celltype_fraction_by_group.png",11*cm,10*cm),PageBreak(),
    P("Example CN maps (k = 10)",H2),
    img(f"{D}/k10/cn_maps/Machine_A_Slides2_ROI_04_CN_k10.png",14*cm,10*cm),img(f"{D}/k10/cn_maps/Machine_B_SN177_ROI_11_CN_k10.png",14*cm,10*cm),
    P("All 49 ROIs, for both k values, are in <b>k10/cn_maps/</b> and <b>k20/cn_maps/</b>.",SM),PageBreak(),
    P("Interpretation",H2),
    P("The CN analysis gives the same tissue-level picture as the proximity analysis, using an independent method. In relapse marrow, a large share of the tissue is "
      "organised into blast-dominated neighbourhoods where blasts mix with granulocytes and macrophages. In non-relapse marrow, blasts are a minority "
      "and the tissue keeps a vascular (endothelial) neighbourhood and more normal myelo-erythroid niches. The endothelial-niche loss is "
      "the most consistent difference across the CN, proximity and composition analyses. A structured vascular niche may help keep residual "
      "disease under control. Its loss, or its replacement by blast-dominated tissue, accompanies relapse."),
    P("Clinically, the fraction of tissue in the blast CN and in the vascular CN are simple, per-biopsy spatial read-outs. They "
      "could complement MRD transcript levels for relapse prediction. This needs confirmation in an independent cohort, because the ROIs come "
      "from 11 patients and machine is partly confounded with group (relapse ROIs are mostly Machine A)."),
    P("Viewing the CNs in MACSiQ View",H2),
    P("<b>MACSiQ_import/</b> holds one CSV per ROI (49 files). Each file has every cell of that ROI and these columns:",B),
    P('"Cell Id","Nuc X","Nuc Y Inv","CN_k10","CN_k20","CellType_int","Relapse"',MONO),
    bul("<b>Cell Id</b> is exactly as exported from MACSiQ View (for example \"1@1\"). This is the key MACSiQ View matches on. "
        "<b>Nuc X / Nuc Y Inv</b> are included so you can check that the right ROI is matched."),
    bul("<b>CN_k10 / CN_k20</b> are 0–9, or −1 for Unassigned/Artifact cells. <b>CellType_int</b> = final_label_int. <b>Relapse</b> = 1 for relapse, 0 for non-relapse. "
        "All values are numbers, so MACSiQ View treats them as features you can gate on. "
        "<b>_LEGEND_value_meanings.csv</b> lists what each number means."),
    P("Steps (menu names can differ between MACSiQ View versions; check the software manual for 'import features' if a label does not match):",B),
    P("1. Open the ROI's analysis in MACSiQ View, the same segmentation the original CSV was exported from.<br/>"
      "2. Use the feature/data <b>import</b> option, choose that ROI's file (e.g. <i>Machine_A_Slides1_ROI_01_CN.csv</i>), "
      "and match on <b>Cell Id</b>.<br/>"
      "3. The new columns appear as cell features. Make a histogram or scatter of <b>CN_k10</b>, gate one value at a time "
      "(e.g. 6.5–7.5 for CN7) and colour that gate on the image overlay.<br/>"
      "4. Compare the overlay with <b>k10/cn_maps/&lt;ROI&gt;_CN_k10.png</b>. The patterns should match. If they look mirrored, the "
      "file was imported into the wrong ROI.",BL),
    P("Files",H2),
    P("<b>build_CN_final.py / CN_final.ipynb</b> is the full analysis (identical code). <b>k10/</b> and <b>k20/</b> hold cells_with_CN.csv, CN names, "
      "enrichment, composition, stats (ROI-level and patient-level), figs 1–4 and cn_maps/. <b>MACSiQ_import/</b> holds the per-ROI import CSVs and the legend.",SM)]
SimpleDocTemplate(f"{D}/CN_final_summary.pdf",pagesize=A4,leftMargin=1.5*cm,rightMargin=1.5*cm,topMargin=1.5*cm,bottomMargin=1.5*cm,
                  title="Cellular neighbourhood analysis — Nandhu_version_final").build(st)
print("wrote PDF")
