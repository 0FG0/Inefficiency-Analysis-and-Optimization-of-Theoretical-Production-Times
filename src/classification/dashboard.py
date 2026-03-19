"""
dashboard.py  —  Koepfer 160-2 Anomaly Dashboard
Genera outputs/dashboard_anomaly.html
Uso: python src/classification/dashboard.py
"""

import os, sys, json, warnings
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
warnings.filterwarnings("ignore")

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feature_engineering import pipeline_classificazione

DATA_PATH   = os.path.join(PROJECT_ROOT, "data",   "processed",     "koepfer_160_2.csv")
MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "classification", "best_classificazione_anomaly_BD.pkl")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "classification", "parametri_classificazione_anomaly_BD.pkl")
OUTPUT_HTML = os.path.join(PROJECT_ROOT, "outputs", "dashboard_anomaly.html")


def carica_e_prepara(path, params):
    df = pd.read_csv(path)
    if "ARTICOLO" in df.columns:
        df["ARTICOLO"] = df["ARTICOLO"].fillna("MISSING_ARTICOLO").astype(str)
    articoli_top = params.get("articoli_top", [])
    df["ARTICOLO_grouped"] = df["ARTICOLO"].where(df["ARTICOLO"].isin(articoli_top), other="ALTRO")
    df["ARTICOLO_grouped"] = df["ARTICOLO_grouped"].fillna("ALTRO").astype(str)
    df = pipeline_classificazione(df)
    soglia_anomalia   = params.get("soglia_anomalia",   df["Indice_Inefficienza"].quantile(0.85))
    soglia_attenzione = params.get("soglia_attenzione", df["Indice_Inefficienza"].quantile(0.60))
    df["classe_vera"] = (df["Indice_Inefficienza"] > soglia_anomalia).astype(int)
    df["classe_vera_tri"] = np.where(
        df["Indice_Inefficienza"] >= soglia_anomalia, 2,
        np.where(df["Indice_Inefficienza"] >= soglia_attenzione, 1, 0)
    )
    if "Data_Ora_Fine" in df.columns:
        df["Data_Ora_Fine"] = pd.to_datetime(df["Data_Ora_Fine"], errors="coerce")
    return df


COLS_TO_DROP = [
    "classe_vera", "Indice_Inefficienza", "Tempo Lavoraz. ORE",
    "Buon Tempo Ciclo ORE", "WO", "ARTICOLO", "Descrizione Articolo",
    "ID DAD", "Descrizione Macchina", "C.d.L. Effett", "Data_Ora_Fine",
]

def predici(model, df):
    X = df.drop(columns=COLS_TO_DROP, errors="ignore")
    for col in ["ARTICOLO_grouped","FASE","Cod CIC","C.d.L. Prev","Descrizione Centro di Lavoro previsto"]:
        if col in X.columns:
            X[col] = X[col].astype("string").fillna("MISSING")
    df = df.copy()

    preds = model.predict(X)
    proba = model.predict_proba(X)
    n_classes = proba.shape[1]

    # classe_pred_tri: predizione triclasse grezza (0=NORMALE, 1=ATTENZIONE, 2=ANOMALIA)
    # usata per mostrare lo stato nella tabella WO
    df["classe_pred_tri"] = preds if n_classes == 3 else preds

    # classe_pred binario: 1 solo se ANOMALIA vera (classe 2), per KPI e grafici
    df["classe_pred"]   = (preds == 2).astype(int) if n_classes == 3 else preds
    df["prob_anomalia"] = proba[:, 2] if n_classes == 3 else proba[:, 1]

    return df

def _safe(v):
    if isinstance(v, np.integer):  return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.bool_):    return bool(v)
    return v

def kpi_globali(df):
    if "Data_Ora_Fine" in df.columns:
        d0, d1 = df["Data_Ora_Fine"].min(), df["Data_Ora_Fine"].max()
        fmt = '%b %Y'
        periodo = f"{d0.strftime(fmt)} – {d1.strftime(fmt)}" if pd.notna(d0) else "Dataset completo"
    else:
        periodo = "Dataset completo"
    n_tot      = len(df)
    n_anomalie = int((df["classe_vera"] == 1).sum())
    n_normali  = int((df["classe_vera"] == 0).sum())
    tasso      = round(n_anomalie / max(n_tot, 1) * 100, 1)
    ore_perse  = 0.0
    if "Tempo Lavoraz. ORE" in df.columns and "Buon Tempo Ciclo ORE" in df.columns:
        ore_perse = round(float((df["Tempo Lavoraz. ORE"] - df["Buon Tempo Ciclo ORE"]).clip(lower=0).sum()), 1)
    delta_anomalie = "—"
    if "Data_Ora_Fine" in df.columns:
        periodi = sorted(df["Data_Ora_Fine"].dt.to_period("M").dropna().unique())
        if len(periodi) >= 2:
            u, p = periodi[-1], periodi[-2]
            nu  = int((df[df["Data_Ora_Fine"].dt.to_period("M")==u]["classe_vera"]==1).sum())
            np_ = int((df[df["Data_Ora_Fine"].dt.to_period("M")==p]["classe_vera"]==1).sum())
            diff = nu - np_
            delta_anomalie = f"{'+' if diff>=0 else ''}{diff} vs {str(p)}"
    return dict(periodo=periodo, n_tot=n_tot, n_anomalie=n_anomalie, n_normali=n_normali,
                tasso=tasso, ore_perse=ore_perse, delta_anomalie=delta_anomalie,
                generato_il=datetime.now().strftime("%d/%m/%Y %H:%M"))

def calendario_tutti_mesi(df):
    if "Data_Ora_Fine" not in df.columns: return []
    MESI = {1:"Gen",2:"Feb",3:"Mar",4:"Apr",5:"Mag",6:"Giu",7:"Lug",8:"Ago",9:"Set",10:"Ott",11:"Nov",12:"Dic"}
    df = df.copy()
    df["_p"] = df["Data_Ora_Fine"].dt.to_period("M")
    df["_g"] = df["Data_Ora_Fine"].dt.day
    result = []
    for p in sorted(df["_p"].dropna().unique(), reverse=True):
        sub = df[df["_p"]==p]
        ga  = set(sub.loc[sub["classe_vera_tri"]==2,"_g"].dropna().astype(int))
        gat = set(sub.loc[sub["classe_vera_tri"]==1,"_g"].dropna().astype(int))
        gn  = set(sub.loc[sub["classe_vera_tri"]==0,"_g"].dropna().astype(int))
        giorni = [{"giorno":d,"stato":"anomaly" if d in ga else "attention" if d in gat else "normal" if d in gn else "empty"} for d in range(1, p.days_in_month+1)]
        result.append({"mese":str(p),"label":f"{MESI.get(p.month,p.month)} {p.year}","giorni":giorni})
    return result

def wo_per_giorno(df):
    if "Data_Ora_Fine" not in df.columns: return {}
    df = df.copy()
    df["_d"] = df["Data_Ora_Fine"].dt.strftime("%Y-%m-%d")
    result = {}
    for ds, g in df.groupby("_d"):
        records = []
        for _, row in g.iterrows():
            tri = int(row.get("classe_vera_tri", row.get("classe_vera", 0)) or 0)
            if tri == 2:
                stato = "ANOMALIA"
            elif tri == 1:
                stato = "ATTENZIONE"
            else:
                stato = "NORMALE"
            orer = row.get("Tempo Lavoraz. ORE", None)
            oret = row.get("Buon Tempo Ciclo ORE", None)
            records.append({"WO":str(row.get("WO","—") or "—"),
                            "articolo":str(row.get("ARTICOLO",row.get("ARTICOLO_grouped","—")) or "—"),
                            "fase":str(row.get("FASE","—") or "—"),"stato":stato,
                            "ore_reali":round(float(orer),2) if pd.notna(orer) else None,
                            "ore_teoriche":round(float(oret),2) if pd.notna(oret) else None})
        result[ds] = records
    return result

def tabella_wo_tutti(df, n=99999, soglia_attenzione=1.1511, soglia_anomalia=1.4220):
    cols = [c for c in ["WO","ARTICOLO","ARTICOLO_grouped","FASE","C.d.L. Prev","Data_Ora_Fine",
                         "Tempo Lavoraz. ORE","Buon Tempo Ciclo ORE","Indice_Inefficienza",
                         "prob_anomalia","classe_vera"] if c in df.columns]
    sub = df[cols].copy()
    if "Data_Ora_Fine" in sub.columns:
        sub = sub.sort_values("Data_Ora_Fine", ascending=False)
    sub = sub.head(n)

    def _s(row):
        idx = row.get("Indice_Inefficienza", None)
        if idx is not None and pd.notna(idx):
            v = float(idx)
            if v >= soglia_anomalia:
                return "ANOMALIA"
            if v >= soglia_attenzione:
                return "ATTENZIONE"
        return "NORMALE"

    sub["stato"] = sub.apply(_s, axis=1)
    if "Data_Ora_Fine" in sub.columns:
        sub["_d"] = sub["Data_Ora_Fine"].dt.strftime("%Y-%m-%d")
        sub["Data_Ora_Fine"] = sub["Data_Ora_Fine"].dt.strftime("%d/%m/%Y")
    for c, r in [("Indice_Inefficienza",3),("Tempo Lavoraz. ORE",2),("Buon Tempo Ciclo ORE",2)]:
        if c in sub.columns: sub[c] = sub[c].round(r)
    if "prob_anomalia" in sub.columns: sub["prob_anomalia"] = (sub["prob_anomalia"]*100).round(2)
    return [{k:_safe(v) for k,v in row.items()} for row in sub.to_dict(orient="records")]

def anomalie_per_articolo(df, n=20):
    if "ARTICOLO_grouped" not in df.columns: return {"labels":[],"values":[],"counts":[]}
    g = df.groupby("ARTICOLO_grouped").agg(totale=("classe_vera","count"),anomalie=("classe_vera","sum"))
    g["tasso"] = (g["anomalie"]/g["totale"]*100).round(1)
    g = g[g.index != "ALTRO"]
    g = g[g["totale"]>=3].sort_values("tasso",ascending=False).head(n)
    return {"labels":g.index.tolist(),"values":g["tasso"].tolist(),"counts":g["anomalie"].astype(int).tolist()}

def anomalie_per_fase(df):
    if "FASE" not in df.columns: return {"labels":[],"values":[]}
    g = df.groupby("FASE").agg(totale=("classe_vera","count"),anomalie=("classe_vera","sum"))
    g["tasso"] = (g["anomalie"]/g["totale"]*100).round(1)
    g = g[g["totale"]>=3].sort_values("tasso",ascending=False).head(6)
    return {"labels":g.index.tolist(),"values":g["tasso"].tolist()}

def fase_per_mese(df):
    if "FASE" not in df.columns or "Data_Ora_Fine" not in df.columns: return {}
    df = df.copy()
    df["_m"] = df["Data_Ora_Fine"].dt.to_period("M")
    result = {}
    for p in sorted(df["_m"].dropna().unique())[-6:]:
        sub = df[df["_m"]==p]
        g = sub.groupby("FASE").agg(totale=("classe_vera","count"),anomalie=("classe_vera","sum"))
        g["tasso"] = (g["anomalie"]/g["totale"]*100).round(1)
        g = g[g["totale"]>=1].sort_values("tasso",ascending=False).head(6)
        result[str(p)] = {"labels":g.index.tolist(),"values":g["tasso"].tolist()}
    return result

def _fmt_mese(s):
    p = s.split("-")
    return f"{p[1]}-{p[0][2:]}" if len(p)==2 else s

def sparkline_ore_perse(df):
    if "Data_Ora_Fine" not in df.columns: return {"labels":[],"values":[],"keys":[]}
    df = df.copy()
    if "Tempo Lavoraz. ORE" in df.columns and "Buon Tempo Ciclo ORE" in df.columns:
        df["_op"] = (df["Tempo Lavoraz. ORE"]-df["Buon Tempo Ciclo ORE"]).clip(lower=0)
    else:
        df["_op"] = pd.Series(0.0, index=df.index)
    df["_m"] = df["Data_Ora_Fine"].dt.to_period("M")
    g = df.groupby("_m")["_op"].sum().sort_index().tail(6)
    return {"labels":[_fmt_mese(str(p)) for p in g.index],"keys":[str(p) for p in g.index],
            "values":[round(float(v),1) for v in g.values]}

def forecast_mese_successivo(df, articoli_top=None):
    if "ARTICOLO_grouped" not in df.columns:
        return []
    if articoli_top:
        sub = df[df["ARTICOLO_grouped"].isin(articoli_top)]
    else:
        sub = df[df["ARTICOLO_grouped"] != "ALTRO"]
    if sub.empty:
        return []
    group_cols = ["ARTICOLO_grouped"] + (["FASE"] if "FASE" in df.columns else [])
    g = sub.groupby(group_cols).agg(
        prob_media=("prob_anomalia","mean"),
        n_lavorazioni=("prob_anomalia","count")
    ).reset_index()
    g["prob_media"] = (g["prob_media"]*100).round(0).astype(int)
    g = g[g["prob_media"] > 14].sort_values("prob_media", ascending=False)
    result = []
    for _, row in g.iterrows():
        prob = float(row["prob_media"])
        risk_cls = "risk-high" if prob>=60 else "risk-medium" if prob>=35 else "risk-low"
        entry = {"ARTICOLO_grouped":str(row["ARTICOLO_grouped"]),
                 "prob":prob,"risk_cls":risk_cls,"n":int(row["n_lavorazioni"])}
        if "FASE" in g.columns:
            entry["FASE"] = str(row["FASE"]) if pd.notna(row.get("FASE")) else "—"
        result.append(entry)
    return result


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Anomaly Dashboard - Koepfer 160-2</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {
    --bg:#0c0e10; --surface:#13171c; --border:#1f2730; --border2:#2a3442;
    --text:#e2e8f0; --muted:#5a6880;
    --amber:#f59e0b; --amber-dim:#7c4f06;
    --red:#ef4444;   --red-dim:#7f1d1d;
    --green:#22c55e; --green-dim:#14532d;
    --blue:#38bdf8;  --blue-dim:#0c4a6e;
    --mono:'DM Mono',monospace; --sans:'DM Sans',sans-serif; --radius:12px;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;min-height:100vh;overflow-x:hidden}
  body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");opacity:.6}
  main{position:relative;z-index:1;max-width:1360px;margin:0 auto;padding:0 2.5rem 4rem}
  header{display:flex;align-items:flex-end;justify-content:space-between;padding:2.5rem 0 1.75rem;border-bottom:1px solid var(--border);margin-bottom:2.5rem;gap:1rem;flex-wrap:wrap}
  .h-eyebrow{font-family:var(--mono);font-size:.68rem;letter-spacing:.16em;color:var(--amber);text-transform:uppercase;margin-bottom:.5rem}
  header h1{font-size:2rem;font-weight:800;letter-spacing:-.02em;line-height:1}
  header h1 em{color:var(--amber);font-style:normal}
  .h-right{text-align:right}
  .h-right .gen{font-family:var(--mono);font-size:.7rem;color:var(--muted)}
  .h-badge{display:inline-flex;align-items:center;gap:.4rem;margin-top:.5rem;background:var(--green-dim);border:1px solid var(--green);color:var(--green);font-family:var(--mono);font-size:.68rem;padding:.28rem .7rem;border-radius:999px;letter-spacing:.08em}
  .h-badge .dot{width:6px;height:6px;background:var(--green);border-radius:50%;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
  .s-title{font-family:var(--mono);font-size:.65rem;letter-spacing:.18em;color:var(--muted);text-transform:uppercase;display:flex;align-items:center;gap:.75rem;margin-bottom:1.25rem}
  .s-title::after{content:'';flex:1;height:1px;background:var(--border)}
  .kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1px;background:var(--border);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:2.5rem}
  .kpi-card{background:var(--surface);padding:1.5rem 1.25rem;position:relative;overflow:hidden;transition:background .18s}
  .kpi-card:hover{background:#181e26}
  .kpi-card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px}
  .kpi-card.c-amber::after{background:var(--amber)}.kpi-card.c-red::after{background:var(--red)}.kpi-card.c-green::after{background:var(--green)}.kpi-card.c-blue::after{background:var(--blue)}
  .kpi-lbl{font-family:var(--mono);font-size:.975rem;letter-spacing:.06em;color:var(--muted);text-transform:uppercase;margin-bottom:.55rem}
  .kpi-val{font-family:var(--sans);font-size:2.6rem;font-weight:800;line-height:1;letter-spacing:-.03em}
  .c-amber .kpi-val{color:var(--amber)}.c-red .kpi-val{color:var(--red)}.c-green .kpi-val{color:var(--green)}.c-blue .kpi-val{color:var(--blue)}
  .kpi-sub{font-family:var(--mono);font-size:1.02rem;color:var(--muted);margin-top:.45rem}
  .kpi-delta{display:inline-flex;align-items:center;gap:.18rem;font-family:var(--mono);font-size:.975rem;margin-top:.35rem}
  .kpi-delta.up{color:var(--red)}.kpi-delta.down{color:var(--green)}.kpi-delta.neutral{color:var(--muted)}
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:2rem}
  @media(max-width:840px){.two-col{grid-template-columns:1fr}}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;animation:fadeUp .45s ease both}
  @keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
  .cal-scroll{height:260px;overflow-y:auto;overflow-x:hidden;padding-right:4px;scrollbar-width:thin;scrollbar-color:var(--border2) transparent}
  .cal-month{margin-bottom:.9rem}.cal-month:last-child{margin-bottom:0}
  .cal-month-label{font-family:var(--mono);font-size:.6rem;letter-spacing:.12em;color:var(--amber);text-transform:uppercase;margin-bottom:.35rem}
  .cal-grid{display:flex;gap:3px;flex-wrap:wrap}
  .cal-day{width:15px;height:15px;border-radius:3px;cursor:default;flex-shrink:0;transition:transform .12s}
  .cal-day:hover{transform:scale(1.45);z-index:10;position:relative}
  .cal-day.normal{background:var(--green-dim);border:1px solid var(--green)}.cal-day.attention{background:var(--amber-dim);border:1px solid var(--amber)}.cal-day.anomaly{background:var(--red-dim);border:1px solid var(--red)}.cal-day.empty{background:var(--border)}
  .cal-legend{display:flex;gap:1.2rem;margin-top:.75rem}
  .cl-item{display:flex;align-items:center;gap:.38rem;font-family:var(--mono);font-size:.65rem;color:var(--muted)}
  .cl-box{width:11px;height:11px;border-radius:2px;flex-shrink:0}
  .bar-list{display:flex;flex-direction:column;gap:.7rem}
  .bar-row{display:flex;align-items:center;gap:.7rem}
  .bar-lbl{font-family:var(--mono);font-size:.68rem;color:var(--muted);width:86px;flex-shrink:0;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .bar-track{flex:1;background:var(--border);border-radius:4px;height:20px;overflow:hidden}
  .bar-fill{height:100%;border-radius:4px;display:flex;align-items:center;padding-left:.45rem;font-family:var(--mono);font-size:.65rem;color:#0c0e10;font-weight:500;transition:width 1.1s cubic-bezier(.22,1,.36,1)}
  .bar-fill.c-red{background:var(--red)}.bar-fill.c-amber{background:var(--amber)}.bar-fill.c-green{background:var(--green)}
  .bar-pct{font-family:var(--mono);font-size:.65rem;color:var(--muted);width:34px;text-align:right;flex-shrink:0}
  .spark{display:flex;align-items:flex-end;gap:4px;height:56px}
  .spark-bar{flex:1;background:var(--amber-dim);border-radius:3px 3px 0 0;transition:background .2s;cursor:pointer}
  .spark-bar.selected{background:var(--amber)}.spark-bar:hover{background:var(--amber);opacity:.85}
  .spark-labels{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.62rem;color:var(--muted);margin-top:.35rem}
  .data-table{width:100%;border-collapse:collapse;font-size:.72rem}
  .data-table th{font-family:var(--mono);font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:400;padding:.55rem .7rem;border-bottom:1px solid var(--border);text-align:left}
  #woSortRow th{font-size:.78rem}
  .data-table td{font-family:var(--mono);font-size:.82rem;padding:.6rem .7rem;border-bottom:1px solid var(--border2);vertical-align:middle;text-align:center}
  .data-table tr:last-child td{border-bottom:none}.data-table tr:hover td{background:#181e26}
  .badge{display:inline-block;padding:.12rem .5rem;border-radius:4px;font-family:var(--mono);font-size:.58rem;letter-spacing:.06em;text-transform:uppercase}
  .badge.b-red{background:var(--red-dim);color:var(--red);border:1px solid var(--red)}.badge.b-amber{background:var(--amber-dim);color:var(--amber);border:1px solid var(--amber)}.badge.b-green{background:var(--green-dim);color:var(--green);border:1px solid var(--green)}
  /* FORECAST CARD */
  .forecast-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:1rem}
  .fc-item{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:1rem 1rem .9rem}
  .fc-label{font-family:var(--mono);font-size:.65rem;color:var(--muted);letter-spacing:.04em}
  .fc-art-val{font-family:var(--mono);font-size:.88rem;font-weight:700;letter-spacing:-.01em}
  .fc-fase-val{font-family:var(--mono);font-size:.78rem;font-weight:600}
  .fc-meta{font-family:var(--mono);font-size:.62rem;color:var(--muted);margin:.35rem 0 .65rem}
  .fc-bar{background:var(--border);border-radius:3px;height:5px;margin-bottom:.6rem;overflow:hidden}
  .fc-fill{height:100%;border-radius:3px}
  .fc-prob-row{display:flex;align-items:baseline;gap:.45rem}
  .fc-prob-lbl{font-family:var(--mono);font-size:.75rem;color:var(--muted)}
  .fc-prob-val{font-family:var(--mono);font-size:1.18rem;font-weight:700;letter-spacing:-.02em;line-height:1}
  .risk-high  .fc-fill{background:var(--red)}  .risk-medium .fc-fill{background:var(--amber)}  .risk-low .fc-fill{background:var(--green)}
  .risk-high  .fc-art-val{color:var(--red)}    .risk-medium .fc-art-val{color:var(--amber)}    .risk-low .fc-art-val{color:var(--green)}
  .risk-high  .fc-fase-val{color:var(--red)}   .risk-medium .fc-fase-val{color:var(--amber)}   .risk-low .fc-fase-val{color:var(--green)}
  .risk-high  .fc-prob-val{color:var(--red)}   .risk-medium .fc-prob-val{color:var(--amber)}   .risk-low .fc-prob-val{color:var(--green)}
  .note{font-family:var(--mono);font-size:.75rem;color:var(--muted);margin-top:1rem;padding:.7rem .9rem;background:var(--bg);border-left:3px solid var(--amber);border-radius:0 4px 4px 0;line-height:1.6}
  .gauge-wrap{display:flex;justify-content:center;margin:.5rem 0 .75rem}
  ::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
  .full-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;margin-bottom:2rem;animation:fadeUp .45s ease both}
  .overflow-table{overflow-x:auto;overflow-y:auto;max-height:330px}
  .overflow-table thead th{position:sticky;background:var(--surface);z-index:2}
  .overflow-table thead tr:first-child th{top:0}
  .overflow-table thead tr:nth-child(2) th{top:33px;background:var(--bg);border-bottom:2px solid var(--border)}
  .period-btns{display:flex;gap:.5rem;margin-bottom:1.25rem;flex-wrap:wrap}
  .period-btn{background:var(--bg);border:1px solid var(--border2);color:var(--muted);font-family:var(--mono);font-size:.68rem;letter-spacing:.08em;padding:.38rem .9rem;border-radius:6px;cursor:pointer;transition:all .18s;text-transform:uppercase}
  .period-btn:hover{border-color:var(--amber);color:var(--amber)}
  .period-btn.active{background:var(--amber-dim);border-color:var(--amber);color:var(--amber)}
  .wo-period-btn{background:var(--bg);border:1px solid var(--border2);color:var(--muted);font-family:var(--mono);font-size:.68rem;letter-spacing:.08em;padding:.38rem .9rem;border-radius:6px;cursor:pointer;transition:all .18s;text-transform:uppercase}
  .wo-period-btn:hover{border-color:var(--amber);color:var(--amber)}
  .wo-period-btn.active{background:var(--amber-dim);border-color:var(--amber);color:var(--amber)}
  /* ── sort + filter ─────────────────────────────────────────────────────── */
  th[data-col]{cursor:pointer;user-select:none;white-space:nowrap}
  th[data-col]:hover{color:var(--text)}
  .sort-icon{font-size:17px;opacity:.5;margin-left:5px;display:inline-block;line-height:1;vertical-align:middle;transition:opacity .15s,color .15s}
  th[data-col]:hover .sort-icon{opacity:.8}
  th.sort-asc  .sort-icon,
  th.sort-desc .sort-icon{opacity:1;color:var(--amber);text-shadow:0 0 6px rgba(245,158,11,.6)}
  .col-filter{width:100%;background:var(--surface);border:1px solid var(--border);border-radius:4px;color:var(--text);font-family:var(--mono);font-size:.78rem;padding:3px 6px;outline:none;transition:border-color .15s}
  .col-filter:focus{border-color:var(--amber)}
  .wo-count-badge{display:inline-block;margin-left:.6rem;padding:1px 8px;border-radius:20px;font-size:.6rem;font-weight:600;background:rgba(56,189,248,0.12);color:var(--blue);vertical-align:middle;letter-spacing:.04em;font-family:var(--mono)}
</style>
</head>
<body><main>
<header>
  <div>
    <div class="h-eyebrow">Monitor Produzione &#8212; Analisi Predittiva</div>
    <h1>Koepfer 160-2 <em>/ Anomalie</em></h1>
  </div>
  <div class="h-right">
    <div class="gen">Generato il __GENERATO_IL__ &nbsp;&middot;&nbsp; __N_TOT__ lavorazioni analizzate</div>
    <div class="h-badge"><span class="dot"></span> Modello attivo</div>
  </div>
</header>
<div class="s-title">Riepilogo </div>
<div class="period-btns">
  <button class="period-btn active" id="btn_all" data-period="all" onclick="filterByPeriod(this.dataset.period)">Tutto</button>
  <button class="period-btn" id="btn_1m" data-period="1m" onclick="filterByPeriod(this.dataset.period)">Ultimo mese</button>
  <button class="period-btn" id="btn_3m" data-period="3m" onclick="filterByPeriod(this.dataset.period)">Ultimi 3 mesi</button>
  <button class="period-btn" id="btn_1y" data-period="1y" onclick="filterByPeriod(this.dataset.period)">Ultimo anno</button>
</div>
<div class="kpi-grid">
  <div class="kpi-card c-red"><div class="kpi-lbl">Anomalie rilevate</div><div class="kpi-val" id="kpiAnomalie">0</div><div class="kpi-sub" id="kpiSubAnomalie">su __N_TOT__ lavorazioni</div><div class="kpi-delta" id="kpiDelta" style="display:none"></div></div>
  <div class="kpi-card c-amber"><div class="kpi-lbl">Tasso anomalia</div><div class="kpi-val" id="kpiTasso">0%</div><div class="kpi-sub">soglia critica: 20%</div><div class="kpi-delta" id="kpiDeltaTasso" style="display:none"></div></div>
  <div class="kpi-card c-amber"><div class="kpi-lbl">Ore perse</div><div class="kpi-val" id="kpiOrePerse">0 h</div><div class="kpi-sub">tempo reale &minus; teorico</div><div class="kpi-delta" id="kpiDeltaOrePerse" style="display:none"></div></div>
  <div class="kpi-card c-green"><div class="kpi-lbl">Lavorazioni normali</div><div class="kpi-val" id="kpiNormali">0</div><div class="kpi-sub" id="kpiPercNormali">__PERC_NORMALI__% del totale</div><div class="kpi-delta" id="kpiDeltaNormali" style="display:none"></div></div>
</div>
<div class="two-col" style="grid-template-columns:1.20fr 1fr">
  <div class="card">
    <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.25rem">
      <button id="calBackBtn" onclick="calShowGrid()" style="display:none;background:none;border:none;cursor:pointer;color:var(--amber);font-size:1.35rem;padding:0;line-height:1;flex-shrink:0;opacity:.85" onmouseover="this.style.opacity=1" onmouseout="this.style.opacity='.85'">&#8592;</button>
      <div class="s-title" style="margin-bottom:0;flex:1">Calendario lavorazioni &#8212; storico completo</div>
    </div>
    <div id="calHint" style="font-family:var(--mono);font-size:.62rem;color:var(--muted);margin-bottom:.85rem;letter-spacing:.04em">Clicca su un quadratino per i dettagli del giorno</div>
    <div class="cal-scroll" id="calScroll"></div>
    <div id="calDetail" style="display:none"></div>
    <div class="cal-legend" id="calLegend">
      <div class="cl-item"><div class="cl-box" style="background:var(--green-dim);border:1px solid var(--green)"></div>Normale</div>
      <div class="cl-item"><div class="cl-box" style="background:var(--amber-dim);border:1px solid var(--amber)"></div>Attenzione</div>
      <div class="cl-item"><div class="cl-box" style="background:var(--red-dim);border:1px solid var(--red)"></div>Anomalia</div>
      <div class="cl-item"><div class="cl-box" style="background:var(--border)"></div>Nessuna prod.</div>
    </div>
  </div>
  <div class="card">
    <div class="s-title">Indice medio inefficienza</div>
    <div style="text-align:right;margin-top:-.6rem;margin-bottom:.9rem;font-family:var(--mono);font-size:.78rem;color:var(--muted);line-height:1.9">
      <div><span style="color:var(--text);font-weight:500">1.0</span> &nbsp;Tempo Reale = Tempo Teorico</div>
      <div><span style="color:var(--text);font-weight:500">2.0</span> &nbsp;Tempo Reale = 2x Tempo Teorico</div>
    </div>
    <div class="gauge-wrap">
      <svg width="320" height="188" viewBox="0 0 230 135">
        <defs><linearGradient id="gArc" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#22c55e"/><stop offset="60%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#ef4444"/>
        </linearGradient></defs>
        <path d="M 25 115 A 90 90 0 0 1 205 115" fill="none" stroke="#1f2730" stroke-width="13" stroke-linecap="round"/>
        <path d="M 25 115 A 90 90 0 0 1 205 115" fill="none" stroke="url(#gArc)" stroke-width="13" stroke-linecap="round" stroke-dasharray="283" stroke-dashoffset="30" opacity=".65"/>
        <line id="gaugeNeedle" x1="115" y1="115" x2="115" y2="34" stroke="var(--amber)" stroke-width="2.5" stroke-linecap="round" transform="rotate(__NEEDLE_DEG__, 115, 115)"/>
        <circle cx="115" cy="115" r="5" fill="var(--amber)"/>
        <text x="18" y="131" fill="#5a6880" font-size="9.5" font-family="DM Mono, monospace">1.0</text>
        <text x="198" y="131" fill="#5a6880" font-size="9.5" font-family="DM Mono, monospace">2.0</text>
      </svg>
    </div>
    <div style="text-align:center;margin-top:-.25rem">
      <div id="gaugeLabel" style="font-family:var(--mono);font-size:2.4rem;font-weight:500;color:#e2e8f0;letter-spacing:-.02em;line-height:1">__INDICE_MEDIO__</div>
      <div style="font-family:var(--mono);font-size:.8rem;color:#5a6880;letter-spacing:.05em;margin-top:.3rem">indice medio</div>
    </div>
  </div>
</div>
<div class="s-title">Ordini di lavoro</div>
<div class="period-btns">
  <button class="wo-period-btn active" id="wo_btn_all" data-period="all" onclick="filterWoByPeriod(this.dataset.period)">Tutto</button>
  <button class="wo-period-btn" id="wo_btn_1m" data-period="1m" onclick="filterWoByPeriod(this.dataset.period)">Ultimo mese</button>
  <button class="wo-period-btn" id="wo_btn_3m" data-period="3m" onclick="filterWoByPeriod(this.dataset.period)">Ultimi 3 mesi</button>
  <button class="wo-period-btn" id="wo_btn_1y" data-period="1y" onclick="filterWoByPeriod(this.dataset.period)">Ultimo anno</button>
</div>
<div class="full-card">
  <div class="overflow-table">
    <table class="data-table" id="woTableEl">
      <thead>
        <tr id="woSortRow">
          <th data-col="WO">WO <span class="sort-icon">↕</span></th>
          <th data-col="ARTICOLO">Articolo <span class="sort-icon">↕</span></th>
          <th data-col="FASE">Fase <span class="sort-icon">↕</span></th>
          <th data-col="C.d.L. Prev">C.d.L. <span class="sort-icon">↕</span></th>
          <th data-col="Data_Ora_Fine">Data fine <span class="sort-icon">↓</span></th>
          <th data-col="Buon Tempo Ciclo ORE">Ore teoriche <span class="sort-icon">↕</span></th>
          <th data-col="Tempo Lavoraz. ORE">Ore reali <span class="sort-icon">↕</span></th>
          <th data-col="Indice_Inefficienza">Indice <span class="sort-icon">↕</span></th>
          <th data-col="stato">Stato <span class="sort-icon">↕</span></th>
        </tr>
        <tr id="woFilterRow">
          <th><input class="col-filter" data-col="WO"                    placeholder="Filtra…"></th>
          <th><input class="col-filter" data-col="ARTICOLO"              placeholder="Filtra…"></th>
          <th><input class="col-filter" data-col="FASE"                  placeholder="Filtra…"></th>
          <th><input class="col-filter" data-col="C.d.L. Prev"           placeholder="Filtra…"></th>
          <th><input class="col-filter" data-col="Data_Ora_Fine"         placeholder="gg/mm/aaaa"></th>
          <th><input class="col-filter" data-col="Buon Tempo Ciclo ORE" placeholder="es. >2"></th>
          <th><input class="col-filter" data-col="Tempo Lavoraz. ORE"    placeholder="es. >2"></th>
          <th><input class="col-filter" data-col="Indice_Inefficienza"   placeholder="es. >1.5"></th>
          <th><input class="col-filter" data-col="stato"                 placeholder="NORMALE…"></th>
        </tr>
      </thead>
      <tbody id="woTable"></tbody>
    </table>
  </div>
</div>
<div class="two-col" style="align-items:stretch">
  <div class="card" style="display:flex;flex-direction:column;overflow:hidden;height:420px">
    <div class="s-title">Tasso anomalia per articolo</div>
    <div style="flex:1;min-height:0;overflow-y:auto;padding-right:4px;scrollbar-width:thin;scrollbar-color:var(--border2) transparent">
      <div class="bar-list" id="artBars"></div>
    </div>
  </div>
  <div class="card" style="height:420px;overflow:hidden">
    <div class="s-title">Ore perse per mese (ultimi 6)</div>
    <div class="spark" id="sparkEl"></div>
    <div class="spark-labels" id="sparkLbl"></div>
    <div id="orePerseDisplay" style="margin-top:1rem;display:flex;align-items:baseline;gap:.55rem"></div>
    <div style="margin-top:1.5rem">
      <div class="s-title">Tasso anomalia per fase &#8212; <span id="faseMeseTitle" style="color:var(--amber)"></span></div>
      <div class="bar-list" id="faseBars"></div>
    </div>
  </div>
</div>
<div class="s-title">Previsione rischio &#8212; mese successivo</div>
<div class="full-card">
  <div class="forecast-grid" id="forecastGrid"></div>
  <div class="note">
    <strong style="color:var(--amber)">Come funziona questa previsione:</strong><br>
    Vengono mostrati gli<strong> articoli noti al modello</strong> (i piu frequenti nel dataset di addestramento, con probabilita anomalia &gt; 14%).<br>
    Per ogni articolo viene calcolata la <strong>probabilita di anomalia</strong> basandosi sullo storico disponibile.<br>
    Risponde alla domanda: <em>Se il mese prossimo si produce uno di questi articoli, qual e il rischio atteso di un&#39;anomalia?</em><br>
    <span style="color:var(--red)">Rosso &ge; 60%</span> &nbsp;&middot;&nbsp; <span style="color:var(--amber)">Arancione 35-59%</span> &nbsp;&middot;&nbsp; <span style="color:var(--green)">Verde &lt; 35%</span>
  </div>
</div>
<div style="font-family:var(--mono);font-size:.65rem;color:var(--muted);margin-top:-.75rem;margin-bottom:2rem;padding:.5rem .75rem;background:var(--surface);border:1px solid var(--border);border-radius:6px">
  Soglie indice inefficienza: &nbsp;<span style="color:var(--amber)">&#9679; ATTENZIONE &ge; __SOGLIA_ATTENZIONE__</span> &nbsp;&middot;&nbsp; <span style="color:var(--red)">&#9679; ANOMALIA &ge; __SOGLIA_ANOMALIA__</span> &nbsp;&middot;&nbsp; <span style="color:var(--muted)">(60° e 85° percentile &mdash; train set)</span>
</div>
</main>
<script>
const ALL_RECORDS=__ALL_RECORDS_JSON__;
const CAL=__CAL_JSON__;
const WO_DATA=__WO_JSON__;
const ART_DATA=__ART_JSON__;
const FASE_DATA=__FASE_JSON__;
const FASE_X_MESE=__FASE_X_MESE_JSON__;
const SPARK=__SPARK_JSON__;
const FORECAST=__FORECAST_JSON__;
const WO_X_GIORNO=__WO_X_GIORNO_JSON__;

// ── Calendario ────────────────────────────────────────────────────────────
function calShowDetail(ds,dl){
  document.getElementById('calScroll').style.display='none';
  document.getElementById('calLegend').style.display='none';
  document.getElementById('calHint').style.display='none';
  document.getElementById('calBackBtn').style.display='';
  const det=document.getElementById('calDetail');
  det.style.display='block';
  const wos=WO_X_GIORNO[ds]||[];
  let h='<div style="font-family:var(--mono);font-size:.72rem;color:var(--amber);margin-bottom:.7rem">'+dl+'</div>';
  if(!wos.length){
    h+='<div style="font-family:var(--mono);font-size:.68rem;color:var(--muted)">Nessuna lavorazione.</div>';
  } else {
    const nA=wos.filter(r=>r.stato==='ANOMALIA').length;
    const nAt=wos.filter(r=>r.stato==='ATTENZIONE').length;
    h+='<div style="font-family:var(--mono);font-size:.62rem;color:var(--muted);margin-bottom:.65rem">'+wos.length+' lavorazioni &middot; <span style="color:var(--red)">'+nA+' anomali'+(nA===1?'a':'e')+'</span>';
    if(nAt>0) h+=' &middot; <span style="color:var(--amber)">'+nAt+' attenzione</span>';
    h+='</div>';
    h+='<div style="overflow-y:auto;max-height:195px"><table class="data-table" style="font-size:.68rem"><thead><tr><th>WO</th><th>Articolo</th><th>Fase</th><th>Stato</th><th>Ore teoriche</th><th>Ore reali</th></tr></thead><tbody>';
    wos.forEach(r=>{
      const bc=r.stato==='ANOMALIA'?'b-red':r.stato==='ATTENZIONE'?'b-amber':'b-green';
      h+='<tr><td>'+r.WO+'</td><td>'+r.articolo+'</td><td>'+r.fase+'</td><td><span class="badge '+bc+'">'+r.stato+'</span></td><td>'+(r.ore_teoriche!==null?r.ore_teoriche+' h':'&mdash;')+'</td><td>'+(r.ore_reali!==null?r.ore_reali+' h':'&mdash;')+'</td></tr>';
    });
    h+='</tbody></table></div>';
  }
  det.innerHTML=h;
}
function calShowGrid(){
  document.getElementById('calScroll').style.display='';
  document.getElementById('calLegend').style.display='';
  document.getElementById('calHint').style.display='';
  document.getElementById('calBackBtn').style.display='none';
  document.getElementById('calDetail').style.display='none';
}
(function(){
  const el=document.getElementById('calScroll');
  CAL.forEach(m=>{
    const w=document.createElement('div');w.className='cal-month';
    const lb=document.createElement('div');lb.className='cal-month-label';lb.textContent=m.label;w.appendChild(lb);
    const g=document.createElement('div');g.className='cal-grid';
    m.giorni.forEach(d=>{
      const sq=document.createElement('div');sq.className='cal-day '+d.stato;
      const dstr=m.mese+'-'+String(d.giorno).padStart(2,'0');
      sq.title=d.giorno+' '+m.label+' '+(d.stato==='anomaly'?'ANOMALIA':d.stato==='attention'?'ATTENZIONE':d.stato==='normal'?'NORMALE':'nessuna produzione');
      if(d.stato!=='empty'){sq.style.cursor='pointer';sq.addEventListener('click',()=>calShowDetail(dstr,d.giorno+' '+m.label));}
      g.appendChild(sq);
    });
    w.appendChild(g);el.appendChild(w);
  });
})();

// ── Tabella WO — stato sort/filter ────────────────────────────────────────
var _woAllRows  = [];
var _woCurRows  = [];   // subset del periodo corrente
var _woSortCol  = 'Data_Ora_Fine';
var _woSortDir  = 'desc';
var _woFilters  = {};
var WO_NUMERIC  = ['Buon Tempo Ciclo ORE','Tempo Lavoraz. ORE','Indice_Inefficienza'];

function matchNumWo(val, expr) {
  var s = expr.trim();
  if (!s) return true;
  var m = s.match(/^([><=!]=?)\s*([\d.]+)$/);
  if (!m) return String(val).toLowerCase().includes(s);
  var op = m[1], num = parseFloat(m[2]);
  if      (op === '>')  return val >  num;
  else if (op === '>=') return val >= num;
  else if (op === '<')  return val <  num;
  else if (op === '<=') return val <= num;
  else                  return Math.abs(val - num) < 0.001;
}

function applyWoFiltersAndSort() {
  var rows = _woCurRows.slice();

  // filtri
  Object.keys(_woFilters).forEach(function(col) {
    var expr = (_woFilters[col] || '').toLowerCase().trim();
    if (!expr) return;
    rows = rows.filter(function(r) {
      var v = r[col];
      if (v == null) return false;
      if (WO_NUMERIC.indexOf(col) >= 0 && typeof v === 'number') return matchNumWo(v, expr);
      return String(v).toLowerCase().includes(expr);
    });
  });

  // sort
  var col = _woSortCol, dir = _woSortDir === 'asc' ? 1 : -1;
  rows.sort(function(a, b) {
    // Per la colonna Data_Ora_Fine usa _d (YYYY-MM-DD) per ordinamento cronologico corretto
    var av = col === 'Data_Ora_Fine' ? (a._d || '') : a[col];
    var bv = col === 'Data_Ora_Fine' ? (b._d || '') : b[col];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
    return String(av).localeCompare(String(bv)) * dir;
  });

  // icone header
  document.querySelectorAll('#woSortRow th[data-col]').forEach(function(th) {
    th.classList.remove('sort-asc','sort-desc');
    var icon = th.querySelector('.sort-icon');
    if (th.dataset.col === col) {
      th.classList.add(_woSortDir === 'asc' ? 'sort-asc' : 'sort-desc');
      if (icon) icon.textContent = _woSortDir === 'asc' ? '↑' : '↓';
    } else {
      if (icon) icon.textContent = '↕';
    }
  });

  // render
  var tb = document.getElementById('woTable');
  tb.innerHTML = '';
  if (!rows.length) {
    tb.innerHTML = '<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:1.5rem">Nessun dato disponibile</td></tr>';
    return;
  }
  var frag = document.createDocumentFragment();
  rows.forEach(function(r) {
    var ind = r.Indice_Inefficienza != null ? r.Indice_Inefficienza : '-';
    var ic  = ind >= __SOGLIA_ANOMALIA__ ? 'var(--red)' : ind >= __SOGLIA_ATTENZIONE__ ? 'var(--amber)' : 'var(--muted)';
    var bc  = r.stato === 'ANOMALIA' ? 'b-red' : r.stato === 'ATTENZIONE' ? 'b-amber' : 'b-green';
    var tr = document.createElement('tr');
    tr.innerHTML =
      '<td>'+(r.WO||'&mdash;')+'</td>'+
      '<td>'+(r.ARTICOLO||r.ARTICOLO_grouped||'&mdash;')+'</td>'+
      '<td>'+(r.FASE||'&mdash;')+'</td>'+
      '<td>'+(r['C.d.L. Prev']||'&mdash;')+'</td>'+
      '<td>'+(r.Data_Ora_Fine||'&mdash;')+'</td>'+
      '<td>'+(r.Buon Tempo Ciclo ORE!=null?r.Buon Tempo Ciclo ORE:'&mdash;')+' h</td>'+
      '<td>'+(r['Tempo Lavoraz. ORE']!=null?r['Tempo Lavoraz. ORE']:'&mdash;')+' h</td>'+
      '<td style="color:'+ic+'">'+ind+'</td>'+
      '<td><span class="badge '+bc+'">'+r.stato+'</span></td>';
    frag.appendChild(tr);
  });
  tb.appendChild(frag);
}

// click header = sort
document.getElementById('woSortRow').addEventListener('click', function(e) {
  var th = e.target.closest('th[data-col]');
  if (!th) return;
  var col = th.dataset.col;
  if (_woSortCol === col) {
    _woSortDir = _woSortDir === 'asc' ? 'desc' : 'asc';
  } else {
    _woSortCol = col;
    _woSortDir = WO_NUMERIC.indexOf(col) >= 0 ? 'desc' : 'asc';
  }
  applyWoFiltersAndSort();
});

// input filtro
document.getElementById('woFilterRow').addEventListener('input', function(e) {
  var inp = e.target;
  if (!inp.classList.contains('col-filter')) return;
  _woFilters[inp.dataset.col] = inp.value;
  applyWoFiltersAndSort();
});

function filterWoByPeriod(p) {
  document.querySelectorAll('.wo-period-btn').forEach(function(b){b.classList.remove('active');});
  document.getElementById('wo_btn_'+p).classList.add('active');
  var ref = window._dataMaxDate || new Date();
  if (p === 'all') {
    _woCurRows = WO_DATA.slice();
  } else {
    var cutDate;
    if      (p === '1m') cutDate = new Date(ref.getFullYear(), ref.getMonth()-1, ref.getDate());
    else if (p === '3m') cutDate = new Date(ref.getFullYear(), ref.getMonth()-3, ref.getDate());
    else if (p === '1y') cutDate = new Date(ref.getFullYear()-1, ref.getMonth(), ref.getDate());
    _woCurRows = WO_DATA.filter(function(r){ return r._d && new Date(r._d) >= cutDate; });
  }
  // reset filtri e sort quando cambia il periodo
  _woFilters = {};
  _woSortCol = 'Data_Ora_Fine';
  _woSortDir = 'desc';
  document.querySelectorAll('.col-filter').forEach(function(inp){ inp.value = ''; });
  applyWoFiltersAndSort();
}

_woCurRows = WO_DATA.slice();
applyWoFiltersAndSort();

// ── Articoli ─────────────────────────────────────────────────────────────
(function(){
  const el=document.getElementById('artBars');
  const max=Math.max(...ART_DATA.values,1);
  ART_DATA.labels.forEach((lb,i)=>{
    const p=ART_DATA.values[i];
    const c=p>=50?'c-red':p>=30?'c-amber':'c-green';
    el.innerHTML+='<div class="bar-row"><div class="bar-lbl" title="'+lb+'">'+lb+'</div><div class="bar-track"><div class="bar-fill '+c+'" style="width:0%" data-w="'+(p/max*100)+'">'+p+'%</div></div><div class="bar-pct">'+p+'%</div></div>';
  });
  requestAnimationFrame(()=>{document.querySelectorAll('#artBars .bar-fill').forEach(b=>b.style.width=b.dataset.w+'%');});
})();

function renderOrePerse(ore,label){
  const el=document.getElementById('orePerseDisplay');if(!el)return;
  const c=ore>=20?'var(--red)':ore>=8?'var(--amber)':'var(--green)';
  el.innerHTML='<span style="font-family:var(--mono);font-size:1.55rem;font-weight:500;color:'+c+';letter-spacing:-.02em;line-height:1">'+ore+' h</span><span style="font-family:var(--mono);font-size:.65rem;color:var(--muted);letter-spacing:.06em">perse &mdash; '+label+'</span>';
}
function renderFaseChart(dati,label){
  const el=document.getElementById('faseBars');
  const t=document.getElementById('faseMeseTitle');
  if(t)t.textContent=label||'';
  el.innerHTML='';
  if(!dati||!dati.labels||!dati.labels.length){el.innerHTML='<div style="font-family:var(--mono);font-size:.68rem;color:var(--muted)">Nessun dato.</div>';return;}
  const max=Math.max(...dati.values,1);
  dati.labels.forEach((lb,i)=>{
    const p=dati.values[i];
    const c=p>=50?'c-red':p>=30?'c-amber':'c-green';
    el.innerHTML+='<div class="bar-row"><div class="bar-lbl" style="width:80px" title="'+lb+'">'+lb+'</div><div class="bar-track"><div class="bar-fill '+c+'" style="width:0%" data-w="'+(p/max*100)+'">'+p+'%</div></div><div class="bar-pct">'+p+'%</div></div>';
  });
  requestAnimationFrame(()=>{el.querySelectorAll('.bar-fill').forEach(b=>b.style.width=b.dataset.w+'%');});
}
(function(){
  const el=document.getElementById('sparkEl');
  const lb=document.getElementById('sparkLbl');
  if(!SPARK.values.length)return;
  const max=Math.max(...SPARK.values,.1);
  const lk=SPARK.keys?SPARK.keys[SPARK.keys.length-1]:null;
  const ll=SPARK.labels[SPARK.labels.length-1]||'';
  renderOrePerse(SPARK.values[SPARK.values.length-1]||0,ll);
  if(lk&&FASE_X_MESE[lk])renderFaseChart(FASE_X_MESE[lk],ll);
  else renderFaseChart(FASE_DATA,'storico');
  SPARK.values.forEach((v,i)=>{
    const bar=document.createElement('div');
    bar.className='spark-bar'+(i===SPARK.values.length-1?' selected':'');
    bar.style.height=(v/max*100)+'%';
    bar.title=SPARK.labels[i]+': '+v+' h';
    bar.addEventListener('click',()=>{
      document.querySelectorAll('#sparkEl .spark-bar').forEach(b=>b.classList.remove('selected'));
      bar.classList.add('selected');
      const k=SPARK.keys?SPARK.keys[i]:null;
      const l=SPARK.labels[i]||'';
      renderOrePerse(SPARK.values[i]||0,l);
      if(k&&FASE_X_MESE[k])renderFaseChart(FASE_X_MESE[k],l);
      else renderFaseChart(FASE_DATA,'storico');
    });
    el.appendChild(bar);
  });
  SPARK.labels.forEach(l=>{const s=document.createElement('span');s.textContent=l;lb.appendChild(s);});
})();

// ── Forecast ──────────────────────────────────────────────────────────────
(function(){
  const el=document.getElementById('forecastGrid');
  FORECAST.forEach(it=>{
    const art=it.ARTICOLO_grouped||it.ARTICOLO||'&mdash;';
    const fase=it.FASE||'&mdash;';
    const prob=Math.round(it.prob);
    el.innerHTML+=
      '<div class="fc-item '+it.risk_cls+'">'+
      '<div style="margin-bottom:.12rem"><span class="fc-label">Articolo:&nbsp;</span><span class="fc-art-val">'+art+'</span></div>'+
      '<div style="margin-bottom:.35rem"><span class="fc-label">Fase:&nbsp;</span><span class="fc-fase-val">'+fase+'</span></div>'+
      '<div class="fc-meta">'+it.n+' lavoraz. osservate</div>'+
      '<div class="fc-bar"><div class="fc-fill" style="width:'+prob+'%"></div></div>'+
      '<div class="fc-prob-row"><span class="fc-prob-lbl">Prob. anomalia:</span><span class="fc-prob-val">'+prob+'%</span></div>'+
      '</div>';
  });
  if(!FORECAST.length)el.innerHTML='<div style="font-family:var(--mono);font-size:.72rem;color:var(--muted)">Dati insufficienti.</div>';
})();

// ── KPI dinamici per periodo ──────────────────────────────────────────────
(function(){
  var mx=null;
  ALL_RECORDS.forEach(function(r){if(r.d){var d=new Date(r.d);if(!mx||d>mx)mx=d;}});
  window._dataMaxDate=mx;
})();

function animateCount(id,target,suffix,decimals,dur){
  var el=document.getElementById(id);if(!el)return;
  var start=null;dur=dur||900;var tok={};el._atok=tok;
  function step(ts){
    if(el._atok!==tok)return;
    if(!start)start=ts;
    var prog=Math.min((ts-start)/dur,1);
    var ease=1-Math.pow(1-prog,3);
    var cur=target*ease;
    el.textContent=(decimals>0?cur.toFixed(decimals):Math.round(cur))+(suffix||"");
    if(prog<1)requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function filterByPeriod(p){
  document.querySelectorAll(".period-btn").forEach(function(b){b.classList.remove("active");});
  document.getElementById("btn_"+p).classList.add("active");
  var ref=window._dataMaxDate||new Date();
  var cutDate=null;var recs;
  if(p=="all"){recs=ALL_RECORDS;}
  else{
    if(p=="1m")cutDate=new Date(ref.getFullYear(),ref.getMonth()-1,ref.getDate());
    else if(p=="3m")cutDate=new Date(ref.getFullYear(),ref.getMonth()-3,ref.getDate());
    else if(p=="1y")cutDate=new Date(ref.getFullYear()-1,ref.getMonth(),ref.getDate());
    recs=ALL_RECORDS.filter(function(r){return r.d&&new Date(r.d)>=cutDate;});
  }
  var nT=recs.length;
  var nA=recs.filter(function(r){return r.p===1;}).length;
  var nN=nT-nA;
  var tasso=nT>0?Math.round(nA/nT*1000)/10:0;
  var percN=nT>0?Math.round(nN/nT*1000)/10:0;
  var op=0;
  recs.forEach(function(r){if(r.r!=null&&r.t!=null)op+=Math.max(0,r.r-r.t);});
  op=Math.round(op*10)/10;
  animateCount("kpiAnomalie",nA,"",0);
  document.getElementById("kpiSubAnomalie").textContent="su "+nT+" lavorazioni";
  animateCount("kpiTasso",tasso,"%",1);
  animateCount("kpiOrePerse",op," h",1);
  animateCount("kpiNormali",nN,"",0);
  document.getElementById("kpiPercNormali").textContent=percN+"% del totale";
  var deltaIds=["kpiDelta","kpiDeltaTasso","kpiDeltaOrePerse","kpiDeltaNormali"];
  if(p=="all"){
    deltaIds.forEach(function(id){var el=document.getElementById(id);if(el)el.style.display="none";});
  } else {
    var prevCutDate;
    if(p=="1m")prevCutDate=new Date(ref.getFullYear(),ref.getMonth()-2,ref.getDate());
    else if(p=="3m")prevCutDate=new Date(ref.getFullYear(),ref.getMonth()-6,ref.getDate());
    else if(p=="1y")prevCutDate=new Date(ref.getFullYear()-2,ref.getMonth(),ref.getDate());
    var prevRecs=ALL_RECORDS.filter(function(r){return r.d&&new Date(r.d)>=prevCutDate&&new Date(r.d)<cutDate;});
    var pA=prevRecs.filter(function(r){return r.p===1;}).length;
    var pN=prevRecs.length-pA;
    var pTasso=prevRecs.length>0?Math.round(pA/prevRecs.length*1000)/10:0;
    var pOp=0;prevRecs.forEach(function(r){if(r.r!=null&&r.t!=null)pOp+=Math.max(0,r.r-r.t);});
    pOp=Math.round(pOp*10)/10;
    function fmtYM(d){return d.getFullYear()+"-"+String(d.getMonth()+1).padStart(2,"0");}
    var refLabel;
    if(p=="1m"){refLabel=fmtYM(cutDate);}
    else{var sM=new Date(prevCutDate.getFullYear(),prevCutDate.getMonth());var eM=new Date(cutDate.getFullYear(),cutDate.getMonth()-1);refLabel=fmtYM(sM)+" - "+fmtYM(eM);}
    function setDelta(id,diff,suffix,goodIfDown){
      var el=document.getElementById(id);if(!el)return;
      el.style.display="";var arrow=diff>0?"\u25b2":"\u25bc";var sign=diff>0?"+":"";
      var isGood=goodIfDown?(diff<=0):(diff>=0);var cls=diff===0?"neutral":(isGood?"down":"up");
      el.className="kpi-delta "+cls;el.textContent=arrow+" "+sign+diff+suffix+" vs "+refLabel;
    }
    setDelta("kpiDelta",nA-pA,"",true);
    setDelta("kpiDeltaTasso",Math.round((tasso-pTasso)*10)/10,"%",true);
    setDelta("kpiDeltaOrePerse",Math.round((op-pOp)*10)/10,"h",true);
    setDelta("kpiDeltaNormali",nN-pN,"",false);
  }
  var sumI=0,cntI=0;
  recs.forEach(function(r){if(r.r!=null&&r.t!=null&&r.t>0){sumI+=r.r/r.t;cntI++;}});
  var indMedio=cntI>0?(sumI/cntI):1.0;
  var targetDeg=-90+Math.max(0,Math.min(1,(indMedio-1.0)/1.0))*180;
  var nl=document.getElementById("gaugeNeedle");
  var gl=document.getElementById("gaugeLabel");
  var _gST=null,_gDur=900;
  function _animGauge(ts){
    if(!_gST)_gST=ts;var prog=Math.min((ts-_gST)/_gDur,1);var ease=1-Math.pow(1-prog,3);
    var curDeg=-90+(targetDeg+90)*ease;var curVal=1.0+(indMedio-1.0)*ease;
    if(nl)nl.setAttribute("transform","rotate("+curDeg.toFixed(2)+", 115, 115)");
    if(gl)gl.textContent=curVal.toFixed(3);
    if(prog<1)requestAnimationFrame(_animGauge);
  }
  requestAnimationFrame(_animGauge);
}
filterByPeriod("all");
</script></body></html>"""


def _needle_deg(v, vmin=1.0, vmax=2.0):
    r = max(0.0, min(1.0, (v-vmin)/max(vmax-vmin,0.001)))
    return int(-90 + r*180)

def tutti_record_per_kpi(df):
    cols = ["classe_vera"]
    if "Data_Ora_Fine" in df.columns: cols = ["Data_Ora_Fine"] + cols
    if "Tempo Lavoraz. ORE" in df.columns: cols.append("Tempo Lavoraz. ORE")
    if "Buon Tempo Ciclo ORE" in df.columns: cols.append("Buon Tempo Ciclo ORE")
    sub = df[[c for c in cols if c in df.columns]].copy()
    records = []
    for _, row in sub.iterrows():
        d = None
        if "Data_Ora_Fine" in row.index and pd.notna(row["Data_Ora_Fine"]):
            d = row["Data_Ora_Fine"].strftime("%Y-%m-%d")
        r = row.get("Tempo Lavoraz. ORE")
        t = row.get("Buon Tempo Ciclo ORE")
        records.append({
            "d": d,
            "p": int(row.get("classe_vera", 0) or 0),
            "r": round(float(r), 3) if r is not None and pd.notna(r) else None,
            "t": round(float(t), 3) if t is not None and pd.notna(t) else None,
        })
    return records


def genera_html(kpi, cal, wo, art, fase, spark, forecast, soglia,
                model_name="XGBoost", indice_medio_globale=1.0,
                fase_x_mese=None, wo_x_giorno=None, all_records=None,
                soglia_attenzione=1.1511):
    n_normali = kpi["n_normali"]
    perc_norm = round(n_normali / max(kpi["n_tot"],1)*100, 1)
    html = HTML_TEMPLATE
    for k,v in {
        "__GENERATO_IL__":      kpi["generato_il"],
        "__N_TOT__":            str(kpi["n_tot"]),
        "__PERIODO__":          kpi["periodo"],
        "__N_ANOMALIE__":       str(kpi["n_anomalie"]),
        "__N_NORMALI__":        str(n_normali),
        "__PERC_NORMALI__":     str(perc_norm),
        "__TASSO__":            str(kpi["tasso"]),
        "__ORE_PERSE__":        str(kpi["ore_perse"]),
        "__DELTA_ANOMALIE__":   kpi["delta_anomalie"],
        "__SOGLIA__":           f"{soglia:.4f}",
        "__SOGLIA_ANOMALIA__":  f"{soglia:.4f}",
        "__SOGLIA_ATTENZIONE__": f"{soglia_attenzione:.4f}",
        "__INDICE_MEDIO__":     str(round(float(indice_medio_globale),3)),
        "__NEEDLE_DEG__":       str(_needle_deg(float(indice_medio_globale))),
        "__MODEL_NAME__":       model_name,
        "__CAL_JSON__":         json.dumps(cal),
        "__WO_JSON__":          json.dumps(wo),
        "__ART_JSON__":         json.dumps(art),
        "__FASE_JSON__":        json.dumps(fase),
        "__SPARK_JSON__":       json.dumps(spark),
        "__FORECAST_JSON__":    json.dumps(forecast),
        "__FASE_X_MESE_JSON__": json.dumps(fase_x_mese or {}),
        "__ALL_RECORDS_JSON__": json.dumps(all_records or []),
        "__WO_X_GIORNO_JSON__": json.dumps(wo_x_giorno or {}),
    }.items():
        html = html.replace(k, v)
    return html

def genera_dashboard(path_input=DATA_PATH, path_output=OUTPUT_HTML,
                     model_path=MODEL_PATH, params_path=PARAMS_PATH):
    print(f"[1/6] Caricamento modello da:  {model_path}")
    model  = joblib.load(model_path)
    params = joblib.load(params_path)
    soglia            = float(params.get("soglia_anomalia",   1.4220))
    soglia_attenzione = float(params.get("soglia_attenzione", 1.1511))
    model_name = type(model.named_steps["model"]).__name__
    print(f"[2/6] Caricamento dati da:     {path_input}")
    df = carica_e_prepara(path_input, params)
    print(f"[3/6] Calcolo predizioni ...")
    df = predici(model, df)
    print(f"[4/6] Aggregazione dati ...")
    kpi          = kpi_globali(df)
    cal          = calendario_tutti_mesi(df)
    wo           = tabella_wo_tutti(df, n=len(df), soglia_attenzione=soglia_attenzione, soglia_anomalia=soglia)
    art          = anomalie_per_articolo(df)
    fase         = anomalie_per_fase(df)
    spark        = sparkline_ore_perse(df)
    forecast     = forecast_mese_successivo(df, articoli_top=params.get("articoli_top", []))
    fase_x_mese  = fase_per_mese(df)
    wo_x_giorno  = wo_per_giorno(df)
    all_records  = tutti_record_per_kpi(df)
    indice_medio = float(df["Indice_Inefficienza"].mean()) if "Indice_Inefficienza" in df.columns else 1.0
    print(f"[5/6] Generazione HTML ...")
    html = genera_html(kpi, cal, wo, art, fase, spark, forecast, soglia, model_name,
                       indice_medio_globale=indice_medio, fase_x_mese=fase_x_mese,
                       wo_x_giorno=wo_x_giorno, all_records=all_records,
                       soglia_attenzione=soglia_attenzione)
    print(f"[6/6] Salvataggio in:          {path_output}")
    os.makedirs(os.path.dirname(path_output), exist_ok=True)
    with open(path_output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  Dashboard generata: {path_output}")
    print(f"    Periodo analizzato:  {kpi['periodo']}")
    print(f"    Anomalie rilevate:   {kpi['n_anomalie']} / {kpi['n_tot']} ({kpi['tasso']}%)")
    print(f"    Ore perse stimate:   {kpi['ore_perse']} h")

if __name__ == "__main__":
    genera_dashboard()