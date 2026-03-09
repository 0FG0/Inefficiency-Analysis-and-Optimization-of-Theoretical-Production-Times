"""
dashboard.py
─────────────────────────────────────────────────────────────────────────────
Genera una dashboard HTML interattiva con i risultati del modello di
classificazione anomalie (classification_models_anomaly_oriented.py).

Come usarlo:
    python src/classification/dashboard.py
    → outputs/dashboard_anomaly.html  (apribile nel browser senza server)

Struttura attesa del progetto:
    progetto/
    ├── data/processed/koepfer_160_2.csv
    ├── models/classification/
    │   ├── best_classificazione_anomaly.pkl
    │   └── parametri_classificazione_anomaly.pkl
    ├── outputs/
    └── src/
        ├── feature_engineering.py
        └── classification/
            └── dashboard.py  ← questo file

Dipendenze: pandas, numpy, joblib, scikit-learn, xgboost
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feature_engineering import pipeline_classificazione

DATA_PATH   = os.path.join(PROJECT_ROOT, "data",    "processed",      "koepfer_160_2.csv")
MODEL_PATH  = os.path.join(PROJECT_ROOT, "models",  "classification",  "best_classificazione_anomaly.pkl")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models",  "classification",  "parametri_classificazione_anomaly.pkl")
OUTPUT_HTML = os.path.join(PROJECT_ROOT, "outputs", "dashboard_anomaly.html")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARICAMENTO DATI + FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def carica_e_prepara(path: str, params: dict) -> pd.DataFrame:
    """
    Carica il CSV, applica lo stesso preprocessing usato in training:
      - ARTICOLO_grouped (top-N + ALTRO)
      - pipeline_classificazione (feature engineering)
      - colonna 'classe' basata sulla soglia salvata
    """
    df = pd.read_csv(path)

    # ARTICOLO
    if "ARTICOLO" in df.columns:
        df["ARTICOLO"] = df["ARTICOLO"].fillna("MISSING_ARTICOLO").astype(str)

    articoli_top = params.get("articoli_top", [])
    df["ARTICOLO_grouped"] = df["ARTICOLO"].where(
        df["ARTICOLO"].isin(articoli_top), other="ALTRO"
    )
    df["ARTICOLO_grouped"] = df["ARTICOLO_grouped"].fillna("ALTRO").astype(str)

    # Feature engineering
    df = pipeline_classificazione(df)

    # Classe vera (usata per confronto con predizioni)
    soglia = params.get("soglia_anomalia", df["Indice_Inefficienza"].quantile(0.85))
    df["classe_vera"] = (df["Indice_Inefficienza"] > soglia).astype(int)

    # Data
    if "Data_Ora_Fine" in df.columns:
        df["Data_Ora_Fine"] = pd.to_datetime(df["Data_Ora_Fine"], errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREDIZIONI
# ─────────────────────────────────────────────────────────────────────────────

COLS_TO_DROP = [
    "classe_vera", "Indice_Inefficienza", "Tempo Lavoraz. ORE",
    "Tempo_Teorico_TOT_ORE", "WO", "ARTICOLO", "Descrizione Articolo",
    "ID DAD", "Descrizione Macchina", "C.d.L. Effett", "Data_Ora_Fine",
]

def predici(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge al DataFrame le colonne:
      - classe_pred   : 0=NORMALE / 1=ANOMALIA
      - prob_anomalia : probabilità [0-1] di essere anomalia
    """
    X = df.drop(columns=COLS_TO_DROP, errors="ignore")

    # Variabili categoriche → string (stesso trattamento del training)
    categorical_cols = [
        "ARTICOLO_grouped", "FASE", "Cod CIC",
        "C.d.L. Prev", "Descrizione Centro di Lavoro previsto"
    ]
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("string").fillna("MISSING")

    df = df.copy()
    df["classe_pred"]   = model.predict(X)
    df["prob_anomalia"] = model.predict_proba(X)[:, 1]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. AGGREGAZIONI PER LA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v):
    """Converte valori numpy in tipi Python nativi per JSON."""
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.bool_,)):    return bool(v)
    return v


def kpi_globali(df: pd.DataFrame) -> dict:
    """
    KPI calcolati su TUTTI i dati del dataset.
    Il delta anomalie confronta l'ultimo mese con il penultimo,
    ma i totali (n_tot, n_anomalie, ore_perse, tasso) sono sull'intero periodo.
    """
    # ── Periodo ──────────────────────────────────────────────────────────────
    if "Data_Ora_Fine" in df.columns:
        data_min = df["Data_Ora_Fine"].min()
        data_max = df["Data_Ora_Fine"].max()
        periodo  = (
            f"{data_min.strftime('%b %Y')} – {data_max.strftime('%b %Y')}"
            if pd.notna(data_min) and pd.notna(data_max)
            else "Dataset completo"
        )
    else:
        periodo = "Dataset completo"

    # ── Totali su tutto il dataset ────────────────────────────────────────────
    n_tot      = len(df)
    n_anomalie = int((df["classe_pred"] == 1).sum())
    n_normali  = int((df["classe_pred"] == 0).sum())
    tasso      = round(n_anomalie / max(n_tot, 1) * 100, 1)

    # Ore perse totali (tempo_reale − tempo_teorico, solo anomalie predette)
    ore_perse = 0.0
    if "Tempo Lavoraz. ORE" in df.columns and "Tempo_Teorico_TOT_ORE" in df.columns:
        mask = df["classe_pred"] == 1
        ore_perse = float(
            (df.loc[mask, "Tempo Lavoraz. ORE"] - df.loc[mask, "Tempo_Teorico_TOT_ORE"])
            .clip(lower=0).sum()
        )
        ore_perse = round(ore_perse, 1)

    # ── Delta: confronto ultimo mese vs penultimo ─────────────────────────────
    delta_anomalie = "—"
    if "Data_Ora_Fine" in df.columns:
        periodi = sorted(df["Data_Ora_Fine"].dt.to_period("M").dropna().unique())
        if len(periodi) >= 2:
            ultimo   = periodi[-1]
            penultimo = periodi[-2]
            n_ultimo    = int((df[df["Data_Ora_Fine"].dt.to_period("M") == ultimo]["classe_pred"] == 1).sum())
            n_penultimo = int((df[df["Data_Ora_Fine"].dt.to_period("M") == penultimo]["classe_pred"] == 1).sum())
            diff  = n_ultimo - n_penultimo
            sign  = "+" if diff >= 0 else ""
            delta_anomalie = f"{sign}{diff} vs {str(penultimo)}"

    return {
        "periodo":        periodo,
        "n_tot":          n_tot,
        "n_anomalie":     n_anomalie,
        "n_normali":      n_normali,
        "tasso":          tasso,
        "ore_perse":      ore_perse,
        "delta_anomalie": delta_anomalie,
        "generato_il":    datetime.now().strftime("%d/%m/%Y %H:%M"),
    }


def calendario_tutti_mesi(df: pd.DataFrame) -> list:
    """
    Restituisce la struttura completa del calendario per TUTTI i mesi presenti.
    Ordine: mese più recente in cima (index 0), mesi più vecchi in fondo.

    Ogni elemento:
      {
        "mese":  "2025-10",
        "label": "Ott 2025",
        "giorni": [ {"giorno": 1, "stato": "normal|anomaly|empty"}, ... ]
      }
    """
    if "Data_Ora_Fine" not in df.columns:
        return []

    MESI_IT = {
        1:"Gen", 2:"Feb", 3:"Mar", 4:"Apr", 5:"Mag", 6:"Giu",
        7:"Lug", 8:"Ago", 9:"Set", 10:"Ott", 11:"Nov", 12:"Dic"
    }

    df = df.copy()
    df["_periodo"] = df["Data_Ora_Fine"].dt.to_period("M")
    df["_giorno"]  = df["Data_Ora_Fine"].dt.day

    periodi = sorted(df["_periodo"].dropna().unique(), reverse=True)  # più recente prima

    result = []
    for periodo in periodi:
        sub = df[df["_periodo"] == periodo]

        giorni_anomalia = set(sub.loc[sub["classe_pred"] == 1, "_giorno"].dropna().astype(int))
        giorni_normali  = set(sub.loc[sub["classe_pred"] == 0, "_giorno"].dropna().astype(int))
        n_giorni        = periodo.days_in_month

        giorni_list = []
        for d in range(1, n_giorni + 1):
            if d in giorni_anomalia:
                stato = "anomaly"
            elif d in giorni_normali:
                stato = "normal"
            else:
                stato = "empty"
            giorni_list.append({"giorno": d, "stato": stato})

        mese_str = str(periodo)  # "2025-10"
        anno     = periodo.year
        mese_n   = periodo.month
        label    = f"{MESI_IT.get(mese_n, mese_n)} {anno}"

        result.append({
            "mese":   mese_str,
            "label":  label,
            "giorni": giorni_list,
        })

    return result


def tabella_wo_tutti(df: pd.DataFrame, n: int = 50) -> list:
    """
    Ultimi N ordini di lavoro (tutti, non solo anomalie), ordinati per data desc.
    Aggiunge la colonna 'stato':
      - ANOMALIA  → classe_pred == 1 AND prob >= 0.60
      - ATTENZIONE → classe_pred == 1 AND prob < 0.60
      - NORMALE   → classe_pred == 0
    """
    cols_utili = [
        "WO", "ARTICOLO", "ARTICOLO_grouped", "FASE", "C.d.L. Prev",
        "Data_Ora_Fine", "Tempo Lavoraz. ORE", "Tempo_Teorico_TOT_ORE",
        "Indice_Inefficienza", "prob_anomalia", "classe_pred",
    ]
    cols_presenti = [c for c in cols_utili if c in df.columns]
    sub = df[cols_presenti].copy()

    if "Data_Ora_Fine" in sub.columns:
        sub = sub.sort_values("Data_Ora_Fine", ascending=False)
    sub = sub.head(n)

    # Stato
    def _stato(row):
        if row.get("classe_pred", 0) == 1:
            return "ANOMALIA" if (row.get("prob_anomalia", 0) or 0) >= 0.60 else "ATTENZIONE"
        return "NORMALE"
    sub["stato"] = sub.apply(_stato, axis=1)

    # Serializzazione
    if "Data_Ora_Fine" in sub.columns:
        sub["Data_Ora_Fine"] = sub["Data_Ora_Fine"].dt.strftime("%d/%m/%Y")
    if "Indice_Inefficienza" in sub.columns:
        sub["Indice_Inefficienza"] = sub["Indice_Inefficienza"].round(3)
    if "prob_anomalia" in sub.columns:
        sub["prob_anomalia"] = (sub["prob_anomalia"] * 100).round(2)
    if "Tempo Lavoraz. ORE" in sub.columns:
        sub["Tempo Lavoraz. ORE"] = sub["Tempo Lavoraz. ORE"].round(2)
    if "Tempo_Teorico_TOT_ORE" in sub.columns:
        sub["Tempo_Teorico_TOT_ORE"] = sub["Tempo_Teorico_TOT_ORE"].round(2)

    return [{k: _safe(v) for k, v in row.items()} for row in sub.to_dict(orient="records")]


def anomalie_per_articolo(df: pd.DataFrame, n: int = 8) -> dict:
    """Top N articoli per tasso di anomalia (% su totale lavorazioni di quell'articolo)."""
    if "ARTICOLO_grouped" not in df.columns:
        return {"labels": [], "values": [], "counts": []}

    g = df.groupby("ARTICOLO_grouped").agg(
        totale=("classe_pred", "count"),
        anomalie=("classe_pred", "sum")
    )
    g["tasso"] = (g["anomalie"] / g["totale"] * 100).round(1)
    g = g[g["totale"] >= 3]  # esclude articoli con troppo pochi dati
    g = g.sort_values("tasso", ascending=False).head(n)

    return {
        "labels": g.index.tolist(),
        "values": g["tasso"].tolist(),
        "counts": g["anomalie"].astype(int).tolist(),
    }


def anomalie_per_fase(df: pd.DataFrame) -> dict:
    """Tasso di anomalia per fase — dataset completo (fallback)."""
    if "FASE" not in df.columns:
        return {"labels": [], "values": []}

    g = df.groupby("FASE").agg(
        totale=("classe_pred", "count"),
        anomalie=("classe_pred", "sum")
    )
    g["tasso"] = (g["anomalie"] / g["totale"] * 100).round(1)
    g = g[g["totale"] >= 3].sort_values("tasso", ascending=False).head(6)

    return {"labels": g.index.tolist(), "values": g["tasso"].tolist()}


def fase_per_mese(df: pd.DataFrame) -> dict:
    """
    Tasso anomalia per fase, per ognuno degli ultimi 6 mesi.
    Ritorna un dict keyed su 'YYYY-MM' (stesso formato di SPARK.keys).
    Usato dal JS per aggiornare il bar chart fase al click sulla sparkline.
    """
    if "FASE" not in df.columns or "Data_Ora_Fine" not in df.columns:
        return {}

    df = df.copy()
    df["_mese"] = df["Data_Ora_Fine"].dt.to_period("M")
    periodi = sorted(df["_mese"].dropna().unique())[-6:]

    result = {}
    for periodo in periodi:
        sub = df[df["_mese"] == periodo]
        g = sub.groupby("FASE").agg(
            totale=("classe_pred", "count"),
            anomalie=("classe_pred", "sum")
        )
        g["tasso"] = (g["anomalie"] / g["totale"] * 100).round(1)
        g = g[g["totale"] >= 1].sort_values("tasso", ascending=False).head(6)
        result[str(periodo)] = {
            "labels": g.index.tolist(),
            "values": g["tasso"].tolist(),
        }

    return result


def _fmt_mese(period_str: str) -> str:
    """Da '2025-10' → '10-25', da '2026-03' → '03-26'."""
    parts = period_str.split("-")
    if len(parts) == 2:
        return f"{parts[1]}-{parts[0][2:]}"
    return period_str


def sparkline_ore_perse(df: pd.DataFrame) -> dict:
    """Ore perse per anomalie — ultimi 6 mesi."""
    if "Data_Ora_Fine" not in df.columns:
        return {"labels": [], "values": [], "keys": []}
    if "Tempo Lavoraz. ORE" not in df.columns or "Tempo_Teorico_TOT_ORE" not in df.columns:
        df = df.copy()
        df["_ore_perse"] = df["classe_pred"].astype(float)
    else:
        df = df.copy()
        df["_ore_perse"] = (
            (df["Tempo Lavoraz. ORE"] - df["Tempo_Teorico_TOT_ORE"]).clip(lower=0)
            * (df["classe_pred"] == 1)
        )

    df["_mese"] = df["Data_Ora_Fine"].dt.to_period("M")
    g = df.groupby("_mese")["_ore_perse"].sum().sort_index().tail(6)  # ultimi 6 mesi

    return {
        "labels": [_fmt_mese(str(p)) for p in g.index],
        "keys":   [str(p) for p in g.index],           # YYYY-MM per cross-reference JS
        "values": [round(float(v), 1) for v in g.values],
    }


def forecast_mese_successivo(df: pd.DataFrame) -> list:
    """
    Stima il rischio anomalia per il mese successivo.
    Strategia: prende l'ultimo mese disponibile, usa le probabilità predette
    dal modello per ogni combinazione (ARTICOLO_grouped, FASE) e calcola
    la probabilità media → questa è la previsione di rischio per il prossimo mese,
    assumendo che gli stessi ordini si ripetano (pattern produttivo stabile).
    """
    if "Data_Ora_Fine" not in df.columns:
        sub = df
    else:
        ultimo_mese = df["Data_Ora_Fine"].dt.to_period("M").max()
        sub = df[df["Data_Ora_Fine"].dt.to_period("M") == ultimo_mese]

    group_cols = [c for c in ["ARTICOLO_grouped", "FASE"] if c in sub.columns]
    if not group_cols:
        return []

    g = sub.groupby(group_cols).agg(
        prob_media=("prob_anomalia", "mean"),
        n_lavorazioni=("prob_anomalia", "count"),
    ).reset_index()

    g["prob_media"] = (g["prob_media"] * 100).round(0).astype(int)
    g = g[g["n_lavorazioni"] >= 1].sort_values("prob_media", ascending=False).head(10)

    result = []
    for _, row in g.iterrows():
        prob = float(row["prob_media"])
        risk_cls = "risk-high" if prob >= 60 else "risk-medium" if prob >= 35 else "risk-low"
        item = {c: str(row[c]) for c in group_cols}
        item.update({
            "prob": prob,
            "risk_cls": risk_cls,
            "n": int(row["n_lavorazioni"]),
        })
        result.append(item)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Anomaly Dashboard — Koepfer 160</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {
    --bg:        #0c0e10;
    --surface:   #13171c;
    --border:    #1f2730;
    --border2:   #2a3442;
    --text:      #e2e8f0;
    --muted:     #5a6880;
    --amber:     #f59e0b;
    --amber-dim: #7c4f06;
    --red:       #ef4444;
    --red-dim:   #7f1d1d;
    --green:     #22c55e;
    --green-dim: #14532d;
    --blue:      #38bdf8;
    --blue-dim:  #0c4a6e;
    --mono:      'DM Mono', monospace;
    --sans:      'DM Sans', sans-serif;
    --radius:    12px;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* noise */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
    opacity: .6;
  }

  main { position: relative; z-index: 1; max-width: 1360px; margin: 0 auto; padding: 0 2.5rem 4rem; }

  /* ── HEADER ─────────────────────────────────────────────────────────────── */
  header {
    display: flex; align-items: flex-end; justify-content: space-between;
    padding: 2.5rem 0 1.75rem; border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem; gap: 1rem; flex-wrap: wrap;
  }
  .h-eyebrow {
    font-family: var(--mono); font-size: .68rem; letter-spacing: .16em;
    color: var(--amber); text-transform: uppercase; margin-bottom: .5rem;
  }
  header h1 {
    font-size: 2rem; font-weight: 800; letter-spacing: -.02em; line-height: 1;
  }
  header h1 em { color: var(--amber); font-style: normal; }
  .h-right { text-align: right; }
  .h-right .gen {
    font-family: var(--mono); font-size: .7rem; color: var(--muted);
  }
  .h-badge {
    display: inline-flex; align-items: center; gap: .4rem; margin-top: .5rem;
    background: var(--green-dim); border: 1px solid var(--green);
    color: var(--green); font-family: var(--mono); font-size: .68rem;
    padding: .28rem .7rem; border-radius: 999px; letter-spacing: .08em;
  }
  .h-badge .dot {
    width: 6px; height: 6px; background: var(--green); border-radius: 50%;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

  /* ── SECTION TITLE ───────────────────────────────────────────────────────── */
  .s-title {
    font-family: var(--mono); font-size: .65rem; letter-spacing: .18em;
    color: var(--muted); text-transform: uppercase;
    display: flex; align-items: center; gap: .75rem; margin-bottom: 1.25rem;
  }
  .s-title::after { content:''; flex:1; height:1px; background:var(--border); }

  /* ── KPI GRID ────────────────────────────────────────────────────────────── */
  .kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: var(--radius);
    overflow: hidden; margin-bottom: 2.5rem;
  }
  .kpi-card {
    background: var(--surface); padding: 1.5rem 1.25rem;
    position: relative; overflow: hidden; transition: background .18s;
  }
  .kpi-card:hover { background: #181e26; }
  .kpi-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  }
  .kpi-card.c-amber::after { background: var(--amber); }
  .kpi-card.c-red::after   { background: var(--red); }
  .kpi-card.c-green::after { background: var(--green); }
  .kpi-card.c-blue::after  { background: var(--blue); }

  .kpi-lbl {
    font-family: var(--mono); font-size: .975rem; letter-spacing: .06em;
    color: var(--muted); text-transform: uppercase; margin-bottom: .55rem;
  }
  .kpi-val {
    font-family: var(--sans); font-size: 2.6rem; font-weight: 800;
    line-height: 1; letter-spacing: -.03em;
  }
  .c-amber .kpi-val { color: var(--amber); }
  .c-red   .kpi-val { color: var(--red); }
  .c-green .kpi-val { color: var(--green); }
  .c-blue  .kpi-val { color: var(--blue); }

  .kpi-sub { font-family: var(--mono); font-size: 1.02rem; color: var(--muted); margin-top: .45rem; }
  .kpi-delta {
    display: inline-flex; align-items: center; gap: .18rem;
    font-family: var(--mono); font-size: .975rem; margin-top: .35rem;
  }
  .kpi-delta.up   { color: var(--red); }
  .kpi-delta.down { color: var(--green); }
  .kpi-delta.neutral { color: var(--muted); }

  /* ── LAYOUT GRID ─────────────────────────────────────────────────────────── */
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }
  @media(max-width: 840px) { .two-col { grid-template-columns: 1fr; } }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.5rem;
    animation: fadeUp .45s ease both;
  }
  @keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }

  /* ── CALENDAR ────────────────────────────────────────────────────────────── */
  .cal-scroll {
    height: 260px; overflow-y: auto; overflow-x: hidden;
    padding-right: 4px;
    scrollbar-width: thin; scrollbar-color: var(--border2) transparent;
  }
  .cal-month { margin-bottom: .9rem; }
  .cal-month:last-child { margin-bottom: 0; }
  .cal-month-label {
    font-family: var(--mono); font-size: .6rem; letter-spacing: .12em;
    color: var(--amber); text-transform: uppercase;
    margin-bottom: .35rem;
  }
  .cal-grid { display: flex; gap: 3px; flex-wrap: wrap; }
  .cal-day {
    width: 15px; height: 15px; border-radius: 3px; cursor: default;
    flex-shrink: 0; transition: transform .12s;
  }
  .cal-day:hover { transform: scale(1.45); z-index: 10; position: relative; }
  .cal-day.normal  { background: var(--green-dim); border: 1px solid var(--green); }
  .cal-day.anomaly { background: var(--red-dim);   border: 1px solid var(--red); }
  .cal-day.empty   { background: var(--border); }

  .cal-legend { display: flex; gap: 1.2rem; margin-top: .75rem; }
  .cl-item { display: flex; align-items: center; gap: .38rem; font-family: var(--mono); font-size: .65rem; color: var(--muted); }
  .cl-box  { width: 11px; height: 11px; border-radius: 2px; flex-shrink: 0; }

  /* ── BAR CHART CUSTOM ────────────────────────────────────────────────────── */
  .bar-list { display: flex; flex-direction: column; gap: .7rem; }
  .bar-row  { display: flex; align-items: center; gap: .7rem; }
  .bar-lbl  { font-family: var(--mono); font-size: .68rem; color: var(--muted); width: 86px; flex-shrink: 0; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bar-track { flex: 1; background: var(--border); border-radius: 4px; height: 20px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 4px; display: flex; align-items: center; padding-left: .45rem; font-family: var(--mono); font-size: .65rem; color: #0c0e10; font-weight: 500; transition: width 1.1s cubic-bezier(.22,1,.36,1); }
  .bar-fill.c-red    { background: var(--red); }
  .bar-fill.c-amber  { background: var(--amber); }
  .bar-fill.c-green  { background: var(--green); }
  .bar-pct { font-family: var(--mono); font-size: .65rem; color: var(--muted); width: 34px; text-align: right; flex-shrink: 0; }

  /* ── SPARKLINE ───────────────────────────────────────────────────────────── */
  .spark { display: flex; align-items: flex-end; gap: 4px; height: 56px; }
  .spark-bar { flex: 1; background: var(--amber-dim); border-radius: 3px 3px 0 0; transition: background .2s; cursor: pointer; }
  .spark-bar.last, .spark-bar.selected { background: var(--amber); }
  .spark-bar:hover { background: var(--amber); opacity: .85; }
  .spark-labels { display: flex; justify-content: space-between; font-family: var(--mono); font-size: .62rem; color: var(--muted); margin-top: .35rem; }
  .spark-mese-title { font-family: var(--mono); font-size: .65rem; color: var(--amber); margin-top: .5rem; letter-spacing: .1em; }

  /* ── TABLE ───────────────────────────────────────────────────────────────── */
  .data-table { width: 100%; border-collapse: collapse; font-size: .72rem; }
  .data-table th {
    font-family: var(--mono); font-size: .6rem; letter-spacing: .1em;
    text-transform: uppercase; color: var(--muted); font-weight: 400;
    padding: .55rem .7rem; border-bottom: 1px solid var(--border); text-align: left;
  }
  .data-table td {
    font-family: var(--mono); padding: .6rem .7rem;
    border-bottom: 1px solid var(--border2); vertical-align: middle;
  }
  .data-table tr:last-child td { border-bottom: none; }
  .data-table tr:hover td { background: #181e26; }

  .badge {
    display: inline-block; padding: .12rem .5rem; border-radius: 4px;
    font-family: var(--mono); font-size: .58rem; letter-spacing: .06em;
    text-transform: uppercase;
  }
  .badge.b-red   { background: var(--red-dim);   color: var(--red);   border: 1px solid var(--red); }
  .badge.b-amber { background: var(--amber-dim); color: var(--amber); border: 1px solid var(--amber); }
  .badge.b-green { background: var(--green-dim); color: var(--green); border: 1px solid var(--green); }

  /* ── FORECAST GRID ───────────────────────────────────────────────────────── */
  .forecast-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 1rem; }
  .fc-item {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem 1rem .9rem;
  }
  .fc-art  { font-weight: 700; font-size: .88rem; margin-bottom: .15rem; letter-spacing: -.01em; }
  .fc-fase { font-family: var(--mono); font-size: .65rem; color: var(--muted); margin-bottom: .55rem; }
  .fc-bar  { background: var(--border); border-radius: 3px; height: 5px; margin-bottom: .4rem; overflow: hidden; }
  .fc-fill { height: 100%; border-radius: 3px; }
  .fc-txt  { font-family: var(--mono); font-size: .65rem; color: var(--muted); }
  .fc-txt strong { color: inherit; }

  .risk-high   .fc-fill { background: var(--red); }
  .risk-medium .fc-fill { background: var(--amber); }
  .risk-low    .fc-fill { background: var(--green); }
  .risk-high   .fc-art  { color: var(--red); }
  .risk-medium .fc-art  { color: var(--amber); }
  .risk-low    .fc-art  { color: var(--green); }

  /* ── NOTE ────────────────────────────────────────────────────────────────── */
  .note {
    font-family: var(--mono); font-size: .68rem; color: var(--muted);
    margin-top: 1rem; padding: .7rem .9rem;
    background: var(--bg); border-left: 3px solid var(--amber);
    border-radius: 0 4px 4px 0; line-height: 1.6;
  }

  /* ── GAUGE ───────────────────────────────────────────────────────────────── */
  .gauge-wrap { display: flex; justify-content: center; margin: .5rem 0 .75rem; }

  /* scrollbar */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

  .full-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.5rem; margin-bottom: 2rem;
    animation: fadeUp .45s ease both;
  }
  .overflow-table { overflow-x: auto; overflow-y: auto; max-height: 330px; }
  .overflow-table thead th { position: sticky; top: 0; background: var(--surface); z-index: 2; }
</style>
</head>
<body>
<main>

<!-- HEADER -->
<header>
  <div>
    <div class="h-eyebrow">Monitor Produzione — Analisi Predittiva</div>
    <h1>Koepfer 160 <em>/ Anomalie</em></h1>
  </div>
  <div class="h-right">
    <div class="gen">Generato il __GENERATO_IL__ &nbsp;·&nbsp; __N_TOT__ lavorazioni analizzate</div>
    <div class="h-badge"><span class="dot"></span> Modello attivo</div>
  </div>
</header>

<!-- KPI -->
<div class="s-title">Riepilogo </div>
<div class="kpi-grid">
  <div class="kpi-card c-red">
    <div class="kpi-lbl">Anomalie rilevate</div>
    <div class="kpi-val">__N_ANOMALIE__</div>
    <div class="kpi-sub">su __N_TOT__ lavorazioni</div>
    <div class="kpi-delta up">▲ __DELTA_ANOMALIE__</div>
  </div>
  <div class="kpi-card c-amber">
    <div class="kpi-lbl">Tasso anomalia</div>
    <div class="kpi-val">__TASSO__%</div>
    <div class="kpi-sub">soglia critica: 20%</div>
  </div>
  <div class="kpi-card c-amber">
    <div class="kpi-lbl">Ore perse</div>
    <div class="kpi-val">__ORE_PERSE__ h</div>
    <div class="kpi-sub">tempo reale − teorico (anomalie)</div>
  </div>
  <div class="kpi-card c-green">
    <div class="kpi-lbl">Lavorazioni normali</div>
    <div class="kpi-val">__N_NORMALI__</div>
    <div class="kpi-sub">__PERC_NORMALI__% del totale</div>
  </div>
</div>

<!-- RIGA 1: calendario + indice -->
<div class="two-col">

  <div class="card">
    <div class="s-title">Calendario lavorazioni — storico completo</div>
    <div class="cal-scroll" id="calScroll"></div>
    <div class="cal-legend">
      <div class="cl-item"><div class="cl-box" style="background:var(--green-dim);border:1px solid var(--green)"></div>Normale</div>
      <div class="cl-item"><div class="cl-box" style="background:var(--red-dim);border:1px solid var(--red)"></div>Anomalia</div>
      <div class="cl-item"><div class="cl-box" style="background:var(--border)"></div>Nessuna prod.</div>
    </div>
  </div>

  <div class="card">
    <div class="s-title">Indice medio inefficienza</div>
    <div style="text-align:right;margin-top:-.6rem;margin-bottom:.9rem;font-family:var(--mono);font-size:.78rem;color:var(--muted);line-height:1.9">
      <div><span style="color:var(--text);font-weight:500">1.0</span> &nbsp;Tempo Reale = Tempo Teorico</div>
      <div><span style="color:var(--text);font-weight:500">2.0</span> &nbsp;Tempo Reale = 2× Tempo Teorico</div>
    </div>
    <div class="gauge-wrap">
      <svg width="320" height="188" viewBox="0 0 230 135">
        <defs>
          <linearGradient id="gArc" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stop-color="#22c55e"/>
            <stop offset="60%"  stop-color="#f59e0b"/>
            <stop offset="100%" stop-color="#ef4444"/>
          </linearGradient>
        </defs>
        <path d="M 25 115 A 90 90 0 0 1 205 115" fill="none" stroke="#1f2730"     stroke-width="13" stroke-linecap="round"/>
        <path d="M 25 115 A 90 90 0 0 1 205 115" fill="none" stroke="url(#gArc)" stroke-width="13" stroke-linecap="round" stroke-dasharray="283" stroke-dashoffset="30" opacity=".65"/>
        <line id="needle" x1="115" y1="115" x2="115" y2="34" stroke="var(--amber)" stroke-width="2.5" stroke-linecap="round" transform="rotate(__NEEDLE_DEG__, 115, 115)"/>
        <circle cx="115" cy="115" r="5" fill="var(--amber)"/>
        <text x="18"  y="131" fill="#5a6880" font-size="9.5"  font-family="DM Mono, monospace">1.0</text>
        <text x="198" y="131" fill="#5a6880" font-size="9.5"  font-family="DM Mono, monospace">2.0</text>
      </svg>
    </div>
    <div style="text-align:center;margin-top:-.25rem">
      <div style="font-family:var(--mono);font-size:2.4rem;font-weight:500;color:#e2e8f0;letter-spacing:-.02em;line-height:1">__INDICE_MEDIO__</div>
      <div style="font-family:var(--mono);font-size:.8rem;color:#5a6880;letter-spacing:.05em;margin-top:.3rem">indice medio</div>
    </div>
  </div>

</div>

<!-- TABELLA WO ANOMALI -->
<div class="s-title">Ordini di lavoro anomali (ultimi rilevati)</div>
<div class="full-card">
  <div class="overflow-table">
    <table class="data-table">
      <thead>
        <tr>
          <th>WO</th><th>Articolo</th><th>Fase</th><th>C.d.L.</th>
          <th>Data fine</th><th>Ore teoriche</th><th>Ore reali</th>
          <th>Indice</th><th>Stato</th><th>Confidenza</th>
        </tr>
      </thead>
      <tbody id="woTable"></tbody>
    </table>
  </div>
</div>

<!-- RIGA 2: articoli + sparkline/fase -->
<div class="two-col">

  <div class="card">
    <div class="s-title">Tasso anomalia per articolo</div>
    <div class="bar-list" id="artBars"></div>
  </div>

  <div class="card">
    <div class="s-title">Ore perse per mese (ultimi 6)</div>
    <div class="spark" id="sparkEl"></div>
    <div class="spark-labels" id="sparkLbl"></div>
    <div style="margin-top:1.5rem">
      <div class="s-title">Tasso anomalia per fase — <span id="faseMeseTitle" style="color:var(--amber)"></span></div>
      <div class="bar-list" id="faseBars"></div>
    </div>
  </div>

</div>

<!-- FORECAST -->
<div class="s-title">Previsione rischio — mese successivo</div>
<div class="full-card">
  <div class="forecast-grid" id="forecastGrid"></div>
  <div class="note">
    <strong style="color:var(--amber)">Come funziona questa previsione:</strong><br>
    Il modello ha già assegnato una probabilità di anomalia (0–100%) ad ogni lavorazione storica.<br>
    Per ogni combinazione <em>Articolo + Fase</em> presente nell'ultimo mese del dataset, viene calcolata la <strong>media di queste probabilità</strong>.<br>
    Il risultato risponde alla domanda: <em>"Se il mese prossimo la macchina produce gli stessi articoli nelle stesse fasi, qual è la probabilità attesa di anomalia?"</em><br>
    Le combinazioni con probabilità ≥ 60% sono in rosso (alta priorità), 35–59% in arancione, &lt; 35% in verde.
    Questa stima si basa sull'assunzione che il piano produttivo del mese successivo sia simile all'ultimo mese registrato.
  </div>
</div>

</main>
<script>
const CAL          = __CAL_JSON__;
const WO_DATA      = __WO_JSON__;
const ART_DATA     = __ART_JSON__;
const FASE_DATA    = __FASE_JSON__;
const FASE_X_MESE  = __FASE_X_MESE_JSON__;
const SPARK        = __SPARK_JSON__;
const FORECAST     = __FORECAST_JSON__;

// ── Calendario multi-mese (più recente in cima, scrollabile) ─────────────────
(function(){
  const el = document.getElementById('calScroll');
  CAL.forEach(m => {
    const wrap = document.createElement('div');
    wrap.className = 'cal-month';

    const lbl = document.createElement('div');
    lbl.className = 'cal-month-label';
    lbl.textContent = m.label;
    wrap.appendChild(lbl);

    const grid = document.createElement('div');
    grid.className = 'cal-grid';
    m.giorni.forEach(d => {
      const sq = document.createElement('div');
      sq.className = 'cal-day ' + d.stato;
      sq.title = `${d.giorno} ${m.label} — ${d.stato === 'anomaly' ? 'ANOMALIA' : d.stato === 'normal' ? 'NORMALE' : 'nessuna produzione'}`;
      grid.appendChild(sq);
    });
    wrap.appendChild(grid);
    el.appendChild(wrap);
  });
})();

// ── Tabella WO ───────────────────────────────────────────────────────────────
(function(){
  const tb = document.getElementById('woTable');
  const anomalyRows = WO_DATA.filter(r => r.classe_pred === 1 || r.stato === 'ANOMALIA' || r.stato === 'ATTENZIONE');
  anomalyRows.forEach(r => {
    const prob = (typeof r.prob_anomalia === 'number') ? r.prob_anomalia : null;
    const probFmt = prob !== null ? prob.toFixed(2) + ' %' : '—';
    const probColor = prob >= 80 ? 'var(--red)' : prob >= 60 ? 'var(--amber)' : 'var(--muted)';
    const indice = r.Indice_Inefficienza ?? '-';
    const indColor = indice >= 1.8 ? 'var(--red)' : indice >= 1.43 ? 'var(--amber)' : 'var(--muted)';
    const stato = r.stato || 'ANOMALIA';
    const badgeCls = stato === 'NORMALE' ? 'b-green' : stato === 'ATTENZIONE' ? 'b-amber' : 'b-red';
    tb.innerHTML += `<tr>
      <td>${r.WO||'—'}</td>
      <td>${r.ARTICOLO||r.ARTICOLO_grouped||'—'}</td>
      <td>${r.FASE||'—'}</td>
      <td>${r['C.d.L. Prev']||'—'}</td>
      <td>${r.Data_Ora_Fine||'—'}</td>
      <td>${r.Tempo_Teorico_TOT_ORE??'—'} h</td>
      <td>${r['Tempo Lavoraz. ORE']??'—'} h</td>
      <td style="color:${indColor}">${indice}</td>
      <td><span class="badge ${badgeCls}">${stato}</span></td>
      <td style="color:${probColor};font-weight:500">${probFmt}</td>
    </tr>`;
  });
  if (anomalyRows.length === 0) {
    tb.innerHTML = '<tr><td colspan="10" style="text-align:center;color:var(--muted);padding:1.5rem">Nessuna anomalia rilevata nel periodo</td></tr>';
  }
})();

// ── Bar chart articoli ────────────────────────────────────────────────────────
(function(){
  const el = document.getElementById('artBars');
  const max = Math.max(...ART_DATA.values, 1);
  ART_DATA.labels.forEach((lbl, i) => {
    const pct = ART_DATA.values[i];
    const cls = pct >= 50 ? 'c-red' : pct >= 30 ? 'c-amber' : 'c-green';
    el.innerHTML += `<div class="bar-row">
      <div class="bar-lbl" title="${lbl}">${lbl}</div>
      <div class="bar-track">
        <div class="bar-fill ${cls}" style="width:0%" data-w="${pct/max*100}">${pct}%</div>
      </div>
      <div class="bar-pct">${pct}%</div>
    </div>`;
  });
  requestAnimationFrame(() => {
    document.querySelectorAll('#artBars .bar-fill').forEach(b => {
      b.style.width = b.dataset.w + '%';
    });
  });
})();

// ── Fase chart: render con dati di un mese specifico ─────────────────────────
function renderFaseChart(dati, meseLabel) {
  const el    = document.getElementById('faseBars');
  const title = document.getElementById('faseMeseTitle');
  if (title) title.textContent = meseLabel || '';
  el.innerHTML = '';
  if (!dati || !dati.labels || !dati.labels.length) {
    el.innerHTML = '<div style="font-family:var(--mono);font-size:.68rem;color:var(--muted)">Nessun dato per questo mese.</div>';
    return;
  }
  const max = Math.max(...dati.values, 1);
  dati.labels.forEach((lbl, i) => {
    const pct = dati.values[i];
    const cls = pct >= 50 ? 'c-red' : pct >= 30 ? 'c-amber' : 'c-green';
    el.innerHTML += `<div class="bar-row">
      <div class="bar-lbl" style="width:80px" title="${lbl}">${lbl}</div>
      <div class="bar-track">
        <div class="bar-fill ${cls}" style="width:0%" data-w="${pct/max*100}">${pct}%</div>
      </div>
      <div class="bar-pct">${pct}%</div>
    </div>`;
  });
  requestAnimationFrame(() => {
    el.querySelectorAll('.bar-fill').forEach(b => { b.style.width = b.dataset.w + '%'; });
  });
}

// ── Sparkline ore perse — con click per aggiornare fase chart ─────────────────
(function(){
  const el  = document.getElementById('sparkEl');
  const lbl = document.getElementById('sparkLbl');
  if (!SPARK.values.length) return;
  const max = Math.max(...SPARK.values, .1);

  // Mostra l'ultimo mese come default nel fase chart
  const lastKey   = SPARK.keys ? SPARK.keys[SPARK.keys.length - 1] : null;
  const lastLabel = SPARK.labels[SPARK.labels.length - 1] || '';
  if (lastKey && FASE_X_MESE[lastKey]) renderFaseChart(FASE_X_MESE[lastKey], lastLabel);
  else renderFaseChart(FASE_DATA, 'storico');

  let selectedIdx = SPARK.values.length - 1;

  SPARK.values.forEach((v, i) => {
    const bar = document.createElement('div');
    const isLast = i === SPARK.values.length - 1;
    bar.className = 'spark-bar' + (isLast ? ' selected' : '');
    bar.style.height = (v / max * 100) + '%';
    bar.title = `${SPARK.labels[i]}: ${v} h perse — clicca per vedere le fasi`;
    bar.dataset.idx = i;
    bar.addEventListener('click', () => {
      document.querySelectorAll('#sparkEl .spark-bar').forEach(b => b.classList.remove('selected'));
      bar.classList.add('selected');
      selectedIdx = i;
      const key   = SPARK.keys ? SPARK.keys[i] : null;
      const label = SPARK.labels[i] || '';
      if (key && FASE_X_MESE[key]) renderFaseChart(FASE_X_MESE[key], label);
      else renderFaseChart(FASE_DATA, 'storico');
    });
    el.appendChild(bar);
  });

  SPARK.labels.forEach(l => {
    const s = document.createElement('span');
    s.textContent = l;
    lbl.appendChild(s);
  });
})();

// ── Forecast ──────────────────────────────────────────────────────────────────
(function(){
  const el = document.getElementById('forecastGrid');
  FORECAST.forEach(it => {
    const artKey  = it.ARTICOLO_grouped || it.ARTICOLO || '—';
    const faseKey = it.FASE || '—';
    const probInt = Math.round(it.prob);
    el.innerHTML += `<div class="fc-item ${it.risk_cls}">
      <div class="fc-art">${artKey}</div>
      <div class="fc-fase">${faseKey} &nbsp;·&nbsp; ${it.n} lavoraz.</div>
      <div class="fc-bar"><div class="fc-fill" style="width:${probInt}%"></div></div>
      <div class="fc-txt">Prob. anomalia: <strong>${probInt}%</strong></div>
    </div>`;
  });
  if (!FORECAST.length) {
    el.innerHTML = '<div style="font-family:var(--mono);font-size:.72rem;color:var(--muted)">Dati insufficienti per la previsione.</div>';
  }
})();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 5. ASSEMBLAGGIO HTML
# ─────────────────────────────────────────────────────────────────────────────

def _needle_deg(indice_medio: float, i_min: float = 1.0, i_max: float = 2.0) -> int:
    """
    Converte l'indice di inefficienza medio in gradi di rotazione per l'ago gauge.
    Arco: da -90° (i_min) a +90° (i_max), la soglia anomalia è a 0°.
    """
    ratio = (indice_medio - i_min) / max(i_max - i_min, 0.001)
    ratio = max(0.0, min(1.0, ratio))
    return int(-90 + ratio * 180)


def genera_html(kpi: dict, cal: list, wo: list, art: dict, fase: dict,
                spark: dict, forecast: list, soglia: float,
                model_name: str = "XGBoost",
                indice_medio_globale: float = 1.0,
                fase_x_mese: dict = None) -> str:

    # Usa indice medio calcolato sull'intero dataset (passato dall'esterno)
    indice_medio = round(float(indice_medio_globale), 3)

    n_normali = kpi["n_normali"]
    perc_norm = round(n_normali / max(kpi["n_tot"], 1) * 100, 1)

    # delta badge direction
    delta_str = kpi["delta_anomalie"]
    delta_cls = "up" if delta_str.startswith("+") else "down" if delta_str.startswith("-") else "neutral"

    html = HTML_TEMPLATE
    replacements = {
        "__GENERATO_IL__":  kpi["generato_il"],
        "__N_TOT__":        str(kpi["n_tot"]),
        "__PERIODO__":      kpi["periodo"],
        "__N_ANOMALIE__":   str(kpi["n_anomalie"]),
        "__N_NORMALI__":    str(n_normali),
        "__PERC_NORMALI__": str(perc_norm),
        "__TASSO__":        str(kpi["tasso"]),
        "__ORE_PERSE__":    str(kpi["ore_perse"]),
        "__DELTA_ANOMALIE__": delta_str,
        "__SOGLIA__":       f"{soglia:.4f}",
        "__INDICE_MEDIO__": str(indice_medio),
        "__NEEDLE_DEG__":   str(_needle_deg(indice_medio)),
        "__MODEL_NAME__":   model_name,
        "__CAL_JSON__":          json.dumps(cal),
        "__WO_JSON__":           json.dumps(wo),
        "__ART_JSON__":          json.dumps(art),
        "__FASE_JSON__":         json.dumps(fase),
        "__SPARK_JSON__":        json.dumps(spark),
        "__FORECAST_JSON__":     json.dumps(forecast),
        "__FASE_X_MESE_JSON__":  json.dumps(fase_x_mese or {}),
    }
    for k, v in replacements.items():
        html = html.replace(k, v)
    return html


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def genera_dashboard(
    path_input:  str = DATA_PATH,
    path_output: str = OUTPUT_HTML,
    model_path:  str = MODEL_PATH,
    params_path: str = PARAMS_PATH,
):
    print(f"[1/6] Caricamento modello da:  {model_path}")
    model  = joblib.load(model_path)
    params = joblib.load(params_path)
    soglia = float(params.get("soglia_anomalia", 1.43))
    model_name = type(model.named_steps["model"]).__name__

    print(f"[2/6] Caricamento dati da:     {path_input}")
    df = carica_e_prepara(path_input, params)

    print(f"[3/6] Calcolo predizioni ...")
    df = predici(model, df)

    print(f"[4/6] Aggregazione dati ...")
    kpi          = kpi_globali(df)
    cal          = calendario_tutti_mesi(df)
    wo           = tabella_wo_tutti(df, n=50)
    art          = anomalie_per_articolo(df)
    fase         = anomalie_per_fase(df)
    spark        = sparkline_ore_perse(df)
    forecast     = forecast_mese_successivo(df)
    fase_x_mese  = fase_per_mese(df)

    # Indice medio su tutti i dati (non solo anomalie)
    indice_medio_globale = float(df["Indice_Inefficienza"].mean()) if "Indice_Inefficienza" in df.columns else 1.0

    print(f"[5/6] Generazione HTML ...")
    html = genera_html(kpi, cal, wo, art, fase, spark, forecast, soglia, model_name,
                       indice_medio_globale=indice_medio_globale,
                       fase_x_mese=fase_x_mese)

    print(f"[6/6] Salvataggio in:          {path_output}")
    os.makedirs(os.path.dirname(path_output), exist_ok=True)
    with open(path_output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅  Dashboard generata: {path_output}")
    print(f"    Periodo analizzato:  {kpi['periodo']}")
    print(f"    Anomalie rilevate:   {kpi['n_anomalie']} / {kpi['n_tot']} ({kpi['tasso']}%)")
    print(f"    Ore perse stimate:   {kpi['ore_perse']} h")


if __name__ == "__main__":
    genera_dashboard()