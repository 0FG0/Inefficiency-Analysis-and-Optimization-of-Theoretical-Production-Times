"""
dashboard_oee.py
────────────────
Genera una dashboard HTML interattiva dall'output di oee_calculator.

Come usarlo:
    python src/dashboard_oee.py
    → outputs/dashboard_oee.html  (apribile nel browser senza server)

Dipendenze: pandas, jinja2  (pip install jinja2)
Grafici: Chart.js via CDN (nessuna installazione)
"""

import os
import sys
import json
import warnings
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from OEE_calculator import calcola_oee, SOGLIA_ACCETTABILE, SOGLIA_OTTIMO
from OEE_feature_engineering import aggiungi_feature_oee, get_feature_cols_oee

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT      = os.path.dirname(os.path.dirname(BASE_DIR))
INPUT_CSV         = os.path.join(PROJECT_ROOT, "data", "processed", "koepfer_160_2.csv")
OUTPUT_HTML       = os.path.join(PROJECT_ROOT, "outputs", "dashboard_oee.html")
MODEL_REG_PATH    = os.path.join(PROJECT_ROOT, "models", "regression", "best_regressione_oee.pkl")
PARAMS_REG_PATH   = os.path.join(PROJECT_ROOT, "models", "regression", "parametri_regressione_oee.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepara_dati(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", encoding="utf-8")
    df = calcola_oee(df)
    df = df.dropna(subset=["OEE"])
    if "Data_Ora_Fine" in df.columns:
        df["Data_Ora_Fine"] = pd.to_datetime(df["Data_Ora_Fine"], errors="coerce")
        df["Mese"]      = df["Data_Ora_Fine"].dt.to_period("M").astype(str)
        df["Settimana"] = df["Data_Ora_Fine"].dt.to_period("W").astype(str)
    return df


def trend_mensile(df: pd.DataFrame) -> dict:
    if "Mese" not in df.columns:
        return {}
    g = df.groupby("Mese")[["OEE","OEE_Disponibilita","OEE_Performance","OEE_Qualita"]].mean().round(4)
    return {
        "labels": g.index.tolist(),
        "oee":    g["OEE"].tolist(),
        "disp":   g["OEE_Disponibilita"].tolist(),
        "perf":   g["OEE_Performance"].tolist(),
        "qual":   g["OEE_Qualita"].tolist(),
    }


def top_wo_critici(df: pd.DataFrame, n: int = 10) -> list:
    cols = [c for c in ["WO", "FASE", "ARTICOLO", "Data_Ora_Fine", "OEE", "OEE_Classe",
                        "OEE_Disponibilita", "OEE_Performance", "OEE_Qualita"] if c in df.columns]
    mask = df["OEE"] < SOGLIA_ACCETTABILE if "OEE" in df.columns else pd.Series(True, index=df.index)
    out = df.loc[mask, cols].dropna(subset=["OEE"]).sort_values("OEE").head(n).round(4).copy()
    for col in ["Data_Ora_Fine", "OEE_Classe"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out.to_dict(orient="records")


def distribuzione_classi(df: pd.DataFrame) -> dict:
    if "OEE_Classe" not in df.columns:
        return {}
    vc = df["OEE_Classe"].astype(str).value_counts()
    order = ["Ottimo", "Accettabile", "Critico"]
    labels = [l for l in order if l in vc.index]
    values = [int(vc.get(l, 0)) for l in labels]
    return {"labels": labels, "values": values}


def oee_per_articolo(df: pd.DataFrame, n: int = 15) -> dict:
    if "ARTICOLO" not in df.columns:
        return {}
    g = df.groupby("ARTICOLO")["OEE"].mean().dropna().sort_values().head(n).round(4)
    return {"labels": g.index.tolist(), "values": g.values.tolist()}


def forecast_oee_per_articolo_fase(df: pd.DataFrame) -> list:
    """
    Per ogni combinazione articolo+fase usa il modello XGBoost di regressione
    per stimare l'OEE atteso nel mese successivo.
    Fallback alla media storica se il modello non e' disponibile.
    """
    if "ARTICOLO" not in df.columns or "OEE" not in df.columns:
        return []

    # ─ Carica modello e parametri ──────────────────────────────────────────
    model, params = None, {}
    try:
        model  = joblib.load(MODEL_REG_PATH)
        params = joblib.load(PARAMS_REG_PATH)
    except Exception:
        pass

    # ─ Feature engineering sullo storico ────────────────────────────────
    articoli_top = params.get("articoli_top", df["ARTICOLO"].value_counts().nlargest(30).index.tolist())
    df_fe = df.copy()
    df_fe["ARTICOLO_grouped"] = df_fe["ARTICOLO"].where(df_fe["ARTICOLO"].isin(articoli_top), other="ALTRO")
    df_fe = aggiungi_feature_oee(df_fe, storico=df_fe)

    feature_cols = params.get("feature_cols", get_feature_cols_oee())
    feature_cols = [c for c in feature_cols if c in df_fe.columns]

    # Feature temporali del mese successivo
    today     = date.today()
    if today.month == 12:
        next_m = date(today.year + 1, 1, 1)
    else:
        next_m = date(today.year, today.month + 1, 1)
    next_month_num = next_m.month
    next_dow       = next_m.weekday()
    next_week      = next_m.isocalendar()[1]

    group_cols = ["ARTICOLO", "FASE"] if "FASE" in df_fe.columns else ["ARTICOLO"]

    # Medie componenti storiche (per display)
    comp_cols = [c for c in ["OEE_Disponibilita", "OEE_Performance", "OEE_Qualita"] if c in df.columns]
    comp_means = df.groupby(group_cols)[comp_cols].mean() if comp_cols else None

    result = []
    for keys, group in df_fe.groupby(group_cols):
        if len(group) < 1:
            continue

        # Riga sintetica: mediana delle feature numeriche, moda per categoriali
        row = {}
        for col in feature_cols:
            if col not in group.columns:
                row[col] = np.nan
                continue
            vals = group[col].dropna()
            if len(vals) == 0:
                row[col] = np.nan
            elif pd.api.types.is_numeric_dtype(group[col]):
                row[col] = float(vals.median())
            else:
                row[col] = vals.mode().iloc[0] if not vals.mode().empty else "MISSING"

        # Override feature temporali con il mese successivo
        for tcol, tval in [("mese", next_month_num), ("giorno_settimana", next_dow), ("settimana_anno", next_week)]:
            if tcol in row:
                row[tcol] = tval

        # Lag e rolling dall'ultimo storico del gruppo
        if "Data_Ora_Fine" in group.columns:
            recent_oee = group.sort_values("Data_Ora_Fine")["OEE"].dropna().tail(3).tolist()
        else:
            recent_oee = group["OEE"].dropna().tail(3).tolist()
        for i, lag_col in enumerate(["lag_1_oee", "lag_2_oee", "lag_3_oee"], start=1):
            if lag_col in row:
                idx = len(recent_oee) - i
                row[lag_col] = recent_oee[idx] if idx >= 0 else np.nan
        if "rolling_oee_3" in row:
            row["rolling_oee_3"] = float(np.mean(recent_oee)) if recent_oee else np.nan

        X_row = pd.DataFrame([row])[feature_cols]
        for col in X_row.select_dtypes(include="object").columns:
            X_row[col] = X_row[col].fillna("MISSING").astype(str)

        # Predizione — modello ML o fallback alla media
        use_model = False
        if model is not None:
            try:
                oee_pred = float(np.clip(model.predict(X_row)[0], 0.0, 1.0))
                use_model = True
            except Exception:
                oee_pred = float(group["OEE"].mean())
        else:
            oee_pred = float(group["OEE"].mean())

        oee_pct  = round(oee_pred * 100, 1)
        risk_cls = "fc-green" if oee_pct >= 85 else "fc-warn" if oee_pct >= 65 else "fc-danger"

        art = keys[0] if isinstance(keys, tuple) else keys
        entry = {
            "articolo":  str(art),
            "oee":       oee_pct,
            "n":         int(len(group)),
            "risk_cls":  risk_cls,
            "use_model": use_model,
        }

        # Componenti storiche per display
        if comp_means is not None:
            try:
                comp_row = comp_means.loc[keys]
                entry["disp"] = round(float(comp_row.get("OEE_Disponibilita", 0) or 0) * 100, 1)
                entry["perf"] = round(float(comp_row.get("OEE_Performance",  0) or 0) * 100, 1)
                entry["qual"] = round(float(comp_row.get("OEE_Qualita",      0) or 0) * 100, 1)
            except KeyError:
                entry["disp"] = entry["perf"] = entry["qual"] = None

        if "FASE" in group_cols:
            fase = keys[1] if isinstance(keys, tuple) else "—"
            entry["fase"] = str(fase) if fase is not None and pd.notna(fase) else "—"

        result.append(entry)

    return result


def tutti_record_per_kpi_oee(df: pd.DataFrame) -> list:
    records = []
    has_date = "Data_Ora_Fine" in df.columns
    for _, row in df.iterrows():
        d = None
        if has_date and pd.notna(row["Data_Ora_Fine"]):
            d = row["Data_Ora_Fine"].strftime("%Y-%m-%d")
        def _v(col):
            v = row.get(col)
            return round(float(v), 4) if v is not None and pd.notna(v) else None
        records.append({"d": d, "oee": _v("OEE"), "disp": _v("OEE_Disponibilita"),
                        "perf": _v("OEE_Performance"), "qual": _v("OEE_Qualita")})
    return records


def kpi_globali(df: pd.DataFrame) -> dict:
    return {
        "oee_medio":   round(df["OEE"].mean() * 100, 1)               if "OEE" in df.columns else "-",
        "disp_media":  round(df["OEE_Disponibilita"].mean() * 100, 1) if "OEE_Disponibilita" in df.columns else "-",
        "perf_media":  round(df["OEE_Performance"].mean() * 100, 1)   if "OEE_Performance" in df.columns else "-",
        "qual_media":  round(df["OEE_Qualita"].mean() * 100, 1)        if "OEE_Qualita" in df.columns else "-",
        "n_critici":   int((df["OEE"] < SOGLIA_ACCETTABILE).sum())     if "OEE" in df.columns else 0,
        "n_wc":        int((df["OEE"] >= SOGLIA_OTTIMO).sum())          if "OEE" in df.columns else 0,
        "totale_wo":   len(df),
        "generato_il": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OEE Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600;700&display=swap');

  :root {{
    --bg:       #0d0f14;
    --surface:  #141720;
    --border:   #1e2130;
    --accent:   #00e5a0;
    --warn:     #f5a623;
    --danger:   #e5004c;
    --text:     #e8eaf0;
    --muted:    #5a6070;
    --radius:   12px;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    min-height: 100vh;
    min-width: 1200px;
    overflow-x: hidden;
  }}

  .page-wrapper {{
    max-width: 1400px;
    min-width: 1200px;
    margin: 0 auto;
  }}

  header {{
    padding: 28px 40px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 16px;
  }}
  header h1 {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
  }}
  header span {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
  }}

  .grid {{
    display: grid;
    gap: 16px;
    padding: 24px 40px;
  }}

  .kpi-row      {{ grid-template-columns: repeat(4, 1fr); }}
  .charts-row   {{ grid-template-columns: 2fr 1fr; }}
  /* WO critici: occupa tutta la riga */
  .wo-row       {{ grid-template-columns: 1fr; }}


  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
  }}

  .kpi-card {{
    position: relative;
    overflow: hidden;
  }}
  .kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
  }}
  .kpi-card.warn::before   {{ background: var(--warn); }}
  .kpi-card.danger::before {{ background: var(--danger); }}

  .kpi-label {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }}
  .kpi-value {{
    font-family: 'DM Mono', monospace;
    font-size: 36px;
    font-weight: 500;
    line-height: 1;
    color: var(--accent);
  }}
  .kpi-card.warn  .kpi-value {{ color: var(--warn); }}
  .kpi-card.danger .kpi-value {{ color: var(--danger); }}
  .kpi-sub {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
  }}

  .card-title {{
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
  }}

  canvas {{ width: 100% !important; }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  th {{
    text-align: left;
    padding: 6px 10px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    background: var(--surface);
    z-index: 1;
  }}
  td {{
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}

  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
  }}
  .badge-critico    {{ background: rgba(229,0,76,0.15);  color: var(--danger); }}
  .badge-accettabile{{ background: rgba(245,166,35,0.15); color: var(--warn); }}
  .badge-wc         {{ background: rgba(0,229,160,0.15);  color: var(--accent); }}

  .oee-bar {{
    display: inline-block;
    height: 6px;
    border-radius: 3px;
    background: var(--danger);
    vertical-align: middle;
    margin-right: 6px;
    transition: width 0.3s;
  }}
  .oee-bar.ok  {{ background: var(--accent); }}
  .oee-bar.mid {{ background: var(--warn); }}

  /* WO table scrollable, mostra almeno 4 righe */
  .wo-table-wrap {{
    overflow-x: auto;
    overflow-y: auto;
    /* header ~32px + 4 righe ~38px ≈ 184px */
    max-height: 220px;
    scrollbar-width: thin;
    scrollbar-color: #1e2130 transparent;
  }}

  footer {{
    padding: 16px 40px;
    color: var(--muted);
    font-size: 11px;
    border-top: 1px solid var(--border);
  }}

  .period-btns {{
    display: flex;
    gap: .5rem;
    padding: 16px 40px 0;
    flex-wrap: wrap;
  }}
  .period-btn {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: .08em;
    padding: .35rem .9rem;
    border-radius: 6px;
    cursor: pointer;
    transition: all .18s;
    text-transform: uppercase;
  }}
  .period-btn:hover  {{ border-color: var(--accent); color: var(--accent); }}
  .period-btn.active {{ background: rgba(0,229,160,0.1); border-color: var(--accent); color: var(--accent); }}

  /* ── Forecast OEE ───────────────────────────────────────────────────────── */
  .fc-select-row {{ display:flex; gap:16px; flex-wrap:wrap; align-items:flex-end; margin-bottom:20px; }}
  .fc-select-group {{ display:flex; flex-direction:column; gap:6px; }}
  .fc-select-label {{ font-size:10px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); }}
  select.fc-select {{
    background: var(--bg); border: 1px solid var(--border); color: var(--text);
    font-family: 'DM Mono', monospace; font-size: 12px; padding: 7px 12px;
    border-radius: 6px; cursor: pointer; min-width: 220px; outline: none;
    appearance: none;
  }}
  select.fc-select:focus {{ border-color: var(--accent); }}
  .fc-result {{ display:none; animation: fadeIn .25s ease; }}
  .fc-result.visible {{ display:block; }}
  @keyframes fadeIn {{ from{{opacity:0;transform:translateY(6px)}} to{{opacity:1;transform:translateY(0)}} }}
  .fc-oee-hero {{ display:flex; align-items:baseline; gap:10px; margin-bottom:16px; }}
  .fc-oee-big {{ font-family:'DM Mono',monospace; font-size:52px; font-weight:500; line-height:1; }}
  .fc-oee-big.fc-green  {{ color:var(--accent); }}
  .fc-oee-big.fc-warn   {{ color:var(--warn); }}
  .fc-oee-big.fc-danger {{ color:var(--danger); }}
  .fc-oee-lbl {{ font-size:12px; color:var(--muted); }}
  .fc-comp-row {{ display:flex; flex-direction:column; gap:10px; margin-bottom:14px; }}
  .fc-comp-item {{ display:flex; align-items:center; gap:12px; }}
  .fc-comp-name {{ font-size:10px; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; color:var(--muted); width:110px; flex-shrink:0; }}
  .fc-comp-track {{ flex:1; background:var(--border); border-radius:3px; height:8px; overflow:hidden; }}
  .fc-comp-fill {{ height:100%; border-radius:3px; transition:width .6s cubic-bezier(.22,1,.36,1); }}
  .fc-comp-fill.fc-green  {{ background:var(--accent); }}
  .fc-comp-fill.fc-warn   {{ background:var(--warn); }}
  .fc-comp-fill.fc-danger {{ background:var(--danger); }}
  .fc-comp-val {{ font-family:'DM Mono',monospace; font-size:11px; color:var(--muted); width:44px; text-align:right; flex-shrink:0; }}
  .fc-meta {{ font-size:11px; color:var(--muted); margin-bottom:2px; }}
  .fc-note {{ margin-top:18px; padding:10px 14px; background:var(--bg); border-left:3px solid var(--warn); border-radius:0 4px 4px 0; font-size:11px; color:var(--muted); line-height:1.7; font-family:'DM Mono',monospace; }}
  .fc-note strong {{ color:var(--warn); }}

  @keyframes distPulse {{
    0%   {{ opacity:1; transform:scale(1); }}
    35%  {{ opacity:0.35; transform:scale(0.96); }}
    100% {{ opacity:1; transform:scale(1); }}
  }}
  .dist-updating {{
    animation: distPulse 0.42s ease-out forwards;
    transform-origin: center;
  }}
</style>
</head>
<body>
<div class="page-wrapper">

<header>
  <h1>OEE Dashboard</h1>
  <span>Generato il {generato_il} &nbsp;·&nbsp; {totale_wo} WO analizzati</span>
</header>

<div class="period-btns">
  <button class="period-btn active" id="btn_all" onclick="filterByPeriod('all')">Tutto</button>
  <button class="period-btn" id="btn_1y" onclick="filterByPeriod('1y')">Ultimo anno</button>
  <button class="period-btn" id="btn_3m" onclick="filterByPeriod('3m')">Ultimi 3 mesi</button>
  <button class="period-btn" id="btn_1m" onclick="filterByPeriod('1m')">Ultimo mese</button>
</div>

<!-- KPI -->
<div class="grid kpi-row">
  <div class="card kpi-card">
    <div class="kpi-label">OEE Medio</div>
    <div class="kpi-value" id="kpiOee">{oee_medio}%</div>
    <div class="kpi-sub">Disponibilità × Performance × Qualità</div>
  </div>
  <div class="card kpi-card">
    <div class="kpi-label">Disponibilità</div>
    <div class="kpi-value" id="kpiDisp">{disp_media}%</div>
    <div class="kpi-sub">Uptime effettivo / programmato</div>
  </div>
  <div class="card kpi-card warn">
    <div class="kpi-label">Performance</div>
    <div class="kpi-value" id="kpiPerf">{perf_media}%</div>
    <div class="kpi-sub">Velocità reale / velocità teorica</div>
  </div>
  <div class="card kpi-card">
    <div class="kpi-label">Qualità</div>
    <div class="kpi-value" id="kpiQual">{qual_media}%</div>
    <div class="kpi-sub">Pezzi buoni / tot pezzi</div>
  </div>
</div>

<!-- TREND + DISTRIBUZIONE -->
<div class="grid charts-row">
  <div class="card">
    <div class="card-title">Trend OEE — dettaglio mensile</div>
    <div id="trendContainer" style="display:flex;flex-direction:column;gap:20px;max-height:330px;overflow-y:auto;padding-right:6px;scrollbar-width:thin;scrollbar-color:#1e2130 transparent;"></div>
  </div>
  <div class="card">
    <div class="card-title">Distribuzione classi OEE</div>
    <canvas id="distChart" height="110"></canvas>
  </div>
</div>

<!-- WO CRITICI — riga intera -->
<div class="grid wo-row">
  <div class="card">
    <div class="card-title">WO critici — OEE sotto soglia</div>
    <div class="wo-table-wrap">
      <table>
        <thead>
          <tr>
            <th>WO</th><th>Articolo</th><th>Fase</th><th>OEE</th><th>D</th><th>P</th><th>Q</th><th>Classe</th>
          </tr>
        </thead>
        <tbody id="alertTable"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ARTICOLI -->
<div class="grid wo-row">
  <div class="card">
    <div class="card-title">Articoli con OEE più basso</div>
    <canvas id="artChart" height="180"></canvas>
  </div>
</div>

<!-- FORECAST OEE -->
<div class="grid wo-row">
  <div class="card">
    <div class="card-title">Previsione OEE — mese successivo</div>
    <div class="fc-select-row">
      <div class="fc-select-group">
        <span class="fc-select-label">Articolo</span>
        <select class="fc-select" id="fcArticolo" onchange="fcOnArticolo()">
          <option value="">— seleziona articolo —</option>
        </select>
      </div>
      <div class="fc-select-group">
        <span class="fc-select-label">Fase</span>
        <select class="fc-select" id="fcFase" onchange="fcOnFase()" disabled>
          <option value="">— seleziona fase —</option>
        </select>
      </div>
    </div>
    <div class="fc-result" id="fcResult">
      <div class="fc-oee-hero">
        <span class="fc-oee-big" id="fcOeeBig"></span>
        <span class="fc-oee-lbl">OEE previsto</span>
      </div>
      <div class="fc-comp-row">
        <div class="fc-comp-item">
          <span class="fc-comp-name">Disponibilità</span>
          <div class="fc-comp-track"><div class="fc-comp-fill" id="fcDispFill" style="width:0%"></div></div>
          <span class="fc-comp-val" id="fcDispVal"></span>
        </div>
        <div class="fc-comp-item">
          <span class="fc-comp-name">Performance</span>
          <div class="fc-comp-track"><div class="fc-comp-fill" id="fcPerfFill" style="width:0%"></div></div>
          <span class="fc-comp-val" id="fcPerfVal"></span>
        </div>
        <div class="fc-comp-item">
          <span class="fc-comp-name">Qualità</span>
          <div class="fc-comp-track"><div class="fc-comp-fill" id="fcQualFill" style="width:0%"></div></div>
          <span class="fc-comp-val" id="fcQualVal"></span>
        </div>
      </div>
      <div class="fc-meta" id="fcMeta"></div>
      <div id="fcBadge" style="margin-top:8px"></div>
    </div>
    <div class="fc-note">
      <strong>Come funziona:</strong><br>
      Seleziona un articolo e una fase: la previsione OEE viene calcolata con il modello <strong>XGBoost</strong> (R²&nbsp;=&nbsp;0.94, MAE&nbsp;=&nbsp;3.4%) addestrato sullo storico della macchina.<br>
      Il modello usa feature di pianificazione (tempi, quantità, fase, articolo, storico recente) proiettate al mese successivo.<br>
      Risponde a: <em>Se il mese prossimo si produce questo articolo in questa fase, quale OEE posso aspettarmi?</em><br>
      <span style="color:var(--accent)">Ottimo &ge; 85%</span> &nbsp;&middot;&nbsp;
      <span style="color:var(--warn)">Accettabile 65–84%</span> &nbsp;&middot;&nbsp;
      <span style="color:var(--danger)">Critico &lt; 65%</span>
    </div>
  </div>
</div>

<footer>OEE = Disponibilità × Performance × Qualità &nbsp;|&nbsp; Ottimo ≥ 85% &nbsp;|&nbsp; Accettabile ≥ 65% &nbsp;|&nbsp; Critico &lt; 65%</footer>
</div>

<script>
const TREND   = {trend_json};
const DIST    = {dist_json};
const ARTS    = {art_json};
const CRITICI = {critici_json};
const KPI     = {kpi_json};

// ── Trend Charts (mensile) ─────────────────────────────────────────────────
if (TREND.labels && TREND.labels.length) {{
  const trendContainer = document.getElementById('trendContainer');
  for (let i = TREND.labels.length - 1; i >= 0; i--) {{
    const mese = TREND.labels[i];
    const wrapper = document.createElement('div');
    const periodLabel = document.createElement('div');
    periodLabel.style.cssText = 'font-size:15px;color:#ffffff;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;';
    periodLabel.textContent = mese;
    const canvas = document.createElement('canvas');
    canvas.height = 280;
    wrapper.appendChild(periodLabel);
    wrapper.appendChild(canvas);
    trendContainer.appendChild(wrapper);
    const vals = [TREND.oee[i], TREND.disp[i], TREND.perf[i], TREND.qual[i]].filter(v => v != null);
    const yMin = Math.max(0, Math.floor(Math.min(...vals) * 20) / 20 - 0.03);
    const yMax = Math.min(1, Math.ceil(Math.max(...vals) * 20) / 20 + 0.02);
    new Chart(canvas, {{
      type: 'bar',
      data: {{
        labels: ['OEE', 'Disponibilità', 'Performance', 'Qualità'],
        datasets: [{{
          data: [TREND.oee[i], TREND.disp[i], TREND.perf[i], TREND.qual[i]],
          backgroundColor: ['rgba(0,229,160,0.7)', 'rgba(91,138,245,0.7)', 'rgba(245,166,35,0.7)', 'rgba(199,125,255,0.7)'],
          borderColor:     ['#00e5a0', '#5b8af5', '#f5a623', '#c77dff'],
          borderWidth: 1,
          borderRadius: 4,
        }}]
      }},
      options: {{
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks:{{ color:'#8890a4', font:{{size:11}} }}, grid:{{ display:false }} }},
          y: {{ min: yMin, max: yMax, ticks:{{ color:'#5a6070', font:{{size:10}}, callback: v => (v*100).toFixed(0)+'%' }}, grid:{{ color:'#1e2130' }} }}
        }}
      }}
    }});
  }}
}}

// ── Distribuzione Classi ───────────────────────────────────────────────────
{{
  var _distLabels = ['Ottimo', 'Accettabile', 'Critico'];
  var _distInit = _distLabels.map(function(k) {{
    var idx = (DIST.labels || []).indexOf(k);
    return idx >= 0 ? DIST.values[idx] : 0;
  }});
  window._distChart = new Chart(document.getElementById('distChart'), {{
    type: 'doughnut',
    data: {{
      labels: _distLabels,
      datasets: [{{ data: _distInit, backgroundColor: ['#00e5a0','#f5a623','#e5004c'], borderWidth: 0 }}]
    }},
    options: {{
      animation: {{ duration: 600, easing: 'easeInOutQuart' }},
      plugins: {{
        legend: {{ position:'bottom', labels:{{ color:'#8890a4', font:{{size:11}}, padding:16 }} }}
      }},
      cutout: '65%'
    }}
  }});
}}

// ── Articoli chart ─────────────────────────────────────────────────────────
if (ARTS.labels && ARTS.labels.length) {{
  const colors = ARTS.values.map(v => v < 0.65 ? '#e5004c' : v < 0.85 ? '#f5a623' : '#00e5a0');
  new Chart(document.getElementById('artChart'), {{
    type: 'bar',
    data: {{
      labels: ARTS.labels,
      datasets: [{{ label: 'OEE medio', data: ARTS.values, backgroundColor: colors, borderRadius: 4, borderSkipped: false }}]
    }},
    options: {{
      indexAxis: 'y',
      plugins: {{ legend:{{ display:false }} }},
      scales: {{
        x: {{ min:0, max:1, ticks:{{ color:'#5a6070', font:{{size:10}}, callback: v => (v*100)+'%' }}, grid:{{ color:'#1e2130' }} }},
        y: {{ ticks:{{ color:'#8890a4', font:{{size:10}} }}, grid:{{ display:false }} }}
      }}
    }}
  }});
}}

// ── Tabella WO critici ─────────────────────────────────────────────────────
const tbody = document.getElementById('alertTable');
CRITICI.forEach(r => {{
  const oee = r.OEE ?? '-';
  const pct = v => typeof v === 'number' ? (v*100).toFixed(1)+'%' : '-';
  const classe = (r.OEE_Classe||'').toLowerCase();
  const badge = (classe === 'ottimo' || classe.includes('world')) ? 'badge-wc' : classe === 'accettabile' ? 'badge-accettabile' : 'badge-critico';
  const barW = typeof oee === 'number' ? Math.round(oee*60) : 0;
  const barClass = oee >= 0.85 ? 'ok' : oee >= 0.65 ? 'mid' : '';
  tbody.innerHTML += `<tr>
    <td>${{r.WO||'-'}}</td>
    <td>${{r.ARTICOLO||'-'}}</td>
    <td>${{r.FASE||'-'}}</td>
    <td><span class="oee-bar ${{barClass}}" style="width:${{barW}}px"></span>${{pct(oee)}}</td>
    <td>${{pct(r.OEE_Disponibilita)}}</td>
    <td>${{pct(r.OEE_Performance)}}</td>
    <td>${{pct(r.OEE_Qualita)}}</td>
    <td><span class="badge ${{badge}}">${{r.OEE_Classe||'-'}}</span></td>
  </tr>`;
}});

// ── Forecast OEE ─────────────────────────────────────────────────────────
(function() {{
  var FC_DATA = {forecast_oee_json};
  var lookup = {{}};
  FC_DATA.forEach(function(it) {{
    var art = it.articolo;
    if (!lookup[art]) lookup[art] = {{}};
    var fase = it.fase !== undefined ? it.fase : '__NOFASE__';
    lookup[art][fase] = it;
  }});
  window._fcLookup = lookup;
  var sel = document.getElementById('fcArticolo');
  Object.keys(lookup).sort().forEach(function(a) {{
    var opt = document.createElement('option');
    opt.value = a; opt.textContent = a;
    sel.appendChild(opt);
  }});
}})();

function fcOnArticolo() {{
  var art = document.getElementById('fcArticolo').value;
  var faseSel = document.getElementById('fcFase');
  faseSel.innerHTML = '<option value="">— seleziona fase —</option>';
  document.getElementById('fcResult').classList.remove('visible');
  if (!art) {{ faseSel.disabled = true; return; }}
  faseSel.disabled = false;
  var fasi = Object.keys(window._fcLookup[art] || {{}}).sort();
  fasi.forEach(function(f) {{
    var opt = document.createElement('option');
    opt.value = f;
    opt.textContent = f === '__NOFASE__' ? '(tutte)' : f;
    faseSel.appendChild(opt);
  }});
  if (fasi.length === 1) {{ faseSel.value = fasi[0]; fcOnFase(); }}
}}

function fcOnFase() {{
  var art  = document.getElementById('fcArticolo').value;
  var fase = document.getElementById('fcFase').value;
  if (!art || !fase) return;
  var it = (window._fcLookup[art] || {{}})[fase];
  if (!it) return;
  var big = document.getElementById('fcOeeBig');
  big.textContent = it.oee + '%';
  big.className = 'fc-oee-big ' + it.risk_cls;
  function setComp(fillId, valId, v) {{
    var fill = document.getElementById(fillId);
    var val  = document.getElementById(valId);
    var c = v >= 85 ? 'fc-green' : v >= 65 ? 'fc-warn' : 'fc-danger';
    fill.style.width = v + '%';
    fill.className = 'fc-comp-fill ' + c;
    val.textContent = v + '%';
  }}
  setComp('fcDispFill', 'fcDispVal', it.disp);
  setComp('fcPerfFill', 'fcPerfVal', it.perf);
  setComp('fcQualFill', 'fcQualVal', it.qual);
  document.getElementById('fcMeta').textContent = it.n + ' lavorazioni osservate nello storico';
  var badge = document.getElementById('fcBadge');
  badge.style.cssText = 'display:inline-block;font-size:10px;padding:2px 8px;border-radius:4px;letter-spacing:.06em;margin-top:2px';
  if (it.use_model) {{
    badge.style.background = 'rgba(0,229,160,0.12)';
    badge.style.border = '1px solid var(--accent)';
    badge.style.color = 'var(--accent)';
    badge.textContent = 'XGBoost R\u00b2=0.94';
  }} else {{
    badge.style.background = 'rgba(245,166,35,0.12)';
    badge.style.border = '1px solid var(--warn)';
    badge.style.color = 'var(--warn)';
    badge.textContent = 'media storica (modello non disponibile)';
  }}
  document.getElementById('fcResult').classList.add('visible');
}}

// ── Filtro periodo ─────────────────────────────────────────────────────────
const ALL_OEE = {all_records_json};
(function() {{
  var mx = null;
  ALL_OEE.forEach(function(r) {{ if (r.d) {{ var d = new Date(r.d); if (!mx || d > mx) mx = d; }} }});
  window._oeeMaxDate = mx;
}})();
function animateCount(id, target, suffix, decimals, dur) {{
  var el = document.getElementById(id);
  if (!el) return;
  var start = null;
  dur = dur || 900;
  var tok = {{}};
  el._atok = tok;
  function step(ts) {{
    if (el._atok !== tok) return;
    if (!start) start = ts;
    var prog = Math.min((ts - start) / dur, 1);
    var ease = 1 - Math.pow(1 - prog, 3);
    var cur = target * ease;
    el.textContent = (decimals > 0 ? cur.toFixed(decimals) : Math.round(cur)) + (suffix || '');
    if (prog < 1) requestAnimationFrame(step);
  }}
  requestAnimationFrame(step);
}}
function filterByPeriod(p) {{
  document.querySelectorAll('.period-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  var btn = document.getElementById('btn_' + p);
  if (btn) btn.classList.add('active');
  var ref = window._oeeMaxDate || new Date();
  var cutDate;
  var recs;
  if (p === 'all') {{
    recs = ALL_OEE;
  }} else {{
    if      (p === '1m') cutDate = new Date(ref.getFullYear(), ref.getMonth() - 1,  ref.getDate());
    else if (p === '3m') cutDate = new Date(ref.getFullYear(), ref.getMonth() - 3,  ref.getDate());
    else                 cutDate = new Date(ref.getFullYear() - 1, ref.getMonth(),  ref.getDate());
    recs = ALL_OEE.filter(function(r) {{ return r.d && new Date(r.d) >= cutDate; }});
  }}
  function avg(field) {{
    var vals = recs.map(function(r) {{ return r[field]; }}).filter(function(v) {{ return v != null; }});
    if (!vals.length) return null;
    return vals.reduce(function(a, b) {{ return a + b; }}, 0) / vals.length;
  }}
  var oee  = avg('oee');
  var disp = avg('disp');
  var perf = avg('perf');
  var qual = avg('qual');
  if (oee  != null) animateCount('kpiOee',  oee  * 100, '%', 1); else document.getElementById('kpiOee').textContent  = '-';
  if (disp != null) animateCount('kpiDisp', disp * 100, '%', 1); else document.getElementById('kpiDisp').textContent = '-';
  if (perf != null) animateCount('kpiPerf', perf * 100, '%', 1); else document.getElementById('kpiPerf').textContent = '-';
  if (qual != null) animateCount('kpiQual', qual * 100, '%', 1); else document.getElementById('kpiQual').textContent = '-';
  if (window._distChart) {{
    var ottimo = 0, accettabile = 0, critico = 0;
    recs.forEach(function(r) {{
      if (r.oee == null) return;
      if (r.oee >= 0.85) ottimo++;
      else if (r.oee >= 0.65) accettabile++;
      else critico++;
    }});
    var canvas = document.getElementById('distChart');
    canvas.classList.remove('dist-updating');
    void canvas.offsetWidth;
    canvas.classList.add('dist-updating');
    setTimeout(function() {{ canvas.classList.remove('dist-updating'); }}, 450);
    window._distChart.data.datasets[0].data = [ottimo, accettabile, critico];
    window._distChart.update();
  }}
}}
filterByPeriod('all');
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def genera_dashboard(path_input: str = INPUT_CSV, path_output: str = OUTPUT_HTML):
    print(f"Caricamento dati da: {path_input}")
    df = prepara_dati(path_input)

    kpi          = kpi_globali(df)
    trend        = trend_mensile(df)
    dist         = distribuzione_classi(df)
    arts         = oee_per_articolo(df)
    critici      = top_wo_critici(df)
    all_records  = tutti_record_per_kpi_oee(df)
    forecast_oee = forecast_oee_per_articolo_fase(df)

    html = HTML_TEMPLATE.format(
        **kpi,
        trend_json        = json.dumps(trend),
        dist_json         = json.dumps(dist),
        art_json          = json.dumps(arts),
        critici_json      = json.dumps(critici),
        kpi_json          = json.dumps(kpi),
        all_records_json  = json.dumps(all_records),
        forecast_oee_json = json.dumps(forecast_oee),
    )

    os.makedirs(os.path.dirname(path_output), exist_ok=True)
    with open(path_output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Dashboard generata: {path_output}")


if __name__ == "__main__":
    genera_dashboard()