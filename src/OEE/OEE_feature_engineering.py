# Prepara le feature necessarie per la regressione OEE futura.
import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering import add_time_features

def aggiungi_feature_oee(df: pd.DataFrame, storico: pd.DataFrame = None) -> pd.DataFrame:
    out = df.copy()
    ref = storico if storico is not None else df

    if "Data_Ora_Fine" in out.columns:
        out = add_time_features(out)

    tempo_tot = out["Tempo Attrezz. ORE"] + out["Tempo Lavoraz. ORE"]
    out["pct_setup"] = np.where(
        tempo_tot > 0,
        out["Tempo Attrezz. ORE"] / tempo_tot,
        np.nan
    )
    out["pct_fermi"] = np.where(
        tempo_tot > 0,
        out["Durata Soste Ore"].fillna(0) / tempo_tot,
        0.0
    )

    tempo_macc = out["Tempo Macc AS400 ORE"].replace(0, pd.NA)
    out["ratio_attr_macc"] = out["Tempo Attr AS400 ORE"] / tempo_macc

    if "Buon Tempo Ciclo ORE" in ref.columns:
        stats_ciclo = (
            ref.groupby(["ARTICOLO", "FASE"])["Buon Tempo Ciclo ORE"]
            .agg(media_ciclo_art="mean", std_ciclo_art="std")
            .reset_index()
        )
        out = out.merge(stats_ciclo, on=["ARTICOLO", "FASE"], how="left")
        out["std_ciclo_art"] = out["std_ciclo_art"].fillna(0)
    else:
        out["media_ciclo_art"] = np.nan
        out["std_ciclo_art"] = np.nan

    scarti = (
        out["Scarti Materiale"].fillna(0)
        + out["Scarti Lavoraz."].fillna(0)
        + out["Pezzi Ripassati"].fillna(0)
    )
    out["pct_scarti"] = np.where(
        out["Tot pezzi Contegg."] > 0,
        scarti / out["Tot pezzi Contegg."],
        np.nan
    )

    if "OEE" in out.columns and "Descrizione Macchina" in out.columns:
        if "Data_Ora_Fine" in out.columns:
            out_sorted = out.sort_values(["Descrizione Macchina", "Data_Ora_Fine"])
        else:
            out_sorted = out.copy()

        grp = out_sorted.groupby("Descrizione Macchina")["OEE"]
        out_sorted["rolling_oee_3"] = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        for lag in [1, 2, 3]:
            out_sorted[f"lag_{lag}_oee"] = grp.transform(lambda x, l=lag: x.shift(l))

        out["rolling_oee_3"] = out_sorted["rolling_oee_3"]
        for lag in [1, 2, 3]:
            out[f"lag_{lag}_oee"] = out_sorted[f"lag_{lag}_oee"]
    else:
        out["rolling_oee_3"] = np.nan
        for lag in [1, 2, 3]:
            out[f"lag_{lag}_oee"] = np.nan

    return out

def get_feature_cols_oee() -> list:
    return [
        "FASE",
        "C.d.L. Prev",
        "ARTICOLO_grouped",
        "Qta totale su AS/400",
        "Tempo Attrezz. ORE",
        "Tempo Macc AS400 ORE",
        "Tempo Attr AS400 ORE",
        "Buon Tempo Ciclo ORE",
        "ratio_attr_macc",
        "pct_setup",
        "pct_fermi",
        "pct_scarti",
        "media_ciclo_art",
        "std_ciclo_art",
        "giorno_settimana",
        "mese",
        "settimana_anno",
        "rolling_oee_3",
        "lag_1_oee",
        "lag_2_oee",
        "lag_3_oee",
    ]