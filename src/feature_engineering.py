# ****** APPUNTI ******
# Il feature engineering è il processo di trasformazione dei dati grezzi
# in variabili (feature) che rendono gli algoritmi di machine learning più efficaci. 
# I suoi obiettivi principali sono due: 
# Rendere i dati compatibili: gli algoritmi non accettano dati testuali o date
# dunque con il feature engineering trasformiamo i dati in formati leggibili dal modello.
# Migliorare le prestazioni: aiuta il modello a cogliere sfumature e relazioni nascoste
# nei dati che altrimenti passerebbero inosservate. 
# bisogna stare attenti ad alcune cose:
# non usare mai la colonna target (Indice_Inefficienza) come input diretto.
# le feature "lag" e "rolling" devono usare solo valori passati (shift(1)),
# mai il valore corrente della riga che stiamo cercando di predire.
# Questo file va applicato prima dello split train/test.

import pandas as pd

# transforming time in numbers (es. 0 = monday)
# in order to see if there are patterns in a certain timeframe
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Data_Ora_Fine"] = pd.to_datetime(df["Data_Ora_Fine"])
    df["giorno_settimana"] = df["Data_Ora_Fine"].dt.weekday        
    df["mese"] = df["Data_Ora_Fine"].dt.month
    df["settimana_anno"] = df["Data_Ora_Fine"].dt.isocalendar().week.astype(int)
    df["anno"] = df["Data_Ora_Fine"].dt.year
    return df

# rolling mean and rolling std 
# in order to see if in the last n processes there's been a trend
def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.sort_values("Data_Ora_Fine").copy()
    past = df["Indice_Inefficienza"].shift(1)

    df[f"rolling_mean_{window}"] = (
        past.rolling(window=window, min_periods=1).mean()
    )
    df[f"rolling_std_{window}"] = (
        past.rolling(window=window, min_periods=1).std()
    )
    return df

# represents relationships between quantities and working times.
def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    # ratio of rejected pieces to total, index of process quality
    tot = df["Tot pezzi Contegg."].replace(0, pd.NA)
    df["ratio_scarti"] = (df["Scarti Materiale"] + df["Scarti Lavoraz."]) / tot
    
    # Ratio of setup time to AS400 machine time
    tempo_macc = df["Tempo Macc AS400 ORE"].replace(0, pd.NA)
    df["ratio_attr_macc"] = df["Tempo Attr AS400 ORE"] / tempo_macc
    return df

# how important was the inefficiency in the previous processing
def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
    df = df.sort_values("Data_Ora_Fine").copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["Indice_Inefficienza"].shift(lag)
    return df

# complete pipeline
def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_rolling_features(df, window=10)
    df = add_lag_features(df, lags=[1, 2, 3])
    df = add_ratio_features(df)

    # removes the first lines that does not have enough lag records
    df = df.dropna(subset=["lag_1", "lag_2", "lag_3"]).reset_index(drop=True)
    return df


