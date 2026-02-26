# ****** APPUNTI ******
# Il feature engineering è il processo di trasformazione dei dati grezzi
# in variabili (feature) che rendono gli algoritmi di machine learning più efficaci. 
# I suoi obiettivi principali sono due: 
# Rendere i dati compatibili: gli algoritmi non accettano dati testuali o date
# dunque con il feature engineering trasformiamo i dati in formati leggibili dal modello.
# Migliorare le prestazioni: aiuta il modello a cogliere sfumature e relazioni nascoste
# nei dati che altrimenti passerebbero inosservate. 



import pandas as pd

# transforming time in numbers 
# in order to see if there are patterns in a certain timeframe
def add_time_features(df):
    df["Data_Ora_Fine"] = pd.to_datetime(df["Data_Ora_Fine"])
    df["giorno_settimana"] = df["Data_Ora_Fine"].dt.weekday
    df["mese"] = df["Data_Ora_Fine"].dt.month
    df["settimana_anno"] = df["Data_Ora_Fine"].dt.isocalendar().week.astype(int)
    df["anno"] = df["Data_Ora_Fine"].dt.year
    
    return df

# rolling mean and std 
# in order to see if in the last 10 processes there's been a trend
def add_rolling_features(df, window=10):
    df = df.sort_values("Data_Ora_Fine")
    
    df[f"rolling_mean_{window}"] = (
        df["Indice_Inefficienza"]
        .rolling(window=window, min_periods=1)
        .mean()
    )  
    df[f"rolling_std_{window}"] = (
        df["Indice_Inefficienza"]
        .rolling(window=window, min_periods=1)
        .std()
    )
    
    return df

# machine deviation
# in order to see how much the the machine is distant from its usual behavior 
def add_deviation_feature(df):
    media_macchina = df["Indice_Inefficienza"].mean()
    df["deviazione_dalla_media"] = (
        df["Indice_Inefficienza"] - media_macchina
    )
    
    return df

# how important was the inefficiency in the previous processing
def add_lag_features(df, lag=1):
    df = df.sort_values("Data_Ora_Fine")
    df[f"lag_{lag}"] = df["Indice_Inefficienza"].shift(lag)
    return df

def feature_engineering_pipeline(df):
    df = add_time_features(df)
    df = add_rolling_features(df, window=10)
    df = add_deviation_feature(df)
    df = add_lag_features(df)
    
    return df


