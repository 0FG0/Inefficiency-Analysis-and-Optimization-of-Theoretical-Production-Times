import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# loading the clean datas of KOEPFER 160/2 machine
df = pd.read_csv("../data/processed/koepfer_160_2.csv")


# find couples with high correlation
print("ANALISI CORRELAZIONI TRA FEATURES:")
corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()

high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.7:
            high_corr.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print("Features fortemente correlate (>0.7):")
for feat1, feat2, corr in sorted(high_corr, key=lambda x: x[2], reverse=True):
    print(f"{feat1} <-> {feat2}: {corr:.3f}")


# Prevision inefficiency
y = df["Indice_Inefficienza"]
X = df.drop(columns=["Indice_Inefficienza"])


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# columns identification
categorical_cols = [
    "FASE",
    "Cod CIC",
    "C.d.L. Prev",
    "Descrizione Centro di Lavoro previsto"
]
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()


# preprocessing
# for linear models (scaling + one hot)
preprocessor_linear = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_cols)
        
    ]
)

# for tree models (just one hot, no scaling)
preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)


# base models
base_models = {
    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor_linear),
        ("model", LinearRegression())
    ]),
    
    "Ridge": Pipeline([
        ("preprocessor", preprocessor_linear),
        ("model", Ridge())
    ]),
    
    "Lasso": Pipeline([
        ("preprocessor", preprocessor_linear),
        ("model", Lasso())
    ]),
    
    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", DecisionTreeRegressor(random_state=42))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", RandomForestRegressor(random_state=42)) 
    ]),
         
    "XGBoost": Pipeline([
        ("preprocessor", preprocessor_tree), 
        ("model", XGBRegressor(random_state=42)) 
    ])
}


# training and comparison of base models
results = []
print(" \nBASE MODELS:")
for name, model in base_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    results.append((name, r2, mse))
    
    print(f"\n{name}")
    print(f"R²: {r2:.3f}")
    print(f"MSE: {mse:.3f}")


# random forest grid search
print(" \nRANDOM FOREST GRID SEARCH")
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", RandomForestRegressor(random_state=42))
])

rf_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 3]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

print("Miglior parametri RF:", rf_grid.best_params_)

best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print("\nRandom Forest Ottimizzata")
print(f"R²: {r2_rf:.3f}")
print(f"MSE: {mse_rf:.3f}")

results.append(("Random Forest Ottimizzata", r2_rf, mse_rf))


# xgboost grid search
print("\nXGBOOST GRID SEARCH")

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        eval_metric="rmse"
    ))
])

xgb_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 6, 10],
    "model__learning_rate": [0.01, 0.1],
    "model__subsample": [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train, y_train)

print("Best XGB params:", xgb_grid.best_params_)

best_xgb = xgb_grid.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

r2_xgb = r2_score(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

print("\nXGBoost Ottimizzata")
print(f"R²: {r2_xgb:.3f}")
print(f"MSE: {mse_xgb:.3f}")

results.append(("XGBoost Ottimizzata", r2_xgb, mse_xgb))


# view
results_df = pd.DataFrame(results, columns=["Model", "R2", "MSE"])
results_df = results_df.sort_values(by="R2", ascending=False)

print("\nCONFRONTO FINALE MODELLI:")
print(results_df)

# ****** APPUNTI ******

# R² -> quanto il modello si avvicina al valore reale,
# R² = 1 valore perfetto, 0 media, < 0 peggio della media 
# MSE -> errore medio al quadrato, di quanti $ si discosta dal valore reale

# nel preprocessing negli alberi non ho messo il drop="if_binary" in quanto 
# Per gli alberi è meglio non droppare mai la prima colonna. 
# Gli alberi traggono vantaggio dalla ridondanza perché possono scegliere di fare
# uno "split" su qualsiasi categoria in modo più diretto.
# per gli alberi non si usa lo StandardScaler() in quanto gli alberi di decisione 
# sono invarianti alla scala. Non gli importa se un numero è 0.001 o 1.000.000,
# perché lavorano per "soglie" (es. Età > 30?). 
# Usando passthrough si risparmiano calcoli inutili.

# quando si runna il codice compare 
# UserWarning: Found unknown categories in columns [0] during transform. These unknown categories will be encoded as all zeros
# questo è dovuto dal fatto che 
# OneHotEncoder(drop="first", handle_unknown="ignore")
# con handle_unknown="ignore" le categorie sconosciute vengono trasformate tutte in zeri
# questo avviene perchè nella fase di training ovviamente non ci sono tutte le possibili variazioni
# dei dati dunque nel test appare una variazione che il modello non ha mai visto

# I risultati mostrano una netta differenza tra modelli lineari e modelli basati su alberi decisionali,
# questi ultimi infatti superano significativamente i modelli lineari.
# suggerendo la presenza di relazioni non lineari tra le variabili operative e l'indice di inefficienza.

# Modelli Lineari:
# Linear Regression → R² ≈ 0.33
# Ridge → R² ≈ 0.32
# Lasso → R² ≈ 0

# Questi modelli spiegano solo circa il 30% della variabilità dell’indice di inefficienza.
# Questo indica che la relazione tra variabili operative e inefficienza non è lineare, 
# sono presenti interazioni complesse tra le feature che i modelli lineari non sono in grado di cogliere. 
# Inoltre come ovviamente ci si aspettava dai dati, esiste multicollinearità 
# (come evidenziato dall’analisi delle correlazioni). in particolare Lasso collassa praticamente a zero, 
# segno che la penalizzazione elimina quasi tutta l’informazione utile 
# in presenza di forte correlazione tra variabili.
 
# Modelli ad Albero:
# Decision Tree → R² ≈ 0.70
# Random Forest → R² ≈ 0.840
# Random Forest Ottimizzata → R² ≈ 0.843
# XGBoost → R² ≈ 0.868
# XGBoost Ottimizzata → R² ≈ 0.858  

# Questi modelli spiegano circa l’85–87% della variabilità. 
# Questo suggerisce che il sistema produttivo presenta dinamiche non lineari e che 
# l’inefficienza dipende da combinazioni di variabili e non da effetti indipendenti.

# XGBoost è il modello che ha ottenuto il miglior risultato con:
# R² ≈ 0.868
# Dopo l'ottimizzazione si note che random forest è migliorata leggermente mentre XGBoost è peggiorato 
# passando da 0.868 a 0.858, questo significa che il dataset è già ben modellato con parametri standard, 
# non c’è forte overfitting nel modello base e che la struttura dei dati è relativamente stabile.
# il fatto che xgboost spieghi circa l'87% dell'inefficienza significa che l’Indice di Inefficienza 
# non è casuale ma è fortemente legato alle variabili operative e che quindi è possibile costruire un 
# sistema predittivo affidabile.