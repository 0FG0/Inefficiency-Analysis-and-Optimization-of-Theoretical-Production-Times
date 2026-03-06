"""
Addestra un modello di regressione per PREDIRE l'OEE futuro
prima che il WO venga eseguito.
Input:  caratteristiche del WO (quantità, articolo, fase, storico macchina, ...)
Target: OEE calcolato a posteriori (Disponibilità x Performance x Qualità)
"""
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from OEE_calculator import calcola_oee
from OEE_feature_engineering import aggiungi_feature_oee, get_feature_cols_oee

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "koepfer_160_2.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "regression", "best_regressione_oee.pkl")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "regression", "parametri_regressione_oee.pkl")

# preprocessing
def build_preprocessors(num_cols: list, cat_cols: list):
    # for linear models: scaling + OHE
    preprocessor_linear = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]), cat_cols),
    ], remainder="drop")
    # for tree models: no scaling, just impute + OHE
    preprocessor_tree = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ], remainder="drop")

    return preprocessor_linear, preprocessor_tree


def valuta_modello_oee(name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    return {"Model": name, "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}

# train
def train(path_input: str = DATA_PATH):
    print(f"\nCaricamento dati: {path_input}")
    df = pd.read_csv(path_input)
    print(f"Righe: {len(df)}")

    df = calcola_oee(df)
    df = df.dropna(subset=["OEE"])
    print(f"Righe con OEE valido: {len(df)}")

    articoli_top = df["ARTICOLO"].value_counts().nlargest(30).index.tolist()
    df["ARTICOLO_grouped"] = df["ARTICOLO"].where(df["ARTICOLO"].isin(articoli_top), other="ALTRO")

    df = aggiungi_feature_oee(df, storico=df)

    feature_cols = get_feature_cols_oee()
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat_cols:
        X[c] = X[c].fillna("MISSING").astype(str)
    num_cols = [c for c in feature_cols if c not in cat_cols]
    y = df["OEE"]

    # analisi correlazioni
    print("\nANALISI CORRELAZIONI TRA FEATURES:")
    corr_matrix = X[num_cols].corr().abs()
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.7:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j],
                ))
    print("Features fortemente correlate (> 0.7):")
    if high_corr:
        for f1, f2, corr in sorted(high_corr, key=lambda x: x[2], reverse=True):
            print(f"  {f1} <-> {f2}: {corr:.3f}")
    else:
        print("  Nessuna coppia con correlazione > 0.7")

    # train/test split temporale
    if "Data_Ora_Fine" in df.columns:
        idx_sorted = df.sort_values("Data_Ora_Fine").index
        split_idx = int(len(idx_sorted) * 0.8)
        X_train = X.loc[idx_sorted[:split_idx]].copy()
        X_test = X.loc[idx_sorted[split_idx:]].copy()
        y_train = y.loc[idx_sorted[:split_idx]].copy()
        y_test = y.loc[idx_sorted[split_idx:]].copy()
        cv_interna = TimeSeriesSplit(n_splits=5)
        cv_descrizione = "TimeSeriesSplit 5-fold"
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        cv_interna = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_descrizione = "KFold 5-fold"

    pre_linear, pre_tree = build_preprocessors(num_cols, cat_cols)

    base_models = {
        "Linear Regression": Pipeline([("pre", pre_linear), ("model", LinearRegression())]),
        "Ridge": Pipeline([("pre", pre_linear), ("model", Ridge())]),
        "Lasso": Pipeline([("pre", pre_linear), ("model", Lasso())]),
        "Decision Tree": Pipeline([("pre", pre_tree), ("model", DecisionTreeRegressor(random_state=42))]),
        "Random Forest": Pipeline([("pre", pre_tree), ("model", RandomForestRegressor(random_state=42))]),
        "XGBoost": Pipeline([("pre", pre_tree), ("model", XGBRegressor(random_state=42, verbosity=0))]),
    }

    results = []
    trained_models = {}

    print("\nBASE MODELS:")
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result = valuta_modello_oee(name, y_test, y_pred)
        results.append(result)
        trained_models[name] = model

    # random forest grid search
    print("\nRANDOM FOREST GRID SEARCH")
    rf_pipeline = Pipeline([
        ("pre", pre_tree),
        ("model", RandomForestRegressor(random_state=42)),
    ])
    rf_param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [2, 5],
    }
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=cv_interna, scoring="r2", n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    print("Migliori parametri RF:", rf_grid.best_params_)
    best_rf = rf_grid.best_estimator_
    result_rf = valuta_modello_oee("Random Forest Ottimizzata", y_test, best_rf.predict(X_test))
    train_rmse_rf = np.sqrt(mean_squared_error(y_train, best_rf.predict(X_train)))
    print(f"  Overfitting check -> Train RMSE: {train_rmse_rf:.4f} | Test RMSE: {result_rf['RMSE']:.4f} | Gap: {result_rf['RMSE'] - train_rmse_rf:.4f}")
    results.append(result_rf)
    trained_models["Random Forest Ottimizzata"] = best_rf

    # xgboost grid search
    print("\nXGBOOST GRID SEARCH")
    xgb_pipeline = Pipeline([
        ("pre",   pre_tree),
        ("model", XGBRegressor(random_state=42, objective="reg:squarederror", verbosity=0)),
    ])
    xgb_param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 6, 10],
        "model__learning_rate": [0.01, 0.1],
        "model__subsample": [0.8, 1.0],
    }
    xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=cv_interna, scoring="r2", n_jobs=-1, verbose=1)
    xgb_grid.fit(X_train, y_train)
    print("Migliori parametri XGB:", xgb_grid.best_params_)
    best_xgb = xgb_grid.best_estimator_
    result_xgb = valuta_modello_oee("XGBoost Ottimizzata", y_test, best_xgb.predict(X_test))
    train_rmse_xgb = np.sqrt(mean_squared_error(y_train, best_xgb.predict(X_train)))
    print(f"  Overfitting check -> Train RMSE: {train_rmse_xgb:.4f} | Test RMSE: {result_xgb['RMSE']:.4f} | Gap: {result_xgb['RMSE'] - train_rmse_xgb:.4f}")
    results.append(result_xgb)
    trained_models["XGBoost Ottimizzata"] = best_xgb

    # confronto finale
    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\nCONFRONTO FINALE MODELLI:")
    print(results_df.to_string(index=False))

    best_model_name = min(results, key=lambda x: (x["RMSE"], -x["R2"]))["Model"]
    best_model = trained_models[best_model_name]
    best_rmse = next(r["RMSE"] for r in results if r["Model"] == best_model_name)
    print(f"\nModello migliore: {best_model_name}  (RMSE: {best_rmse:.4f})")

    # cross-validation esterna (stima robusta)
    if "Data_Ora_Fine" in df.columns and len(X) >= 10:
        cv_esterna = TimeSeriesSplit(n_splits=5)
    else:
        cv_esterna = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_mae_neg = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_mse_neg = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="neg_mean_squared_error",  n_jobs=-1)
    cv_r2 = cross_val_score(best_model, X, y, cv=cv_esterna, scoring="r2",                      n_jobs=-1)
    cv_mae = -cv_mae_neg
    cv_rmse = np.sqrt(-cv_mse_neg)

    print(f"\nSTIMA ROBUSTA CV ESTERNA ({cv_descrizione}) - media ± std:")
    print(f"  MAE:  {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
    print(f"  RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"  R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # metriche OEE specifiche sul modello migliore
    y_pred_final = best_model.predict(X_test)
    n_test = len(y_test)
    n_entro_5pct = ((y_test - y_pred_final).abs() <= 0.05).sum()
    n_entro_10pct = ((y_test - y_pred_final).abs() <= 0.10).sum()
    print(f"\n  Predizioni entro ± 5 punti OEE : {n_entro_5pct}/{n_test} ({n_entro_5pct/n_test:.0%})")
    print(f"  Predizioni entro ± 10 punti OEE: {n_entro_10pct}/{n_test} ({n_entro_10pct/n_test:.0%})")

    # save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    params = {
        "feature_cols": feature_cols,
        "articoli_top": articoli_top,
        "model_name": best_model_name,
        "mae_test": mean_absolute_error(y_test, y_pred_final),
        "r2_test": r2_score(y_test, y_pred_final),
        "soglia_alert": 0.65,
    }
    joblib.dump(params, PARAMS_PATH)

    print(f"\nModello salvato: {MODEL_PATH}")
    return best_model, params

if __name__ == "__main__":
    train()