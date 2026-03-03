import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.metrics import make_scorer, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")
from feature_engineering import pipeline_classificazione

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "koepfer_160_2.csv")
OUTPUT_CM_PATH = os.path.join(PROJECT_ROOT, "outputs", "confusion_matrix_rf_anomaly.png")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "classification", "best_classificazione_anomaly.pkl")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "classification", "parametri_classificazione_anomaly.pkl")

# load datas
df = pd.read_csv(DATA_PATH)

if "ARTICOLO" in df.columns:
    df["ARTICOLO"] = df["ARTICOLO"].fillna("MISSING_ARTICOLO").astype(str)

# ARTICOLO_grouped + OHE 
counts = df['ARTICOLO'].value_counts()
TOP_N_ARTICOLI = 20
articoli_top = counts.head(TOP_N_ARTICOLI).index
df['ARTICOLO_grouped'] = df['ARTICOLO'].where(
    df['ARTICOLO'].isin(articoli_top),
    other='ALTRO'
)
df['ARTICOLO_grouped'] = df['ARTICOLO_grouped'].fillna('ALTRO').astype(str)

df = pipeline_classificazione(df)

# class definition (target)
p60 = df['Indice_Inefficienza'].quantile(0.60)
p85 = df['Indice_Inefficienza'].quantile(0.85)

print(f"Soglia ATTENZIONE (60° percentile): {p60:.4f}")
print(f"Soglia ANOMALIA   (85° percentile): {p85:.4f}")

SOGLIA_ATTENZIONE = p60
SOGLIA_ANOMALIA   = p85

def classifica_inefficienza(valore):
    if valore <= SOGLIA_ATTENZIONE:
        return 0   
    elif valore <= SOGLIA_ANOMALIA:
        return 1   
    else:
        return 2   

df["classe"] = df["Indice_Inefficienza"].apply(classifica_inefficienza)

print("DISTRIBUZIONE CLASSI:")
classe_counts = df["classe"].value_counts().sort_index()
labels = {0: "NORMALE", 1: "ATTENZIONE", 2: "ANOMALIA"}
for cls, cnt in classe_counts.items():
    pct = cnt / len(df) * 100
    print(f"  Classe {cls} ({labels[cls]}): {cnt} campioni ({pct:.1f}%)")

# warning if classes are unbalanced
min_pct = (classe_counts / len(df) * 100).min()
if min_pct < 10:
    print(f"\nClasse minoritaria sotto il 10% ({min_pct:.1f}%).")
    print("Considera class_weight='balanced' nei modelli.")

# X and y def
cols_to_drop = [
    "classe",
    "Indice_Inefficienza",
    "Tempo Lavoraz. ORE",
    "Tempo_Teorico_TOT_ORE",
    "WO",
    "ARTICOLO",
    "Descrizione Articolo",
    "ID DAD",
    "Descrizione Macchina",
    "C.d.L. Effett",
    "Data_Ora_Fine",
]

y = df["classe"]
X = df.drop(columns=cols_to_drop, errors="ignore")

# train/test split 
if "Data_Ora_Fine" in df.columns:
    idx_sorted = df.sort_values("Data_Ora_Fine").index
    split_idx = int(len(idx_sorted) * 0.8)
    train_idx = idx_sorted[:split_idx]
    test_idx = idx_sorted[split_idx:]
    X_train, X_test = X.loc[train_idx].copy(), X.loc[test_idx].copy()
    y_train, y_test = y.loc[train_idx].copy(), y.loc[test_idx].copy()
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
# columns identification
categorical_cols = [
    col for col in [
        "ARTICOLO_grouped",
        "FASE",
        "Cod CIC",
        "C.d.L. Prev",
        "Descrizione Centro di Lavoro previsto"
    ]
    if col in X.columns
]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()


# convert null in -> 'MISSING'
for col in categorical_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('string').fillna('MISSING')
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('string').fillna('MISSING')

# preprocessing
# for linear models (scaling + one hot)
preprocessor_linear = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

# for tree models (just one hot, no scaling)
preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

# base models
base_models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor_linear),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),
    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=8,
            random_state=42
        ))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42
        ))
    ]),
    "XGBoost": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", XGBClassifier(
            random_state=42,
            verbosity=0,
            eval_metric="mlogloss",
            max_depth=4,
            min_child_weight=5,
            reg_alpha=0.5,
            reg_lambda=3.0,
            gamma=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False
        ))
    ]),
    "SVM": Pipeline([
        ("preprocessor", preprocessor_linear),
        ("model", SVC(
            class_weight="balanced",
            kernel="rbf",
            probability=True, 
            random_state=42
        ))
    ])
}

def valuta_modello(name, y_true, y_pred, y_proba=None):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred,
                                 target_names=["NORMALE", "ATTENZIONE", "ANOMALIA"],
                                 zero_division=0))
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            print(f"  ROC-AUC (weighted OvR): {auc:.4f}")
        except Exception:
            pass

def valuta_generalizzazione(name, y_train_true, y_train_pred, y_test_true, y_test_pred, soglia_gap=0.05):
    train_acc = accuracy_score(y_train_true, y_train_pred)
    test_acc = accuracy_score(y_test_true, y_test_pred)
    gap = train_acc - test_acc

    if gap > soglia_gap:
        diagnosi = "OVERFITTING (training accuracy molto più alta della validation/test)"
    else:
        diagnosi = "GENERALIZZAZIONE BUONA (training ≈ validation/test)"

    print(f"\nGeneralizzazione - {name}")
    print(f"  Training Accuracy:   {train_acc:.4f}")
    print(f"  Validation Accuracy: {test_acc:.4f}")
    print(f"  Gap (train-test):    {gap:.4f}")
    print(f"  Diagnosi:            {diagnosi}")

# training and comparison of base models
results = []
trained_models = {}
predizioni_test = {}
print("\nBASE MODELS:")
for name, model in base_models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test) if hasattr(model.named_steps["model"], "predict_proba") else None

    valuta_modello(name, y_test, y_pred, y_proba)
    valuta_generalizzazione(name, y_train, y_pred_train, y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    results.append({"Model": name, "Accuracy": acc, "F1-macro": f1})
    trained_models[name] = model
    predizioni_test[name] = y_pred

# random forest grid search
print("\n\nRANDOM FOREST - GRID SEARCH")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
])

rf_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [4, 6, 8],
    "model__min_samples_split": [10, 20],
    "model__min_samples_leaf": [5, 10],
    "model__max_features": ["sqrt", 0.7]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring = make_scorer(recall_score, labels=[2], average="macro", zero_division=0), 
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)
print("Migliori parametri RF:", rf_grid.best_params_)

best_rf = rf_grid.best_estimator_        
soglia_anomalia = 0.30
soglia_attenzione = 0.35
proba_train = best_rf.predict_proba(X_train)
y_pred_train_custom = np.where(
    proba_train[:, 2] >= soglia_anomalia, 2,
    np.where(proba_train[:, 1] >= soglia_attenzione, 1, 0)
)
proba = best_rf.predict_proba(X_test) 
y_pred_custom = np.where(
    proba[:, 2] >= soglia_anomalia, 2,
    np.where(proba[:, 1] >= soglia_attenzione, 1, 0)
)
valuta_modello("Random Forest Ottimizzata (soglie custom)",
               y_test,
               y_pred_custom,
               proba)
valuta_generalizzazione("Random Forest Ottimizzata (soglie custom)",
                       y_train,
                       y_pred_train_custom,
                       y_test,
                       y_pred_custom)
results.append({
    "Model":    "Random Forest Ottimizzata",
    "Accuracy": accuracy_score(y_test, y_pred_custom),
    "F1-macro": f1_score(y_test, y_pred_custom, average="macro", zero_division=0)
})
trained_models["Random Forest Ottimizzata"] = best_rf
predizioni_test["Random Forest Ottimizzata"] = y_pred_custom

# xgboost grid search
print("\n\nXGBOOST - GRID SEARCH")

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor_tree),
    ("model", XGBClassifier(
        random_state=42,
        verbosity=0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=3.0,
        gamma=1.0,
        tree_method="hist",
        use_label_encoder=False
    ))
])

xgb_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 6, 10],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8],
    "model__min_child_weight": [3, 5],
    "model__reg_alpha": [0.0, 0.5],
    "model__reg_lambda": [1.0, 3.0],
    "model__gamma": [0.0, 1.0]
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring = make_scorer(recall_score, labels=[2], average="macro", zero_division=0),
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)
print("Migliori parametri XGB:", xgb_grid.best_params_)

best_xgb = xgb_grid.best_estimator_
soglia_anomalia = 0.30
soglia_attenzione = 0.35
proba_train = best_xgb.predict_proba(X_train)
y_pred_train_custom = np.where(
    proba_train[:, 2] >= soglia_anomalia, 2,
    np.where(proba_train[:, 1] >= soglia_attenzione, 1, 0)
)
proba = best_xgb.predict_proba(X_test)
y_pred_custom = np.where(
    proba[:, 2] >= soglia_anomalia, 2,
    np.where(proba[:, 1] >= soglia_attenzione, 1, 0)
)
valuta_modello("XGBoost Ottimizzata (soglie custom)",
               y_test,
               y_pred_custom,
               proba)
valuta_generalizzazione("XGBoost Ottimizzata (soglie custom)",
                       y_train,
                       y_pred_train_custom,
                       y_test,
                       y_pred_custom)

results.append({
    "Model":    "XGBoost Ottimizzata",
    "Accuracy": accuracy_score(y_test, y_pred_custom),
    "F1-macro": f1_score(y_test, y_pred_custom, average="macro", zero_division=0)
})
trained_models["XGBoost Ottimizzata"] = best_xgb
predizioni_test["XGBoost Ottimizzata"] = y_pred_custom

# final comparison 
results_df = pd.DataFrame(results).sort_values("F1-macro", ascending=False)
print("\n\nCONFRONTO FINALE MODELLI:")
print(results_df.to_string(index=False))
best_model_name = max(results, key=lambda x: x["F1-macro"])["Model"]
best_model = trained_models[best_model_name]
best_f1 = max(r["F1-macro"] for r in results)
best_y_pred = predizioni_test[best_model_name]
recall_anomalia = recall_score(y_test, best_y_pred, labels=[2], average="macro", zero_division=0)
print(f"  Recall ANOMALIA: {recall_anomalia:.4f}")
print(f"\nModello migliore: {best_model_name}  (F1-macro: {best_f1:.4f})")

# confusion matrix
print(f"\nMatrice di confusione - {best_model_name}:")
cm = confusion_matrix(y_test, best_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMALE", "ATTENZIONE", "ANOMALIA"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_CM_PATH), exist_ok=True)
plt.savefig(OUTPUT_CM_PATH, dpi=150)
plt.show()
print(f"Matrice salvata in {OUTPUT_CM_PATH}")

#save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(best_model, MODEL_PATH)

parametri_anomaly = {
    "articoli_top": articoli_top.tolist(),
    "top_n_articoli": TOP_N_ARTICOLI,
    "soglia_attenzione": SOGLIA_ATTENZIONE,
    "soglia_anomalia": SOGLIA_ANOMALIA,
    "tipo_soglie": "percentili_60_85",
    "soglia_proba_anomalia": 0.30,
    "soglia_proba_attenzione": 0.35,
}
joblib.dump(parametri_anomaly, PARAMS_PATH)

print(f"Modello salvato in {MODEL_PATH}")
print(f"Parametri salvati in {PARAMS_PATH}")



# ****** APPUNTI ******

# prima di addestrare il modello devo droppare delle colonne per far si che l'addestramento
# sia sensato e privo di overfitting ovvero chiaramente togliendo prima di tutte la colonna 
# target ovvero i dati che vogliamo predire e poi tante altre colonne che creerebbero una 
# quantità di dati inutili dal momento che nel preprocessing utilizziamo onehotencoding
# che trasforma ogni variabile categorica in un formato numerico che il modello può comprendere
# così facendo però viene creata una colonna per ogni categoria distinta. ad esempio:
# La colonna ARTICOLO  può avere moltissimi valori diversi (alta cardinalità) con 
# articoli rari che compaiono solo 1-2 volte
# dunque con :

# TOP_N_ARTICOLI = 20
# articoli_top = counts.head(TOP_N_ARTICOLI).index
# 
# df['ARTICOLO_grouped'] = df['ARTICOLO'].where(
#     df['ARTICOLO'].isin(articoli_top),
#     other='ALTRO'
# )

# in questo modo mantengo solo i 20 articoli più frequenti, Tutti gli altri vengono trasformati in: ALTRO
# dal momemento poi che articolo_grouped è all'interno di categorical_cols 
# Nel preprocessing:
# ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
# Non sto creando una colonna per ogni articolo esistente ma 20 + 1 = 21 colonne
# in questo modo a differenza del frequency encoding non perdo l'identità dell'articolo
# come funziona frequency encoding invece: 

# senza frequency encoding: 

#    cliente  articolo  articolo_Mouse  articolo_Monitor  articolo_Tastiera
# 0    Mario     Mouse               1                 0                   0
# 1    Luigi  Tastiera               0                 0                   1
# 2 Giovanni     Mouse               1                 0                   0
# 3    Mario   Monitor               0                 1                   0
# 4     Anna  Tastiera               0                 0                   1

#con frequency encoding si avrà invece: 

#    cliente  articolo  articolo_freq
# 0    Mario     Mouse              2
# 1    Luigi  Tastiera              2
# 2 Giovanni     Mouse              2
# 3    Mario   Monitor              1
# 4     Anna  Tastiera              2

# il problema è che in questo modo si perde l'identità dell'articolo ad esempio nel nostro caso se
# una macchina è inefficiente ogni volta che deve produrre un mouse, il modello ora considera sia il mouse
# che la tastiera come lo stesso oggetto avendo la stessa frequenza in questo modo non è in grado di distinguere
# il fatto che magari la produzione del mouse è inefficiente mentre quella della tastiera si

# il campione di dati su cui vengono addestrati i modelli è molto limitato, poco meno di 400 righe per i 
# modelli è veramente poco per avere delle performance reali buone senza overfitting.
# i modelli che ho addestrato specialmente xgboost ottimizzato sono molto rigidi
# cercando di ridurre quanto possibile l'overfitting anche se come vediamo dall'output c'è sempre data 
# la poca quantità di dati.
# per dati con quantità di righe maggiori, si potrebbe pensare di addestrare modelli più complessi e 
# meno regolarizzati, ma con 387 righe si rischia di avere delle performance falsate.


# COMMENTO DATI OTTENUTI

# metriche calcolate:
# ACCURACY = calcola la percentuale di risposte corrette

# F1-SCORE = combina le metriche precisione e richiamo in un unico valore 
# Precisione (Precision): Indica quanti dei casi positivi predetti sono effettivamente positivi (precisione delle previsioni positive).
# Richiamo (Recall/Sensitivity): Indica quanti dei casi positivi reali sono stati effettivamente identificati dal modello

# ROC-AUC = misura quanto un modello è in grado di distinguere tra le classi a diversi livelli di soglia.
# Curva ROC (Receiver Operating Characteristic): È un grafico che mostra la performance del modello tracciando due parametri
# al variare della soglia di classificazione:
# True Positive Rate (TPR) / Sensibilità / Recall: Indica la percentuale di casi positivi correttamente identificati.
# False Positive Rate (FPR): Indica la percentuale di casi negativi erroneamente classificati come positivi.
# AUC (Area Under the Curve): Rappresenta l'area geometrica sottesa alla curva ROC.

# F1-score (per classe) = Media armonica di precision e recall di quella singola classe
# F1-macro = Media semplice degli F1-score di tutte le classi (NORMALE + ATTENZIONE + ANOMALIA / 3), 
# senza pesare per quante osservazioni ha ogni classe
# F1-weighted = Media degli F1-score pesata per il numero di campioni (support) di ogni classe

# inizialmente bisogna decidere le soglie ovvero trasformare l'Indice_Inefficienza in classi discrete.
# in base alle quali classifichiamo come normale, attenzione e anomalia  i risultati che otteniamo. 
# soglie utilizzate nella classificazione standard basate sulla deviazione standard:
#   NORMALE     ≤ mean
#   ATTENZIONE  < mean < mean + 1std
#   ANOMALIA    > mean + 1std
# in base ai nostri dati:
#   NORMALE     ≤ 1.185 -> lavorazione nella norma
#   ATTENZIONE  1.185 - 1.47 -> lieve inefficienza, da monitorare
#   ANOMALIA    > 1.47 -> inefficienza significativa
# la media e l'std sono però più influenzati dalla skweness e non
# hanno risultato dare valori particolarmente performanti

# risultati allenando il modello in maniera standard:

# Distribuzione classi:
# NORMALE → 65.6%
# ATTENZIONE → 19.6%
# ANOMALIA → 14.7%
 
# Il migliore è:
 
# XGBoost
# Accuracy: 0.756
# F1-macro: 0.616
# ROC-AUC: 0.8921
# È coerente che batta Random Forest: con dataset piccoli e non lineari, XGBoost tende a essere più stabile.
 
# Logistic buono con (0.68 accuracy)
# Random Forest base collassa su ATTENZIONE
# XGBoost mantiene equilibrio sulle 3 classi
# Classe ATTENZIONE
# In tutti i modelli:
# Precision bassa (~0.42 XGB)
# Recall medio-bassa (~0.33)
# questo accade in quanto ATTENZIONE è una classe di confine statistico.
# È la zona grigia tra normale e anomalia.
# Il modello fa fatica perché i dati sono simili a NORMALE e le feature
# probabilmente non separano abbastanza bene.
# 
# come si può notare dalla matrice confusione i dati che vengono predetti meglio sono i dati
# relativi alla classe normale (47), mentre i peggiori sono della classe attenzione (5) e anomalie (7)

# risultati allenando il modello in maniera anomaly oriented:

# modificando il codice cambiando le soglie usando i quantili (60° percentile) (85° percentile) al posto 
# della media e dell'std che sono più influenzati dalla skweness in quanto i dati presentano una lunga coda 
# sulla destra e la maggiorparte dei dati sulla sinistra con il valore di 1.00 ovvero che la macchina performa 
# come dovrebbe e solo una piccola percentuale di volte risultano esserci degli outliers quando la macchina
# produce con un tempo maggiore o molto maggiore rispetto al tempo previsto, essendo influenzati da ciò
# non hanno risultato dare valori particolarmente performanti.

# soglie utilizzate nella classificazione standard basate sulla deviazione standard:
#   NORMALE     ≤ 60° percentile
#   ATTENZIONE  < 60° percentile < 85° percentile
#   ANOMALIA    > 85° percentile
# in base ai nostri dati:
#   NORMALE     ≤ 1.1466 -> lavorazione nella norma
#   ATTENZIONE  1.1466 - 1.4311 -> lieve inefficienza, da monitorare
#   ANOMALIA    > 1.4311 -> inefficienza significativa

# addestrando inoltre i modelli ottimizzati in modo tale che non 
# sia semplicemente una valutazione standard ma aumentando il recall delle anomalie ovvero:
# al posto di utilizzare y_pred = best_xgb.predict(X_test)
# che non permette di scegliere la soglia ma usa di default 0.5 
# rendendo il modello più bravo a non lasciarsi sfuggire i casi positivi reali, 
# accettando però il rischio di accusare qualche innocente per errore.

# nella classificazione anomaly oriented si è inoltre cambiato lo 
# scoring che è quel fattore che valuta i modelli basandoti solo su quanto sono bravi a identificare correttamente 
# gli esempi della classe ANOMALIE, ignorando le altre classi durante l'ottimizzazione
  
# risultati allenando il modello aumentando le recall delle anomalie:
# Distribuzione classi:
# NORMALE → (59.9%)
# ATTENZIONE → (25.1%)
# ANOMALIA → (15.0%)
# Il migliore è chiaramente:
# XGBoost Ottimizzata
# Accuracy: 0.794
# F1-macro: 0.759
# ROC-AUC: 0.9148
 
# come è possibile notare rispetto ai dati precedenti ogni metrica del modello che ha performato
# meglio (sempre XGBoost Ottimizzata) sono tutti migliorati
 
# Logistic è migliorato anch'esso da 0.68 a 0.74 di accuracy
  
# c'è stato un grosso miglioramento nei risultati di tutti i modelli, guardando la matrice di confusione
# però notiamo che i valori della classe normale sono diminuiti e sono ora (39), diminuiscono perché la
# soglia è scesa (60° percentile (1.1466) rispetto alla media (1.1855)), il miglioramento vero però è che
# ora ATTENZIONE e ANOMALIA hanno più campioni, i valori della classe attenzione ora sono (14), un netto
# miglioramento rispetto ai (5) precedente seppur ancora un dato abbastanza basso, (9) invece nella classe
# anomalia, piccolo miglioramente rispetto al (7) precedente. 
# un altro dato molto interessante è il ROC-AUC di XGBoost Ottimizzato che sale da 0.9037 a 0.9148. Questo 
# è particolarmente significativo perché ROC-AUC misura la capacità discriminativa del modello indipendentemente
# dalla soglia di classificazione, quindi il modello è genuinamente migliorato nella separazione delle classi, 
# non solo per effetto delle soglie percentili.
 
# a confermare ulteriormente i dati che abbiamo ottenuto nella matrice riguardo la classe ATTENZIONE nel modello 
# nuovo in XGBoost ottimizzato abbiamo notevoli miglioramenti di precision (precision 0.42 -> precision 0.58) e di 
# recall (recall 0.33 -> recall 0.74). Rispetto al modello standard, il miglioramento è netto.
 
# c'è da fare una discriminante però avendo solo 387 righe totali e un test set di 78 campioni, i risultati vanno 
# letti con cautela statistica. 1 predizione in più o in meno sulla classe ANOMALIA (12 campioni nel test) sposta 
# il recall dell'8%. I risultati sono promettenti ma andrebbero confermati su più dati.