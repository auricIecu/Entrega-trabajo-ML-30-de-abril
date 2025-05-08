import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# 1. Cargar el DataFrame ya procesado
import os
merged_path = 'merged_data.csv'
if not os.path.exists(merged_path):
    raise FileNotFoundError(f'No se encontró el archivo {merged_path}. Asegúrate de generarlo antes de ejecutar este script.')
df = pd.read_csv(merged_path, parse_dates=['date'])

# 2. Chequear que las columnas necesarias existen
deseadas = ['close_btc', 'hash_rate', 'difficulty_btc']
faltantes = [col for col in deseadas if col not in df.columns]
if faltantes:
    raise ValueError(f'Faltan las siguientes columnas en merged_data.csv: {faltantes}')
df = df.dropna(subset=deseadas)


# 3. Renombrado y target multiclase como en Modelo.ipynb
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

# Renombrar columna si es necesario
if 'close_btc' in df.columns and 'close' not in df.columns:
    df = df.rename(columns={'close_btc': 'close'})

# Cargar lista de features
import pickle
with open('feature_list.pkl', 'rb') as f:
    features = pickle.load(f)

# Target multiclase
# (Ajusta esto si tu lógica de señales es diferente)
if 'signal' in df.columns:
    df['target'] = df['signal']
else:
    raise ValueError('No se encontró la columna de señales (signal) en merged_data.csv')

# División temporal simple: solo train y test
# Ajusta la fecha de corte según tu preferencia
fecha_corte = '2024-06-17'
df = df.sort_values('date')
train_mask = df['date'] < fecha_corte
test_mask = df['date'] >= fecha_corte

X_train = df.loc[train_mask, features]
y_train = df.loc[train_mask, 'target']
X_test = df.loc[test_mask, features]
y_test = df.loc[test_mask, 'target']

# 4. Definir pipelines y grids igual que Modelo.ipynb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipelines = {
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ]),
    'AdaBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classifier', AdaBoostClassifier(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
    ]),
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'))
    ]),
    'SVC': Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classifier', SVC(probability=True, random_state=42))
    ])
}

# Grids de hiperparámetros (ajusta según tu notebook)
grids = {
    'Random Forest': {
        'feature_selection__k': [5, 10, 15],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__class_weight': ['balanced', 'balanced_subsample', None]
    },
    'Gradient Boosting': {
        'feature_selection__k': [5, 10, 15],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 10]
    },
    'AdaBoost': {
        'feature_selection__k': [5, 10, 15],
        'classifier__n_estimators': [50, 100]
    },
    'XGBoost': {
        'feature_selection__k': [5, 10, 15],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 10]
    },
    'Logistic Regression': {
        'feature_selection__k': [5, 10, 15],
        'classifier__C': [0.01, 0.1, 1.0],
        'classifier__class_weight': ['balanced', None]
    },
    'SVC': {
        'feature_selection__k': [5, 10, 15],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__class_weight': ['balanced', None]
    }
}

from sklearn.metrics import f1_score, classification_report, confusion_matrix
results = {}
print('Entrenando y evaluando modelos con GridSearchCV...')
for name, pipe in pipelines.items():
    print(f"\n{name}\n{'='*50}")
    grid = GridSearchCV(pipe, grids[name], cv=5, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train, y_train)
    preds = grid.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    results[name] = {'model': grid.best_estimator_, 'f1': f1, 'params': grid.best_params_}
    print(f"Mejor F1-score en test: {f1:.4f}")
    print(f"Mejores parámetros: {grid.best_params_}")
    print(classification_report(y_test, preds))
    print('Matriz de confusión:')
    print(confusion_matrix(y_test, preds))

# 5. Seleccionar y guardar el mejor modelo
best_name = max(results, key=lambda k: results[k]['f1'])
best_model = results[best_name]['model']
best_f1 = results[best_name]['f1']
best_params = results[best_name]['params']

meta = {
    'model_name': best_name,
    'model': best_model,
    'features': features,
    'f1_score': best_f1,
    'params': best_params
}
with open('best_model_for_streamlit.pkl', 'wb') as f:
    pickle.dump(meta, f)
print(f"\nMejor modelo: {best_name} (F1-score={best_f1:.4f}) guardado como best_model_for_streamlit.pkl")

def predict_next_day(input_dict):
    """input_dict: {'close': valor, 'hash_rate': valor, 'difficulty': valor}"""
    with open('bitcoin_rf_model.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        features = data['features']
    X_new = pd.DataFrame([input_dict])[features]
    pred = model.predict(X_new)[0]
    return 'Sube' if pred == 1 else 'Baja'

# Ejemplo de uso:
# resultado = predict_next_day({'close': 70000, 'hash_rate': 400, 'difficulty': 80000000})
# print('Predicción para mañana:', resultado)

# Este script es el punto de partida ideal para conectar con Streamlit y para presentaciones de negocio.
