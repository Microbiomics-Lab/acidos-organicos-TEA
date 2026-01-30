# Libraries

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import sys

# ======= Config enviorement ======= #
BASE_DIR = Path(__file__).resolve().parents[2] 
SRC_DIR = BASE_DIR / 'src'

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
# =================================== #

from features.cleaning_data import ColumnDropper, load_and_prep_data

DATA_DIR = SRC_DIR / "data"

FILE_CUEST = DATA_DIR / "Cuest_OAT_V3.xlsx"
FILE_CASOS = DATA_DIR / "results-OAT_20casos_V3.xlsx"

META_COLS = ['Código', 'Autismo', 'Sexo']

HIGH_CORR_VARS = [
    'Carboxicítrico', 'Fumárico', '3-metilglutárico',
    '3-hidroxibutírico', 'Adípico', '2-hidroxibutírico'
]

print("Cargando datos...")
X, y = load_and_prep_data(FILE_CASOS, HIGH_CORR_VARS)

# Developing model

pipe_pls = Pipeline(
    [
        ('Scale', StandardScaler()),
        ('Model', PLSRegression(n_components=2))
    ]
)

cv_method = LeaveOneOut() 
y_cv = cross_val_predict(pipe_pls, X, y, cv=cv_method)

y_pred_binary = (y_cv > 0.5).astype(int)

acc = accuracy_score(y, y_pred_binary)
q2 = r2_score(y, y_cv)

print(f"Accuracy: {acc:.2f}")
print(f"Q2 (Predictividad): {q2:.2f}\n") 

cm = confusion_matrix(y, y_pred_binary)

print("\nMatriz de Confusión:")
print(cm, "\n")

if q2 < 0.4:
    print("\nEl modelo ajusta (R2 alto) pero no predice bien (Q2 bajo). Posible sobreajuste.")
elif abs(acc - 0.5) < 0.1:
    print("\El modelo no es mejor que lanzar una moneda.\n")