import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
import os


BASE_DIR = Path(__file__).resolve().parent 
os.chdir(BASE_DIR)

from features.cleaning_data import ColumnDropper, load_and_prep_data

DATA_DIR = BASE_DIR / "data"

FILE_CUEST = DATA_DIR / "Cuest_OAT_V3.xlsx"
FILE_CASOS = DATA_DIR / "results-OAT_20casos_V3.xlsx"

META_COLS = ['Código', 'Autismo', 'Sexo']

HIGH_CORR_VARS = [
    'Carboxicítrico', 'Fumárico', '3-metilglutárico',
    '3-hidroxibutírico', 'Adípico', '2-hidroxibutírico'
]

def main():
    print("Cargando datos...")
    X, y = load_and_prep_data(FILE_CASOS, HIGH_CORR_VARS)

    ml_pipeline = Pipeline([
        ('dropper', ColumnDropper(columns_to_drop=HIGH_CORR_VARS)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegressionCV(
            cv=LeaveOneOut(),     
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            max_iter=5000,
            n_jobs=-1,
            random_state=42
        ))
    ])

    print("Evaluando modelo (Nested Cross-Validation)...")
    
    cv_scores = cross_val_score(ml_pipeline, X, y, cv=LeaveOneOut(), scoring='accuracy')
    y_probs = cross_val_predict(ml_pipeline, X, y, cv=LeaveOneOut(), method='predict_proba')

    print(f"Accuracy promedio (LOOCV): {cv_scores.mean():.2f}")

    print("Entrenando modelo final para despliegue...")

    ml_pipeline.fit(X, y)

    print("Modelo listo.")

if __name__ == "__main__":
    main()
