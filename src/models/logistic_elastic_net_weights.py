# Libraries

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
import logging
import os
import pickle


BASE_DIR = Path(__file__).resolve().parents[2] 
SRC_DIR = BASE_DIR / 'src'

from features.cleaning_data import ColumnDropper, load_and_prep_data
from constant.constant_variables import HIGH_CORR_VARS, META_COLS

DATA_DIR = SRC_DIR / "data"

FILE_CUEST = DATA_DIR / "Cuest_OAT_V3.xlsx"
FILE_CASOS = DATA_DIR / "processed/df_casos.csv"

logging.info("Uploading Data...")

def dev_logistic_elastic_net_weights(file=FILE_CASOS):
    
    data = pd.read_csv(file)
    X, y = data.drop(columns=META_COLS), data['Autismo']

    pipe_log = Pipeline(
        [
            ("Stand", StandardScaler()),
            ("model", PLSRegression(n_components=2))
        ]
    )

    pipe_log.fit(X, y)

    pls_model = pipe_log.named_steps["model"]

    x_scores = pls_model.x_scores_ 

    var_por_componente = np.var(x_scores, axis=0)
    X_scaled = pipe_log.named_steps["Stand"].transform(X)
    total_var = np.sum(np.var(X_scaled, axis=0))

    explained_variance_ratio = var_por_componente / total_var

    scores_df = pd.DataFrame(
        x_scores, 
        columns=['Componente 1', 'Componente 2']
    )

    scores_df['Diagnostico'] = y
    scores_df['Code'] = data['Código']

    total_var = np.var(X_scaled, axis=0).sum()

    data_to_save = {
        'scores_df' : scores_df,
        'total_var' : total_var,
        'var_comp_1' : explained_variance_ratio[0],
        'var_comp_2' : explained_variance_ratio[1]
    }

    filename = DATA_DIR / 'processed/pls_weights.pkl'

    with open(filename, 'wb') as file:
        pickle.dump(data_to_save, file)

    print(f"Data was upload in '{filename}'")

def dev_logistic_elastic_net(path:str):

    X, y = load_and_prep_data(path, HIGH_CORR_VARS)
    
    y_true_all = []
    y_pred_probs = []

    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X, y):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) 
        
        pls_selector = PLSRegression(n_components=2)
        pls_selector.fit(X_train_scaled, y_train)

        coefs = np.abs(pls_selector.coef_).flatten()

        top_indices = np.argsort(coefs)[-10:]
        
        X_train_best = X_train_scaled[:, top_indices]
        X_test_best = X_test_scaled[:, top_indices]

        pls_final = PLSRegression(n_components=2)
        pls_final.fit(X_train_best, y_train)
        
        pred = pls_final.predict(X_test_best)
        
        y_true_all.append(y_test)
        y_pred_probs.append(pred)

    y_true_all = np.array(y_true_all)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_bin = (y_pred_probs > 0.5).astype(int)

    final_acc = accuracy_score(y_true_all, y_pred_bin)
    final_q2 = r2_score(y_true_all, y_pred_probs)

    print("-" * 40)
    print(f"Accuracy: {final_acc:.2f}")
    print(f"Q2:       {final_q2:.2f}")
    print("-" * 40)

if __name__ == "__main__":

    logging.info("Starting with the model...")

    dev_logistic_elastic_net_weights(FILE_CASOS)
    
    logging.info("The model 'Logistic Regression was develop!!!'")