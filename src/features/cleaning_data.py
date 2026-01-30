# Libraries

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import sys
import numpy as np
from statsmodels.stats.multitest import multipletests

# Config path

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
SRC_DIR = PROJECT_ROOT / 'src'

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from features.utilities import classify_point
from constant.constant_variables import HIGH_CORR_VARS

def prepare_data(data_path: str):

    try:
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)

        elif data_path.endswith('.xlsx'):
            data = pd.read_excel(data_path)
        else:
            data = pd.read_pickle(data_path)

    except Exception as e:
        data_path = data_path.split('.')
        len_data_path = len(data_path)
        print(f"This code was not develop for this type of content {data_path[len_data_path]}\n Try with - '.csv', '.xlsx' or '.pkl' \n {e}")

    data = data.drop(columns= HIGH_CORR_VARS)

    data.to_csv(SRC_DIR / 'data' / 'processed/df_casos.csv', index=False)

    print("--- The data is ready!!! ---")
    

class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Transformador compatible con Scikit-Learn para eliminar columnas
    dentro de un Pipeline.
    """
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        cols = [c for c in self.columns_to_drop if c in X.columns]
        return X.drop(columns=cols)

def load_and_prep_data(file, del_col) :
    """
    This code is prepararing and uploading the data
    necesary for the datasets.

    ARGS:
        file: The origin the object is getting the data, the PATH.
        del_col: These are the columns necesary to eleminate from
            the Datasets.
    """
    if not file.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file}")
        
    try:
        if file.endswith('.csv'):
            data = pd.read_csv(data_path)

        elif file.endswith('.xlsx'):
            data = pd.read_excel(data_path)
        else:
            data = pd.read_pickle(data_path)

    except Exception as e:
        data_path = data_path.split('.')
        len_data_path = len(data_path)
        print(f"This code was not develop for this type of content {data_path[len_data_path]}\n Try with - '.csv', '.xlsx' or '.pkl' \n {e}")

    
    y = data['Autismo']
    X = data.drop(columns=del_col)
    
    return X, y

def standarize_ids(id_series):
    """Convierte IDs como 'P-01' a 'P01' para asegurar el merge."""
    return id_series.astype(str).str.replace("-", "0", regex=False).tolist()

def prepare_volcano_data(results, alpha=0.05):
    """
    Procesa los resultados estadísticos, aplica corrección FDR 
    y categoriza los puntos para el Volcano Plot.
    """
    df = pd.DataFrame(results)
    
    _, pvals_corrected, _, _ = multipletests(df['P_Value'], method='fdr_bh')
    df['P_Adj'] = pvals_corrected
    
    df['MinusLog10P'] = -np.log10(df['P_Value'])

    df['Categoria'] = df.apply(classify_point, axis=1)
    
    return df