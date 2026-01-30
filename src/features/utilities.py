# utilities.py

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

def drop_high_corr_columns(df, threshold=0.85):
    """
    La utilidad del siguiente código es poder eliminar variables que estén altamente correlacionadas,
    esto ocurre porque si ambas tienen una alta correlación estarían explicando la misma información,
    en otras palabras es información redundante.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    stacked_corr = upper.stack()
    high_corr_pairs = stacked_corr[stacked_corr > threshold]

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop, pd.DataFrame(high_corr_pairs.items(), columns= ['Variables Alt. Corr.', 'Coef. Corr.'])

def classify_point(row, alpha = .05, fc_threshold = 1):
    
    es_significativo = row['P_Adj'] < alpha    
    # Cambio que ha sido detectado como relevante
    
    es_cambio_fuerte = abs(row['Log2FC']) > fc_threshold
    
    if es_significativo and es_cambio_fuerte:
        return 'Significativo y Relevante'
    elif es_significativo:
        return 'Significativo (Poco cambio)'
    elif es_cambio_fuerte:
        return 'No significativo y Relevante'
    else:
        return 'NS (No significativo)'
    
def calculate_vip(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vips

def comparison_one_by_time(X, y, t0, t1):
    results = []

    for col in X.columns:
        group_sano = X.loc[y == t0, col]
        group_enfermo = X.loc[y == t1, col]

        # Cálculo del Fold Change (FC)
        # FC = Media_Enfermo / Media_Sano
        # Sumamos 1e-9 para evitar división 
        # por cero si algún valor es 0
        
        mean_sano = group_sano.mean() + 1e-9
        mean_enfermo = group_enfermo.mean() + 1e-9
        fc = mean_enfermo / mean_sano
        log2_fc = np.log2(fc)
        
        # Prueba de Hipótesis con la U de Mann-Whitney,
        #  aquí no vamos a observar la normalidad
        # alternative='two-sided' porque no sabemos
        #  si sube o baja. Aquí vamos analizar si
        #  las dos poblaciones provienen de la misma
        #  o son diferentes.
        
        stat, p_val = mannwhitneyu(group_sano,
                                    group_enfermo,
                                    alternative='two-sided')
        
        results.append({
            'Metabolito': col,
            'Log2FC': log2_fc,
            'P_Value': p_val
        })
    
    return results

def ifelse(test, true, false):
    return true if test else false