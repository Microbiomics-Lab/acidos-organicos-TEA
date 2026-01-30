# Libraries

import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import os

# Config path

path = Path().cwd()
path = path.parent if path.name == "models" else path
os.chdir(path)

from src.features.utilities import calculate_vip

# Getting data

df_cuest = pd.read_excel('Data/Cuest_OAT_V3.xlsx')
df_casos = pd.read_excel('Data/results-OAT_20casos_V3.xlsx')

del_col = ['Código', 'Autismo', 'Sexo']


# High-correlation
dropped = [
    'Carboxicítrico', 'Fumárico', '3-metilglutárico',
    '3-hidroxibutírico', 'Adípico', '2-hidroxibutírico'
    ]

del_col += dropped

X = df_casos.drop(columns= del_col)
y = df_casos.Autismo

# Developing model

pipe_pls = Pipeline(
    [
        ('Scale', StandardScaler()),
        ('Model', PLSRegression(n_components=2))
    ]
)

pipe_pls.fit(X, y)

vip_scores = calculate_vip(pipe_pls)

vip_df = pd.DataFrame({
    'Metabolito': X.columns,
    'VIP': vip_scores
}).sort_values(by='VIP', ascending=False)

top_biomarkers = vip_df[vip_df['VIP'] > 1]
top_biomarkers['VIP'] = top_biomarkers['VIP'].round(2)
top_biomarkers.reset_index(inplace=True)