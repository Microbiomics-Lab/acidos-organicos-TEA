# Libraries
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / 'src'

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Custom Imports
from features.save_img import save_figure
from features.utilities import comparison_one_by_time
from features.cleaning_data import prepare_volcano_data
# -------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -------------------------------------------------------------------------

def plot_volcano(df: pd.DataFrame, title='', fc_threshold=1, alpha=0.05, save_name=None):
    """
    Generates and saves a Volcano Plot from a processed DataFrame.
    """

    MARKERS = {
        'Significativo y Relevante': 'X', 
        'Significativo (Poco cambio)': 'o',
        'No significativo y Relevante': 'o',
        'NS (No significativo)': 'o'
    }
    
    PALETTE = {
        'Significativo y Relevante': "#000000", 
        'Significativo (Poco cambio)': "#000000",
        'No significativo y Relevante': "#000000",
        'NS (No significativo)': "#DBDFE2"
    }

    del_col = ['Código', 'Autismo', 'Sexo']

    dropped = [
    'Carboxicítrico', 'Fumárico', '3-metilglutárico',
    '3-hidroxibutírico', 'Adípico', '2-hidroxibutírico'
    ]

    del_col += dropped

    X, y = df.drop(columns=del_col), df['Autismo']
    t0, t1 = 0, 1

    df_results = comparison_one_by_time(X, y, t0, t1)

    df_results = prepare_volcano_data(df_results)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.scatterplot(
        data=df_results, 
        x='Log2FC', 
        y='MinusLog10P', 
        hue='Categoria', 
        style='Categoria',
        markers=MARKERS, 
        palette=PALETTE, 
        alpha=0.7, 
        ax=ax, 
        s=60
    ) 

    ax.axhline(-np.log10(alpha), color='black', linestyle='--', linewidth=0.8, label=f'p={alpha}')
    ax.axvline(fc_threshold, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(-fc_threshold, color='black', linestyle='-', linewidth=0.8, label=rf'$FC \in (-{fc_threshold}, {fc_threshold})$')

    top_hits = df_results[df_results['Categoria'].str.contains('Relevante', na=False)]
    
    for _, row in top_hits.iterrows():
        ax.text(
            x=row['Log2FC'], 
            y=row['MinusLog10P'] + 0.1, 
            s=row['Metabolito'], 
            fontsize=9, 
            ha='center'
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('log2(Fold Change)', fontsize=12)
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    ax.set_xlim((-10, 32)) 
    ax.grid(False)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    save_figure(fig, save_name, subfolder='reports/figures', fmt='png')
    
    plt.close(fig)
    return fig


def plot_distribution_by_group(df: pd.DataFrame, x_col: str, group_col='Autismo', bins=8, title=None, save_name=None):
    """
    Generates a comparative histogram between groups for a numerical variable.
    
    Args:
        df (pd.DataFrame): Data source.
        x_col (str): Name of the numerical column to plot (e.g., 'Edad').
        group_col (str): Name of the grouping column (e.g., 'Autismo').
        bins (int): Number of histogram bins.
        title (str): Chart title.
        save_name (str): Filename for saving (without extension).
    """
    df_viz = df.copy()

    # Mapping for visualization labels (0->CON, 1->TEA)
    if df_viz[group_col].dtype in ['int64', 'int32', 'float64']:
        df_viz['Viz_Group'] = df_viz[group_col].map({0: 'CON', 1: 'TEA'})
    else:
        df_viz['Viz_Group'] = df_viz[group_col]

    fig, ax = plt.subplots(figsize=(10, 6))

    own_colors = {
    'TEA': "#000000",
    'CON': "#B9BDC0"
    }
    
    sns.histplot(
        data=df_viz,
        x=x_col,
        hue='Viz_Group',
        bins=bins,
        palette=own_colors,
        multiple='dodge',      
        kde=True,              
        shrink=0.8,            
        ax=ax
    )

    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    
    if title:
        ax.set_title("", fontsize=14)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if ax.get_legend():
        ax.get_legend().set_title("Grupo")

    if save_name is None:
        save_name = f'Composition_{x_col}_by_{group_col}'
        
    save_figure(fig, save_name, subfolder='reports/figures', fmt='png')
    plt.close(fig)


def plot_biomarker_clustermap(df, feature_list, save_name, sample_col='Código'):
    """
    Generic function for Hierarchical Clustering Heatmaps (Clustermap).
    Replaces separate functions for VIP and LASSO.
    """
    
    valid_features = [f for f in feature_list if f in df.columns]
    
    df_clust = df[valid_features].copy()

    if sample_col in df.columns:
        yticklabels=df[sample_col]
        df_clust.index.name = 'Muestra' 

    # Generate Clustermap
    g = sns.clustermap(
        df_clust,
        method='ward',      
        metric='euclidean', 
        z_score=1,          
        cmap='vlag',       
        vmin=-1.5, 
        yticklabels=yticklabels,
        vmax=1.5,
        center=0,
        figsize=(10, 8),
        dendrogram_ratio=(.15, .2),
        cbar_pos=(0.02, 0.8, 0.03, 0.15)
    )
    

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    
    save_figure(g.fig, save_name, 'reports/figures', fmt='png')
    plt.close(g.fig)

# -------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -------------------------------------------------------------------------
def run_composition_pipeline():
    """
    Función maestra que orquesta la generación de todos los gráficos de composición.
    Puede ser llamada desde main.py o ejecutada directamente.
    """
    print("--- Starting Composition Plots Generation ---")

    DATA_DIR = PROJECT_ROOT / "src" / "data"  
    FILE_CASES = DATA_DIR / "results-OAT_20casos_V3.xlsx"
    
    if not FILE_CASES.exists():
        print(f"ERROR: Data file not found at {FILE_CASES}")
        return

    print(f"Loading data from: {FILE_CASES}")
    df_cases = pd.read_excel(FILE_CASES)

    df_cases.columns = df_cases.columns.str.replace(r'\s+', '', regex=True)
    
    VIP_VARS = [
        '2-hidroxisocapróico', 'N-acetil-aspártico', 'Fenilpirúvico', 
        '3-metilglutacónico', 'Piroglutámico', 'Uracilo', 'Metilsucciínico', 
        'Feniláctico', 'Biotina', 'Acetoacético'
    ]
    
    LASSO_VARS = [
        '2-hidroxisocapróico', 'N-acetil-aspártico', '2-hidroxifenilacético',
        'Biotina', 'Fenilpirúvico', '5-hidroximetil-2-furóico', 'Glicérico',
        'Acetoacético', '2-hidroxihipúrico'
    ]

    print("Generating Volcano Plot...")
    plot_volcano(df_cases, save_name='Composition_Volcano1')

    print("Generating Age Histogram...")
    plot_distribution_by_group(df_cases, x_col='Edad', title="Age Distribution by Group", save_name='Composition_Hist_TOP1')

    print("Generating VIP Clustermap...")
    plot_biomarker_clustermap(df_cases, VIP_VARS, save_name='Composition_ClustVIP_TOP1')

    print("Generating LASSO Clustermap...")
    plot_biomarker_clustermap(df_cases, LASSO_VARS, save_name='Composition_ClustLASSO_TOP1')

    print("--- Composition Process Finished ---")

# --- BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    run_composition_pipeline()