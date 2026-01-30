# Libraries
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

# -------------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / 'src'

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Custom Imports
from features.save_img import save_figure
from features.utilities import ifelse

# -------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -------------------------------------------------------------------------

def plot_pls_score(df, variance_dict, save_name='Evaluation_PLS_DA_Plot'):
    """
    Generates the PLS-DA Score Plot.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'Componente 1', 'Componente 2', and 'Code'.
        variance_dict (dict): Dictionary with 'total', 'pc1', 'pc2' variance values.
        save_name (str): Output filename.
    """

    df = df

    df['Code'] = [ifelse(df['Code'][i].startswith('TEA'), 'TEA', 'CON') for i in range(len(df))]

    total = variance_dict['total']
    pc1_pct = variance_dict['pc1'] / total * 100
    pc2_pct = variance_dict['pc2'] / total * 100

    markers = {'TEA': 'X', 'CON': 'o'}

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.scatterplot(
        data=df, 
        x='Componente 1', 
        y='Componente 2', 
        markers=markers,
        style='Code',     
        color='black',    
        s=80,
        ax=ax              
    )

    legend_elements = [
        Line2D([0], [0], marker='X', color='w', label='TEA',
               markerfacecolor="#000000", markersize=10),
        Line2D([0], [0], marker='o', color='w', label='CON',
               markerfacecolor="#000000", markersize=10)
    ]

    ax.legend(handles=legend_elements, loc="lower left", title="Group")
    ax.set_xlabel(f'Component 1 ({pc1_pct:.2f}% var)', fontsize=12)
    ax.set_ylabel(f'Component 2 ({pc2_pct:.2f}% var)', fontsize=12)
    ax.set_title(' ', fontsize=14)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.grid(False)

    save_figure(fig, save_name, subfolder='reports/figures', fmt='png')

def plot_vip_lollipop(df, threshold=1.0, save_name='Evaluation_Feature_Importance_Lollipop_VIP'):
    """
    Generates a Lollipop Chart for Variable Importance in Projection (VIP).
    """

    my_range = range(len(df))
    
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.hlines(y=my_range, xmin=0, xmax=df['VIP'], color='black', linewidth=1.5)
    ax.plot(df['VIP'], my_range, "o", color='black', markersize=8)

    ax.axvline(threshold, color='grey', linestyle='--', linewidth=0.8, label=rf'$VIP > 1$')

    for i in my_range:
        value = df['VIP'].iloc[i]
        ax.text(
            x=value + 0.05,        
            y=i,                  
            s=f"{value:.2f}",     
            va='center',        
            fontsize=10,
            color='black'
        )

    ax.set_yticks(my_range)
    ax.set_yticklabels(df['Metabolito'], fontsize=11)
    ax.set_xlabel('VIP Score', fontsize=12)
    ax.set_ylabel('Metabolite', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, df['VIP'].max() * 1.15) 
    ax.grid(False)
    ax.legend(loc='lower right', frameon=True)

    save_figure(fig, save_name, subfolder='reports/figures', fmt='png')


def plot_lasso_coefficients(df, save_name='Evaluation_LASSO_Barplot'):
    """
    Generates a Barplot for LASSO Coefficients.
    Includes auto-sorting for better visualization.
    """
    # Sort by coefficient magnitude for a cleaner plot
    df_sorted = df.sort_values('Abs_Coef', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=df_sorted, 
        x='Coeficiente', 
        y='Variable', 
        color='black',  
        ax=ax           
    )

    ax.axvline(0, color='grey', linestyle='-', linewidth=1)
    ax.set_title(' ', fontsize=14)
    ax.set_xlabel('Coefficient Magnitude', fontsize=12)
    ax.set_ylabel('Biomarker')

    # Clean style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure(fig, save_name, subfolder='reports/figures', fmt='png')
    plt.close(fig)


def run_evaluation_pipeline():
    """
    Orchestrates the loading of data and generation of evaluation plots.
    """
    print("--- Starting Evaluation Plots Generation ---")


    PROCESSED_DIR = SRC_DIR / "data" / "processed" 

    print("Loading processed data...")
    try:

        data = pd.read_pickle(PROCESSED_DIR / 'pls_weights.pkl')

        total_var = data['total_var']
        scores_df = data['scores_df']
        pc1 = data['var_comp_1']
        pc2 = data['var_comp_2']


        top_biomarkers = pd.read_csv(PROCESSED_DIR / 'top_biomarkers2.csv')
        survivors = pd.read_csv(PROCESSED_DIR / 'LASSO_SURVIVORS.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    pls_variance = {
        'total': total_var,
        'pc1': pc1,
        'pc2': pc2
    }
    
    # Fig 1: PLS-DA
    print("Generating PLS-DA Score Plot...")
    plot_pls_score(scores_df, pls_variance, save_name='Evaluation_PLS_DA_Plot1')

    # Fig 2: VIP Lollipop
    print("Generating VIP Lollipop Chart...")
    plot_vip_lollipop(top_biomarkers, save_name='Evaluation_Feature_Importance_Lollipop_VIP1')

    # Fig 3: LASSO Coefficients
    print("Generating LASSO Barplot...")
    plot_lasso_coefficients(survivors, save_name='Evaluation_LASSO_Barplot1')

    print("--- Evaluation Process Finished Successfully! ---")

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation_pipeline()

