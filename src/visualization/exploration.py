# Libraries
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -------------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / 'src'

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from features.save_img import save_figure
from features.cleaning_data import standarize_ids

# -------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -------------------------------------------------------------------------

PALETTE_GROUPS = {
    'TEA': "#000000",
    'CON': "#B9BDC0"
}

PALETTE_LEVELS = {
    'Nivel 1: autismo leve': "#B9BDC0", 
    'Nivel 2: autismo moderado': "#626B72", 
    'Nivel 3: autismo severo': "#000000"
}

# Ordering
ORDER_LEVELS = [
    'Nivel 1: autismo leve', 
    'Nivel 2: autismo moderado', 
    'Nivel 3: autismo severo'
]

# -------------------------------------------------------------------------
# DATA LOADING & PROCESSING
# -------------------------------------------------------------------------

def load_and_process_data():
    """
    Loads raw data, performs cleaning, and merges clinical cases 
    with questionnaire data (Levels of Autism).
    """
    # Paths
    DATA_RAW = PROJECT_ROOT / "src" / "data"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    
    FILE_QUEST = DATA_RAW / "Cuest_OAT_V3.xlsx"
    FILE_CASES = DATA_RAW / "results-OAT_20casos_V3.xlsx"
    FILE_IMPORTANCE = DATA_PROCESSED / 'feature_importance_v1.csv'

    print("Loading data...")
    if not FILE_CASES.exists() or not FILE_QUEST.exists():
        print(f"Error: Data files not found in {DATA_RAW}")
        return None, None, None

    df_quest = pd.read_excel(FILE_QUEST)
    df_cases = pd.read_excel(FILE_CASES)
    
    if FILE_IMPORTANCE.exists():
        df_importance = pd.read_csv(FILE_IMPORTANCE)
    else:
        df_importance = pd.DataFrame() 

    df_cases.columns = df_cases.columns.str.replace(r'\s+', '', regex=True)

    df_cases['Autismo'] = df_cases['Autismo'].map({0: 'CON', 1: 'TEA'})

    # Process Questionnaire Data (for Levels)
    df_cases_subset = df_cases.iloc[:20].copy()
    
    # Process Questionnaire
    df_quest_clean = df_quest.iloc[:21].copy()
    df_quest_clean.drop(index=0, inplace=True)
    df_quest_clean.reset_index(drop=True, inplace=True)
    
    # Standardize IDs for merging
    df_quest_clean['Participante'] = standarize_ids(df_quest_clean['Participante'])

    cols_to_use = ['Participante', "Nivel_aut"]
    
    cols_cases = list(df_cases_subset.columns)
    metadata_cols = ['Código', 'Autismo', 'Sexo']
    cols_biomarkers = [c for c in cols_cases if c not in metadata_cols]
    
    df_levels = df_quest_clean[cols_to_use].merge(
        df_cases_subset, 
        right_on='Código', 
        left_on='Participante'
    )
    
    final_cols = cols_biomarkers + cols_to_use + ['Código', 'Autismo']
    final_cols = [c for c in final_cols if c in df_levels.columns]
    
    df_levels = df_levels[final_cols]

    return df_cases, df_levels, df_importance

# -------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -------------------------------------------------------------------------

def plot_missing_values(df, save_name='Exploration_No_null_fig'):
    """Generates a heatmap of missing values."""
    cols_to_drop = ['Código', 'Autismo', 'Sexo']
    cols_existing = [c for c in cols_to_drop if c in df.columns]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.drop(columns=cols_existing).isna(), cbar=False, ax=ax)
    ax.set_title("Missing Values Heatmap")
    
    save_figure(fig, save_name, 'reports/figures', fmt='png')
    plt.close(fig)

def plot_feature_importance(df_weights, save_name='Exploration_FeatureImportance'):
    """Generates a bar plot for Feature Importance."""
    if df_weights.empty:
        print("Skipping Feature Importance plot (Data not found).")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importancia', y='Variable', data=df_weights, color='black', ax=ax)
    ax.set_title("Feature Importance")
    
    save_figure(fig, save_name, 'reports/figures', fmt='png')
    plt.close(fig)

def plot_metabolite_boxplot(df, x_col, y_col, palette, order=None, title="", save_name="Boxplot"):
    """
    Generic function for Boxplots (Used for both Group and Level comparisons).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.boxplot(
        data=df,
        x=x_col,
        y=y_col,
        palette=palette,
        order=order,
        ax=ax,
        orient='h' 
    )
    
    ax.set_ylabel(y_col)
    ax.set_title(title)
    
    save_figure(fig, save_name, 'reports/figures', fmt='png')
    plt.close(fig)

def plot_joint_distribution(df, x_col, y_col, hue_col, palette, save_name):
    """Generates a Joint Plot (Scatter + Histograms)."""
    g = sns.jointplot(
        data=df, 
        x=x_col,
        y=y_col,
        color="#4CB391",
        hue=hue_col,
        kind="scatter",
        palette=palette,
        height=7
    )

    if g.ax_joint.get_legend():
        g.ax_joint.get_legend().set_title("Group")
        
    save_figure(g.fig, save_name, 'reports/figures', fmt='png')

def plot_biomarker_pairplot(df, var_list, hue_col, palette, save_name):
    """Generates a Pairplot for selected biomarkers."""

    valid_vars = [v for v in var_list if v in df.columns]

    df_viz = df[valid_vars].rename(columns={hue_col: 'Group'})
    
    g = sns.pairplot(
        data=df_viz,
        hue='Group',
        palette=palette,
        corner=True
    )
    
    save_figure(g.fig, save_name, 'reports/figures', fmt='png')

# -------------------------------------------------------------------------
# ORCHESTRATOR FUNCTION - THE WRAPPER
# -------------------------------------------------------------------------

def run_exploration_pipeline():
    """
    Master function to execute the full EDA pipeline.
    Call this function from main.py
    """

    print("--- Starting Exploratory Data Analysis (EDA) ---")
    
    df_cases, df_levels, df_importance = load_and_process_data()
    
    if df_cases is None:
        print("EDA aborted due to missing data.")
        return
    
    # Fig 1: Missing Values
    print("Plotting Null Matrix...")
    plot_missing_values(df_cases, save_name='Exploration_No_null_fig1')
    
    # Fig 2: Feature Importance
    print("Plotting Feature Importance...")
    plot_feature_importance(df_importance, save_name='Exploration_FeatureImportance1')
    
    # Fig 3: Boxplot by Group (TEA vs CON)
    print("Plotting Boxplot (Group)...")
    plot_metabolite_boxplot(
        df=df_cases,
        x_col='2-hidroxisocapróico',
        y_col='Autismo',
        palette=PALETTE_GROUPS,
        save_name='Exploration_BoxPlot_Group1'
    )
    
    # Fig 4: Boxplot by Autism Level
    print("Plotting Boxplot (Levels)...")
    # Rename column temporarily for the plot label
    df_levels_viz = df_levels.rename(columns={'Nivel_aut': 'Nivel de Autismo'})
    
    plot_metabolite_boxplot(
        df=df_levels_viz,
        x_col='2-hidroxisocapróico',
        y_col='Nivel de Autismo',
        palette=PALETTE_LEVELS,
        order=ORDER_LEVELS,
        save_name='Exploration_BoxPlot_Levels1'
    )
    
    # Fig 5: Joint Plot
    print("Plotting Joint Plot...")
    plot_joint_distribution(
        df=df_cases,
        x_col='3-metilglutacónico',
        y_col='N-acetil-aspártico',
        hue_col='Autismo',
        palette=PALETTE_GROUPS,
        save_name='Exploration_JointPlot_Metabolites1'
    )
    
    # Fig 6: Pair Plot
    print("Plotting Pair Plot...")
    VARS_TO_PAIR = [
        'Acetoacético', 'Subérico', 'Fenilpirúvico', 'B2', 'Autismo'
    ]
    plot_biomarker_pairplot(
        df=df_cases,
        var_list=VARS_TO_PAIR,
        hue_col='Autismo',
        palette=PALETTE_GROUPS,
        save_name='Exploration_Pairplot_Biomarkers1'
    )

    print("--- EDA Finished Successfully! ---")

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    run_exploration_pipeline()
