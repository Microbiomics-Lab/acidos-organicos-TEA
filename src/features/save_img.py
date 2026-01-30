from pathlib import Path
import os
import matplotlib.pyplot as plt

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parents[2]

# os.chdir(DEFAULT_FIGURES_DIR)

def save_figure(fig, fig_id, subfolder="", fmt=['png', "jpg", "pdf", "svg"], dpi=300, close=True):
    """
    Guarda una figura de Matplotlib asegurando que el directorio exista.
    
    Args:
        fig: El objeto figura de matplotlib.
        fig_id: Nombre del archivo (sin extensión).
        subfolder: (Opcional) Subcarpeta dentro de reports/figures (ej. "exploracion").
        fmt: Formato (png, jpg, pdf).
        dpi: Resolución (300 es estándar para impresión/papers).
        close: Si cerrar la figura después de guardar para liberar memoria.
    """
    save_path = DEFAULT_FIGURES_DIR / subfolder
    save_path.mkdir(parents=True, exist_ok=True)
    
    file_path = save_path / f"{fig_id}.{fmt}"
    
    print(f"Guardando gráfico en: {file_path}")
    
    fig.savefig(file_path, format=fmt, dpi=dpi, bbox_inches='tight')
    
    if close:
        plt.close(fig)