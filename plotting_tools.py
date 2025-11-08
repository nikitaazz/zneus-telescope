import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.lines import Line2D


def visualize_distributions(df, exclude_cols=[], 
                            title='Prehľad distribúcií všetkých atribútov', 
                            figsize=(20, 22), n_cols=4):
    """
    Vytvorí histogramy s KDE pre všetky numerické atribúty v DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame s dátami na vizualizáciu
    exclude_cols : list
        Zoznam stĺpcov, ktoré sa majú vynechať z vizualizácie
    title : str
        Hlavný nadpis grafu
    figsize : tuple
        Veľkosť celého grafu (šírka, výška)
    n_cols : int
        Počet stĺpcov v gridu (defaultne 4)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure objekt s histogramami
    """
    # Získanie stĺpcov na vizualizáciu
    cols_to_visualize = [col for col in df.columns 
                         if col not in exclude_cols]
    
    print(f"Počet vizualizovaných atribútov: {len(cols_to_visualize)}")
    print(f"Atribúty: {cols_to_visualize}")
    
    # Výpočet počtu riadkov
    n_rows = (len(cols_to_visualize) + n_cols - 1) // n_cols
    
    # Vytvorenie figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten()
    
    # Iterácia cez všetky atribúty
    for idx, col in enumerate(cols_to_visualize):
        ax = axes_flat[idx]
        
        # Získanie dát a základných štatistík
        data_all = df[col].dropna()
        mean_all = data_all.mean()
        median_all = data_all.median()
        skewness_all = stats.skew(data_all)
        
        # Vykreslenie histogramu s KDE
        sns.histplot(data_all, bins=30, kde=True, color='skyblue', 
                     ax=ax, stat='density', alpha=0.6)
        
        # Pridanie čiar pre priemer a medián
        ax.axvline(mean_all, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Mean: {mean_all:.2f}')
        ax.axvline(median_all, color='green', linestyle='--', linewidth=1.5, 
                   label=f'Median: {median_all:.2f}')
        
        # Nastavenie titulku a štítkov
        ax.set_title(f'{col} - Distribution', fontsize=10, fontweight='bold')
        ax.set_xlabel(None)
        ax.set_ylabel('Density', fontsize=8)
        
        # Pridanie textu so šikmosťou
        textstr = f'Skew: {skewness_all:.2f}'
        ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Legenda a mriežka
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Skrytie prázdnych subplotov
    for idx in range(len(cols_to_visualize), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    # Hlavný titulok
    fig.suptitle(title, fontsize=20, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig

def visualize_outliers_boxplots(df, exclude_cols=['oximetry', 'latitude', 'longitude'], 
                                  title='Boxplots všetkých atribútov', figsize=(20, 5)):
    """
    Vytvorí boxploty pre všetky numerické atribúty v DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame s dátami na vizualizáciu
    exclude_cols : list
        Zoznam stĺpcov, ktoré sa majú vynechať z vizualizácie
    title : str
        Hlavný nadpis grafu
    figsize : tuple
        Základná veľkosť jedného riadku grafov (šírka, výška_riadku)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure objekt s boxplotmi
    """
    # Získanie všetkých numerických stĺpcov okrem vylúčených
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]
    
    print(f"Počet vizualizovaných atribútov: {len(numeric_cols)}")
    print(f"Atribúty: {numeric_cols}")
    
    # Výpočet rozloženia grafov
    num_cols = len(numeric_cols)
    n_rows = (num_cols + 3) // 4  # 4 grafy na riadok
    n_cols = min(4, num_cols)
    
    # Vytvorenie figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)
    
    # Flatten axes pre jednoduchšiu prácu
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if num_cols == 1 else axes
    
    # Vytvorenie boxplotu pre každý atribút
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        data = df[col].dropna()
        
        # Vytvorenie boxplotu
        bp = ax.boxplot([data], vert=True, patch_artist=True, 
                         showmeans=True, meanline=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         meanprops=dict(color='green', linewidth=2, linestyle='--'),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))
        
        ax.set_title(f'{col}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Hodnota', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Výpočet a zobrazenie počtu outlierov
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = ((data < lower_bound) | (data > upper_bound)).sum()
        
        ax.text(0.5, 0.98, f'Outliers: {outliers_count}', 
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Skrytie prázdnych subplotov
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    # Pridanie legendy
    try:
        from matplotlib.lines import Line2D
        mean_line = Line2D([0], [0], color='green', linewidth=2, linestyle='--')
        median_line = Line2D([0], [0], color='red', linewidth=2)
        fig.legend([mean_line, median_line], ['Mean', 'Median'], 
                   loc='upper right', fontsize=10)
    except Exception:
        pass
    
    plt.tight_layout()
    return fig
