import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# Given a 2D array of shape (n_classes, n_vars), make pretty boxplots
def pretty_boxplot(data, 
                   labels: list[str] = None,
                   figsize: tuple[float, float] = (9, 6),
                   box_width: float = 0.8,
                   color_map: str = 'deep',
                   y_min: int = 47,
                   tick_rotation: tuple[int, int] = (0, 0),
                   f_mult: float = None,
                   title: str = 'Expression Detection Accuracy',
                   xlabel: str = 'Subject Number',
                   ylabel: str = 'Accuracy (%)'
                ) -> None: 
    if labels is None: 
        labels = [f'{i}' for i in range(1, np.shape(data)[1]+1)]
    
    # Create dataframe to ease seaborn plotting          
    df = pd.DataFrame(data, columns=labels)
    
    # Font correction factor 
    if f_mult is None:
        f_mult = np.sqrt(figsize[0]*figsize[1]/54)
    
    # Make figure 
    plt.figure(figsize=figsize, dpi=300)    
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.linewidth'] = 0.75
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    ax = plt.gca()
    sns.boxplot(data=df, 
                width=box_width, 
                palette=color_map,
                linewidth=0.5, 
                showfliers=False, 
                ax=ax)
    colors = sns.color_palette(color_map, 20)

    # Add rectangles from 0 to the bottom of each boxplot
    for i, col in enumerate(df.columns):
        # Get First Quartile (y-coordinate of the bottom of the boxplot)
        q1 = df[col].quantile(0.75)
        
        # Get the color for this boxplot
        box_color = colors[i % len(colors)]
        
        # Create a rectangle from 0 to Q1
        rect = Rectangle(
            (i - box_width/2,  # x position (left edge of boxplot)
            0),                # y position (starting at 0)
            box_width,         # width (same as boxplot width)
            q1,                # height (from 0 to Q1)
            facecolor=box_color,
            edgecolor='black',
            linewidth=0.25,
            alpha=0.5   # slightly more solid than the boxplot
        )
        ax.add_patch(rect)

    # Add median value annotations
    medians = df.median().values
    pos = np.arange(len(medians))
    
    for tick, median in zip(pos, medians):
        qb = df[df.columns[tick]].median()
        ax.text(tick, qb - 5, 
                f'{median:.1f}%', 
                horizontalalignment='center',
                color='black', 
                fontweight='bold', size=round(12*f_mult),
                bbox=dict(facecolor='white', 
                          alpha=0.8, 
                          edgecolor='none', 
                          boxstyle='round,pad=0.1'))

    # Set proper limits with some padding
    _, top = ax.get_ylim()
    ax.set_ylim(y_min, min(103, top + 5))

    # Set axis labels
    plt.ylabel(ylabel, 
               fontsize=round(18*f_mult))
    if xlabel is not None:
        plt.xlabel(xlabel, 
                   fontsize=round(18*f_mult))

    # Ensure y-axis ticks are multiples of 10 
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))

    # Aesthetics
    plt.title(title, 
              fontsize=round(20*f_mult),
              pad=20)
    
    # Add a subtle grid on the y-axis only
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Rotate x-axis labels if they are long
    plt.xticks(rotation=tick_rotation[0], 
               ha='center', 
               fontsize=round(16*f_mult), 
               fontweight='regular')
    plt.yticks(rotation=tick_rotation[1], 
               fontsize=round(16*f_mult), 
               fontweight='regular')
    
    plt.tight_layout()
    plt.show()
    
def pretty_scatterplot(x: list[float],
                       y: list[float],
                       labels: list[str], 
                       figsize: tuple[float, float] = (9, 6),
                       x_title: str = 'Time', 
                       x_label: str = 'Time (ms)',
                       y_title: str = 'Accuracy',
                       y_label: str = 'Accuracy (%)', 
                       color_map: str = 'deep',
                       
                    ) -> None:
    df = pd.DataFrame({ x_title: x, 
                        y_title: y,
                        'Label': labels
                    })

    # Make figure
    plt.figure(figsize=figsize, dpi=300)    
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.linewidth'] = 0.75
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    
    ax = plt.gca() 
    sns.scatterplot(data=df,
                    x=x_title,
                    y=y_title,
                    s=120,  # Point size
                    hue='Label',
                    alpha=1,
                    palette=color_map,
                    ax=ax)
    
    # Aesthetics
    plt.xlabel(x_label, 
               fontsize=18, 
               fontweight='bold')
    plt.ylabel(y_label, 
               fontsize=18, 
               fontweight='bold')
    plt.title('Time vs Accuracy Scatter Plot', 
              fontsize=18, 
              fontweight='bold')
    plt.legend(fontsize=14,  
               loc='lower right', 
               frameon=True, 
               edgecolor='black')
    
    # Add a subtle grid on the y-axis only
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to make room for labels
    plt.tight_layout()
    plt.show()