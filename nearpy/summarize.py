#%% Read a confusion matrix and provide summary statistics on gesture accuracy as well as classifier accuracy per person 
from pathlib import Path 
import numpy as np 
import seaborn as sns
import pickle 
import pandas as pd
import matplotlib.pyplot as plt 

gestures = {
    'B': 'Blink',
    'P': 'Pucker',
    'RE': 'Raise Eyebrow',
    'S': 'Smile',
    'LW': 'Left Wink',
    'RW': 'Right Wink',
    'LS': 'Left Smile',
    'RS': 'Right Smile',
    'F': 'Furrow'
}
base_path = Path('/Users/aakash/Downloads/facial expressions')
files = base_path.glob('*.pkl')

ges_accs_across_clfs = {} # We will have num_clfs worth of these 
ges_accs_across_subs = {}
num_clfs = 0 # Will need to dynamically update as some files may be corrupt

subjects = None 
num_subjects = 0

num_gestures = len(gestures.keys())

def get_ges_accs(confmat): 
    accs = [None] * np.shape(confmat)[0]
    for idx, row in enumerate(confmat): 
        accs[idx] = row[idx]/np.sum(row)
    return accs
    
for f_idx, file in enumerate(files):
    try:
        with open(file, 'rb') as f:
            cmat, acc = pickle.load(f)
        
        # Get per gesture accuracy 
        if subjects is None: 
            subjects = set(cmat.keys())
            num_subjects = len(subjects)
            
        # First pass 
        if len(ges_accs_across_subs.keys()) == 0: 
            for sub in set(cmat.keys()):
                ges_accs_across_subs[sub] = []
                
        num_clfs += 1
        
        # Store in list for boxplot using seaborn
        tmp_clf = np.zeros((num_subjects, num_gestures))
        
        for sub_id, sub_cmat in cmat.items():  
            # Compute tmp here
            accs = get_ges_accs(sub_cmat)
            accs = np.round(accs, 4) * 100
            tmp_clf[sub_id-1] = accs
            # Append to array
            tmp_sub = accs
            ges_accs_across_subs[sub_id].append(tmp_sub)
                         
        ges_accs_across_clfs[file.name] = tmp_clf
        
    except: 
        print(f'{file.name} does not have any data')
        continue

#%% Make per-subject box plots
for sub in subjects: 
    df = pd.DataFrame(ges_accs_across_subs[sub], columns=list(gestures.keys()))
    plt.figure()
    sns.boxplot(data=df, width=0.6)
    plt.ylim([0, 1])
    plt.show()
    
#%% Make per-classifier box plots 
for k, v in ges_accs_across_clfs.items(): 
    df = pd.DataFrame(v, columns=list(gestures.keys()))
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Set Nature-like style
    sns.set_style("ticks")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    
    # sns.set_style("whitegrid")
    sns.boxplot(data=df, width=0.6, palette="deep")
    plt.title(k)
    plt.show()

#%% Show summary stats for classifier
for k, v in ges_accs_across_clfs.items():
    print(f'For classifer {k}')
    mean_acc = np.mean(v, axis=0)
    std_acc = np.std(v, axis=0)
    for idx, ges in enumerate(gestures.values()): 
        print(f'{ges} accuracy = {mean_acc[idx]} with S.D. = {std_acc[idx]}')
        
#%% 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# For each classifier and its corresponding accuracies
for k, v in ges_accs_across_clfs.items():
    # Create DataFrame
    df = pd.DataFrame(v, columns=list(gestures.values()))
    
    # Create figure with appropriate size and DPI for publication quality
    plt.figure(figsize=(8, 7), dpi=300)
    
    # Set Nature-like style
    sns.set_style("ticks")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    
    # Create boxplot with professional color palette
    ax = sns.boxplot(data=df, width=0.6, 
                    fliersize=3, linewidth=0.5,
                    showfliers=False)
    
    # Add swarmplot for data points with transparency
    # sns.swarmplot(data=df, color='black', alpha=0.5, size=3)
    
    # Add median value annotations
    medians = df.median().values
    pos = np.arange(len(medians))
    for tick, median in zip(pos, medians):
        ax.text(tick, median - 2, f'{median:.1f}%', 
               horizontalalignment='center',
               color='black', fontweight='bold', size=12,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))
    
    # Set proper limits with some padding
    bottom, top = ax.get_ylim()
    # Ensure the y-axis starts at 0 if possible, and gives appropriate padding at the top
    ax.set_ylim(max(0, bottom - 5), min(100, top + 5))
    
    # Set y-axis label
    plt.ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    
    # Set x-axis label
    plt.xlabel('Gesture', fontsize=18, fontweight='bold')
    
    # Use integer ticks on y-axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    
    # Improve title with classifier name
    plt.title(f'Classification Accuracy by Gesture Type\n{k}', fontsize=14, fontweight='bold')
    
    # Add a subtle grid on the y-axis only
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels if they are long
    plt.xticks(rotation=45, ha='right', fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    
    # Adjust layout and save with high quality
    plt.tight_layout()
    
    # Optional: Save the figure
    # plt.savefig(f'{k}_boxplot.png', dpi=600, bbox_inches='tight')
    
    # Show plot
    plt.show()