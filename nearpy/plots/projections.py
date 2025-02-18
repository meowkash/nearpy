import pandas as pd
import seaborn as sns 
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

def visualize_tsne(data, labels, perplexity=15, dim=2):
    # By default, my T-SNE visualizations are 2-dimensional
    ts = TSNE(n_components=dim, learning_rate='auto', perplexity=perplexity)
    z = ts.fit_transform(data)
    df = pd.DataFrame()
    df["y"] = labels.astype(int)
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.set_style("whitegrid")
    g = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), palette='muted', data=df)
    g.set(title=f't-SNE with perplexity = {perplexity}') 
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

def perform_pca(data_dict):
    """Performs PCA on the dataset and returns explained variance ratio."""
    data_combined = np.vstack(list(data_dict.values())).reshape(-1, 16)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_combined)
    pca = PCA()
    pca.fit(data_scaled)
    return pca.explained_variance_ratio_

def perform_ica(data_dict, n_components=5):
    """Performs ICA to extract independent components."""
    data_combined = np.vstack(list(data_dict.values())).reshape(-1, 16)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_combined)
    ica = FastICA(n_components=n_components, random_state=42)
    return ica.fit_transform(data_scaled)

def perform_mrmr(df, k=5):
    """Performs MRMR feature selection based on mutual information."""
    data_combined = np.vstack([np.array(mag).reshape(299, 16) for mag in df['mag']]).reshape(-1, 16)
    y = df.index.repeat(299)  # Assuming labels are indices for classification purposes
    mi = mutual_info_classif(data_combined, y)
    selected_indices = np.argsort(mi)[-k:]  # Select top-k features
    return selected_indices, mi[selected_indices]