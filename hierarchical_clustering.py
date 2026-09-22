import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "placement_Dataset_kmeans_ready.csv")
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

MAX_RECORDS = 3000

def load_data():
    df = pd.read_csv(DATASET_PATH).replace([np.inf, -np.inf], np.nan)
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.loc[:, numeric.nunique(dropna=False) > 1]
    if numeric.empty:
        return None, "No usable numerical columns found."
    if len(numeric) > MAX_RECORDS:
        numeric = numeric.sample(MAX_RECORDS, random_state=42)
    X = StandardScaler().fit_transform(numeric)
    return X, None

def run_hierarchical(n_clusters=3, linkage="ward"):
    try:
        X, error = load_data()
        if error:
            return {"error": error}
        n_clusters = int(n_clusters)
        if n_clusters < 2 or n_clusters >= len(X):
            return {"error": f"Number of clusters must be between 2 and {len(X)-1}."}
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X)
        score = None
        if len(np.unique(labels)) > 1:
            score = round(float(silhouette_score(X, labels)), 4)

        visual = PCA(n_components=2, random_state=42).fit_transform(X) if X.shape[1] >= 2 else np.column_stack([X[:,0], np.zeros(len(X))])
        path = os.path.join(CHART_DIR, "hierarchical_clusters.png")
        plt.figure(figsize=(9,6))
        for c in sorted(np.unique(labels)):
            mask = labels == c
            plt.scatter(visual[mask,0], visual[mask,1], label=f"Cluster {c+1}", alpha=.65)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Hierarchical Clustering")
        plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()

        return {
            "n_clusters": n_clusters, "linkage": linkage,
            "records_used": len(X),
            "cluster_counts": np.bincount(labels, minlength=n_clusters).tolist(),
            "silhouette_score": score,
            "chart": "charts/hierarchical_clusters.png"
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}
