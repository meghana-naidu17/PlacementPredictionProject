import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "placement_Dataset_kmeans_ready.csv"
)

CHART_DIR = os.path.join(
    BASE_DIR,
    "static",
    "charts"
)

os.makedirs(CHART_DIR, exist_ok=True)


# =========================================================
# LOAD DATASET
# =========================================================

def load_kmeans_data():

    try:

        df = pd.read_csv(DATASET_PATH)

        # Replace infinity values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Keep only numerical columns
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        if numeric_df.empty:
            return None, None, "No numerical columns found in the dataset."

        # Fill missing values using median
        numeric_df = numeric_df.fillna(
            numeric_df.median(numeric_only=True)
        )

        # Remove columns having only one unique value
        constant_columns = [
            column
            for column in numeric_df.columns
            if numeric_df[column].nunique() <= 1
        ]

        if constant_columns:
            numeric_df = numeric_df.drop(
                columns=constant_columns
            )

        if numeric_df.empty:
            return None, None, "No usable numerical columns found."

        # Standardization
        scaler = StandardScaler()

        X = scaler.fit_transform(numeric_df)

        return X, numeric_df, None

    except FileNotFoundError:

        return (
            None,
            None,
            f"Dataset not found: {DATASET_PATH}"
        )

    except Exception as e:

        return (
            None,
            None,
            f"{type(e).__name__}: {str(e)}"
        )


# =========================================================
# VALIDATE K
# =========================================================

def validate_k(k, number_of_records):

    try:

        k = int(k)

    except (ValueError, TypeError):

        return None, "K must be a valid integer."

    if k < 2:

        return None, "K must be at least 2."

    if k >= number_of_records:

        return (
            None,
            f"K must be less than the number of records ({number_of_records})."
        )

    return k, None


# =========================================================
# CREATE CLUSTER CHART
# =========================================================

def create_cluster_chart(X, labels):

    filename = "kmeans_clusters.png"

    filepath = os.path.join(
        CHART_DIR,
        filename
    )

    plt.figure(figsize=(9, 6))

    # -----------------------------------------------------
    # PCA
    # -----------------------------------------------------
    # PCA converts multiple numerical columns into
    # two dimensions so that clusters can be visualized.
    # -----------------------------------------------------

    if X.shape[1] >= 2:

        pca = PCA(
            n_components=2,
            random_state=42
        )

        X_visual = pca.fit_transform(X)

        x_label = "Principal Component 1"
        y_label = "Principal Component 2"

    else:

        X_visual = np.column_stack(
            (
                X[:, 0],
                np.zeros(X.shape[0])
            )
        )

        x_label = "Feature 1"
        y_label = "Feature 2"

    number_of_clusters = len(
        np.unique(labels)
    )

    for cluster in range(number_of_clusters):

        cluster_points = (
            labels == cluster
        )

        plt.scatter(
            X_visual[cluster_points, 0],
            X_visual[cluster_points, 1],
            label=f"Cluster {cluster + 1}",
            alpha=0.65
        )

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(
        "K-Means Cluster Visualization"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=130,
        bbox_inches="tight"
    )

    plt.close()

    return "charts/" + filename


# =========================================================
# FINAL K-MEANS CALCULATION
# =========================================================

def calculate_final_clusters(
    X,
    numeric_df,
    k
):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    # -----------------------------------------------------
    # Cluster counts
    # -----------------------------------------------------

    cluster_counts = np.bincount(
        labels,
        minlength=k
    ).tolist()

    # -----------------------------------------------------
    # Centroids
    # -----------------------------------------------------

    centroids = []

    for cluster_number, centroid in enumerate(
        model.cluster_centers_
    ):

        centroid_values = []

        for value in centroid:

            centroid_values.append(
                round(float(value), 4)
            )

        centroids.append(
            {
                "cluster": cluster_number + 1,
                "values": centroid_values
            }
        )

    # -----------------------------------------------------
    # Silhouette score
    # -----------------------------------------------------

    try:

        silhouette = silhouette_score(
            X,
            labels
        )

        silhouette = round(
            float(silhouette),
            4
        )

    except Exception:

        silhouette = None

    # -----------------------------------------------------
    # Cluster chart
    # -----------------------------------------------------

    cluster_chart = create_cluster_chart(
        X,
        labels
    )

    return {

        "cluster_counts": cluster_counts,

        "centroids": centroids,

        "inertia": round(
            float(model.inertia_),
            4
        ),

        "iterations": int(
            model.n_iter_
        ),

        "silhouette_score": silhouette,

        "cluster_chart": cluster_chart,

        "k": k

    }


# =========================================================
# MANUAL K-MEANS
# =========================================================

def manual_kmeans(
    X,
    numeric_df,
    k
):

    k, error = validate_k(
        k,
        len(X)
    )

    if error:

        return {
            "error": error
        }

    result = calculate_final_clusters(
        X,
        numeric_df,
        k
    )

    result["method"] = "manual"

    result["selected_by"] = (
        "Manual K-Means"
    )

    return result


# =========================================================
# ELBOW METHOD
# =========================================================

def calculate_elbow(X):

    max_k = min(
        10,
        len(X) - 1
    )

    if max_k < 2:

        return {
            "error":
            "Not enough records to calculate the Elbow Method."
        }

    k_values = list(
        range(2, max_k + 1)
    )

    inertia_values = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X)

        inertia_values.append(
            float(model.inertia_)
        )

    # -----------------------------------------------------
    # Create Elbow Graph
    # -----------------------------------------------------

    filename = "kmeans_elbow.png"

    filepath = os.path.join(
        CHART_DIR,
        filename
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        k_values,
        inertia_values,
        marker="o"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Inertia / WCSS"
    )

    plt.title(
        "Elbow Method for Optimal K"
    )

    plt.xticks(
        k_values
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=130,
        bbox_inches="tight"
    )

    plt.close()

    return {

        "method": "elbow",

        "elbow_stage": "graph",

        "show_k_input": True,

        "chart":
            "charts/" + filename,

        "k_values":
            k_values,

        "inertia_values":
            [
                round(value, 4)
                for value in inertia_values
            ]

    }


# =========================================================
# SILHOUETTE METHOD
# =========================================================

def calculate_silhouette(X):

    max_k = min(
        10,
        len(X) - 1
    )

    if max_k < 2:

        return {
            "error":
            "Not enough records to calculate the Silhouette Method."
        }

    k_values = list(
        range(2, max_k + 1)
    )

    silhouette_values = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        score = silhouette_score(
            X,
            labels
        )

        silhouette_values.append(
            float(score)
        )

    # -----------------------------------------------------
    # Find best K
    # -----------------------------------------------------

    best_index = int(
        np.argmax(silhouette_values)
    )

    best_k = k_values[best_index]

    best_score = silhouette_values[
        best_index
    ]

    # -----------------------------------------------------
    # Create Silhouette Graph
    # -----------------------------------------------------

    filename = "kmeans_silhouette.png"

    filepath = os.path.join(
        CHART_DIR,
        filename
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        k_values,
        silhouette_values,
        marker="o"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.title(
        "Silhouette Method"
    )

    plt.xticks(
        k_values
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=130,
        bbox_inches="tight"
    )

    plt.close()

    return {

        "method": "silhouette",

        "chart":
            "charts/" + filename,

        "k_values":
            k_values,

        "silhouette_values":
            [
                round(value, 4)
                for value in silhouette_values
            ],

        "best_k":
            best_k,

        "best_score":
            round(
                best_score,
                4
            )

    }


# =========================================================
# MAIN K-MEANS FUNCTION
# =========================================================

def run_kmeans(
    method,
    k=None
):

    X, numeric_df, error = (
        load_kmeans_data()
    )

    if error:

        return {
            "error": error
        }

    # =====================================================
    # MANUAL
    # =====================================================

    if method == "manual":

        if k is None:

            return {
                "error":
                "Please enter K."
            }

        return manual_kmeans(
            X,
            numeric_df,
            k
        )

    # =====================================================
    # ELBOW
    # =====================================================

    if method == "elbow":

        # -------------------------------------------------
        # First request:
        # Generate elbow graph
        # -------------------------------------------------

        if k is None:

            result = calculate_elbow(
                X
            )

            return result

        # -------------------------------------------------
        # Second request:
        # User entered K after observing graph
        # -------------------------------------------------

        k, error = validate_k(
            k,
            len(X)
        )

        if error:

            return {
                "error": error
            }

        result = calculate_final_clusters(
            X,
            numeric_df,
            k
        )

        result["method"] = "elbow"

        result["elbow_stage"] = (
            "calculated"
        )

        result["selected_by"] = (
            "Elbow Method - User Selected K"
        )

        return result

    # =====================================================
    # SILHOUETTE
    # =====================================================

    if method == "silhouette":

        silhouette_result = (
            calculate_silhouette(X)
        )

        if silhouette_result.get("error"):

            return silhouette_result

        best_k = silhouette_result[
            "best_k"
        ]

        result = calculate_final_clusters(
            X,
            numeric_df,
            best_k
        )

        result["method"] = (
            "silhouette"
        )

        result["selected_by"] = (
            "Silhouette Method"
        )

        result["silhouette_chart"] = (
            silhouette_result["chart"]
        )

        result["best_k"] = best_k

        result["best_score"] = (
            silhouette_result["best_score"]
        )

        result["k_values"] = (
            silhouette_result["k_values"]
        )

        result["silhouette_values"] = (
            silhouette_result[
                "silhouette_values"
            ]
        )

        return result

    # =====================================================
    # INVALID METHOD
    # =====================================================

    return {
        "error":
        "Invalid K-Means method selected."
    }