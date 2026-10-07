"""
Unsupervised Patient Data Analysis
----------------------------------
Pipeline:
  1. Load data (your CSV, or a synthetic patient dataset if none is given)
  2. Preprocess (impute, scale, encode)
  3. Dimensionality reduction (PCA)
  4. Clustering (KMeans, Hierarchical, DBSCAN) + evaluation
  5. Cluster profiling and visualisation

Usage:
    python patient_clustering.py                    # uses synthetic data
    python patient_clustering.py --csv patients.csv # uses your own data
    python patient_clustering.py --csv patients.csv --k 4

Install:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT_DIR = "output"
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------
def generate_synthetic_data(n=600, seed=RANDOM_STATE):
    """Create a fake patient dataset with 3 hidden groups."""
    rng = np.random.default_rng(seed)
    groups = rng.choice([0, 1, 2], size=n, p=[0.4, 0.35, 0.25])

    # (age, bmi, systolic_bp, glucose, cholesterol, heart_rate)
    centers = {
        0: (35, 23, 115, 90, 170, 70),    # healthy / young
        1: (55, 29, 135, 115, 215, 78),   # metabolic risk
        2: (70, 27, 150, 160, 240, 85),   # high-risk / elderly
    }
    spreads = (7, 2.5, 8, 12, 18, 6)

    rows = [rng.normal(centers[g], spreads) for g in groups]
    df = pd.DataFrame(rows, columns=["age", "bmi", "systolic_bp",
                                     "glucose", "cholesterol", "heart_rate"])
    df["gender"] = rng.choice(["M", "F"], size=n)
    df["smoker"] = np.where(rng.random(n) < (0.1 + 0.15 * groups), "Yes", "No")

    # inject missing values to mimic real data
    for col in ["bmi", "glucose", "cholesterol"]:
        df.loc[rng.random(n) < 0.04, col] = np.nan
    df.insert(0, "patient_id", [f"P{i:04d}" for i in range(n)])
    return df


def load_data(csv_path):
    if csv_path:
        print(f"[+] Loading data from {csv_path}")
        return pd.read_csv(csv_path)
    print("[+] No CSV supplied - generating synthetic patient data")
    return generate_synthetic_data()


# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------
def build_preprocessor(df):
    id_like = [c for c in df.columns if "id" in c.lower()]
    features = df.drop(columns=id_like)

    num_cols = features.select_dtypes(include=np.number).columns.tolist()
    cat_cols = features.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]), cat_cols),
    ])
    return features, preprocessor, num_cols, cat_cols


# ----------------------------------------------------------------------
# 3. Dimensionality reduction
# ----------------------------------------------------------------------
def run_pca(X, variance=0.90):
    pca_full = PCA(random_state=RANDOM_STATE).fit(X)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, variance) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cum) + 1), cum, marker="o")
    plt.axhline(variance, color="r", ls="--", label=f"{int(variance*100)}%")
    plt.xlabel("Components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pca_variance.png", dpi=120)
    plt.close()

    print(f"[+] PCA: {n_comp} components retain {variance:.0%} of variance")
    X_reduced = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit_transform(X)
    X_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    return X_reduced, X_2d


# ----------------------------------------------------------------------
# 4. Clustering
# ----------------------------------------------------------------------
def find_best_k(X, k_range=range(2, 9)):
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(list(k_range), inertias, marker="o")
    ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
    ax[1].plot(list(k_range), sils, marker="o", color="green")
    ax[1].set(title="Silhouette score", xlabel="k", ylabel="Score")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/choose_k.png", dpi=120)
    plt.close()

    best_k = list(k_range)[int(np.argmax(sils))]
    print(f"[+] Best k by silhouette: {best_k}")
    return best_k


def evaluate(name, X, labels):
    mask = labels != -1  # ignore DBSCAN noise
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2:
        return {"model": name, "clusters": n_clusters}
    return {
        "model": name,
        "clusters": n_clusters,
        "silhouette": round(silhouette_score(X[mask], labels[mask]), 3),
        "davies_bouldin": round(davies_bouldin_score(X[mask], labels[mask]), 3),
        "calinski_harabasz": round(calinski_harabasz_score(X[mask], labels[mask]), 1),
    }


def run_clustering(X, k):
    models = {
        "KMeans": KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE),
        "Hierarchical": AgglomerativeClustering(n_clusters=k, linkage="ward"),
        "DBSCAN": DBSCAN(eps=1.2, min_samples=10),
    }
    labels, results = {}, []
    for name, model in models.items():
        lab = model.fit_predict(X)
        labels[name] = lab
        results.append(evaluate(name, X, lab))

    summary = pd.DataFrame(results)
    print("\n[+] Model comparison:\n", summary.to_string(index=False), "\n")
    summary.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
    return labels


# ----------------------------------------------------------------------
# 5. Profiling & visualisation
# ----------------------------------------------------------------------
def plot_clusters(X_2d, labels, name):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels,
                    palette="tab10", s=30, alpha=0.8)
    plt.title(f"{name} clusters (PCA 2D projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/clusters_{name.lower()}.png", dpi=120)
    plt.close()


def profile_clusters(df, labels, num_cols, cat_cols):
    data = df.copy()
    data["cluster"] = labels

    num_profile = data.groupby("cluster")[num_cols].mean().round(2)
    num_profile["size"] = data["cluster"].value_counts().sort_index()
    print("[+] Numeric cluster profile (means):\n", num_profile, "\n")
    num_profile.to_csv(f"{OUT_DIR}/cluster_profile.csv")

    for c in cat_cols:
        print(f"[+] {c} distribution per cluster (%):")
        print((pd.crosstab(data["cluster"], data[c], normalize="index") * 100)
              .round(1), "\n")

    # heatmap of standardised means -> easy to see what defines each cluster
    z = (num_profile[num_cols] - num_profile[num_cols].mean()) / \
        num_profile[num_cols].std(ddof=0)
    plt.figure(figsize=(8, 4))
    sns.heatmap(z, annot=num_profile[num_cols], fmt=".1f",
                cmap="coolwarm", center=0)
    plt.title("Cluster profiles (colour = relative level, text = mean)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cluster_heatmap.png", dpi=120)
    plt.close()

    return data


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unsupervised Patient Data Analysis")
    parser.add_argument("--csv", type=str, default=None, help="Path to patient CSV")
    parser.add_argument("--k", type=int, default=None, help="Number of clusters")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data(args.csv)
    print(f"[+] Shape: {df.shape}\n{df.head()}\n")

    features, preprocessor, num_cols, cat_cols = build_preprocessor(df)
    X = preprocessor.fit_transform(features)
    X = X.toarray() if hasattr(X, "toarray") else X

    X_reduced, X_2d = run_pca(X)

    k = args.k or find_best_k(X_reduced)
    all_labels = run_clustering(X_reduced, k)

    for name, lab in all_labels.items():
        plot_clusters(X_2d, lab, name)

    # Use KMeans for the final patient profiles
    result = profile_clusters(features, all_labels["KMeans"], num_cols, cat_cols)
    id_cols = [c for c in df.columns if "id" in c.lower()]
    final = pd.concat([df[id_cols], result], axis=1)
    final.to_csv(f"{OUT_DIR}/patients_with_clusters.csv", index=False)

    print(f"[+] Done. Results saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
