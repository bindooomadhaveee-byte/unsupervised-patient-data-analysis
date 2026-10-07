# Unsupervised Patient Data Analysis

Clusters patients into groups using unsupervised learning (PCA + KMeans, Hierarchical, DBSCAN).

## Setup
```
pip install -r requirements.txt
```

## Run
```
python patient_clustering.py                               # synthetic data
python patient_clustering.py --csv sample_patients.csv     # sample CSV
python patient_clustering.py --csv your_file.csv --k 4     # your data, 4 clusters
```

## Outputs (saved in `output/`)
- pca_variance.png, choose_k.png
- clusters_kmeans.png, clusters_hierarchical.png, clusters_dbscan.png
- cluster_heatmap.png
- model_comparison.csv, cluster_profile.csv, patients_with_clusters.csv

## Project files
- patient_clustering.py - main script
- sample_patients.csv - example input data
- requirements.txt - dependencies
