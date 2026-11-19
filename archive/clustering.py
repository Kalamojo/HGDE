# Clustering Algorithms Implementation for Human-Guided Dynamic Embeddings

import numpy as np
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering)
from sklearn.preprocessing import normalize

def run_kmeans(embeddings, n_clusters, use_normalization=True, random_state=123):
    """
    Baseline KMeans clustering algorithm
    Input: 'embeddings' that is a np.ndarray, 'n_clusters' which is the number of clusters, and 'random_state' for reproducibility 
    Return: Cluster labels and trained model
    """
    if use_normalization:
        X = normalize(embeddings, norm='l2')
    else:
        X = embeddings

    model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, max_iter=300, random_state=random_state)
    labels = model.fit_predict(X)

    return labels, model

def run_dbscan(embeddings, use_normalization=True, eps=0.5, min_samples=5):
    """
    DBSCAN clustering algorithm
    Input: 'embeddings' that is a np.ndarray, 'eps' which is the maximum distance between two samples for them to be considered as in the same neighborhood, and 'min_samples' which is the number of samples in a neighborhood for a point to be considered as a core point
    Return: Cluster labels and trained model
    """
    if use_normalization:
        X = normalize(embeddings, norm='l2')
    else:
        X = embeddings

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    return labels, model

def run_hierarchical(embeddings, n_clusters, use_normalization=True):
    """
    Agglomerative clustering algorithm
    Input: 'embeddings' that is a np.ndarray and 'n_clusters' which is the number of clusters
    Return: Cluster labels and trained model
    """
    if use_normalization:
        X = normalize(embeddings, norm='l2')
    else:
        X = embeddings 

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X)

    return labels, model

def run_spectral(embeddings, n_clusters, use_normalization=True, random_state=123):
    """
    Spectral clustering algorithm
    Input: 'embeddings' that is a np.ndarray, 'n_clusters' which is the number of clusters, and 'random_state' for reproducibility 
    Return: Cluster labels and trained model
    """
    if use_normalization:
        X = normalize(embeddings, norm='l2')
    else:
        X = embeddings

    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=random_state)
    labels = model.fit_predict(X)

    return labels, model

def run_meta(embeddings):
    return None, None

def cluster_embeddings(embeddings, algorithm="kmeans", **kwargs):

    algorithms = {
        "kmeans": run_kmeans,
        "dbscan": run_dbscan,
        "hierarchical": run_hierarchical,
        "spectral": run_spectral
    }

    if algorithm not in algorithms:
        print("Use kmeans, dbscan, hierarchial, spectral, or meta")
    
    return algorithms[algorithm](embeddings, **kwargs)