from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import streamlit as st

def perform_pca_and_kmeans(data, n_clusters=3, n_components=None, init_method='k-means++', max_iter=300, n_init=10):
    # Vérifier la dimensionnalité des données
    n_samples, n_features = data.shape

    if n_features < 2:
        st.error("Les données doivent avoir au moins deux attributs pour appliquer PCA avec 2 composantes principales.")

    # Déterminer le nombre de composantes à utiliser pour PCA
    if n_components is None:
        n_components = min(n_samples, n_features)

    # Appliquer PCA avec le nombre approprié de composantes
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(data)
    
    # Appliquer K-means sur les données projetées
    kmeans = KMeans(n_clusters=n_clusters, init=init_method, max_iter=max_iter, n_init=n_init, random_state=42)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Compter le nombre de points dans chaque cluster
    points_count = dict(Counter(labels))
    
    return X, labels, centroids, points_count

def perform_pca_and_dbscan(data, n_components=None, eps=0.5, min_samples=5):
    # Vérifier la dimensionnalité des données
    n_samples, n_features = data.shape

    if n_features < 2:
        st.error("Les données doivent avoir au moins deux attributs pour appliquer PCA avec 2 composantes principales.")

    # Déterminer le nombre de composantes à utiliser pour PCA
    if n_components is None:
        n_components = min(n_samples, n_features)

    # Appliquer PCA avec le nombre approprié de composantes
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(data)

    # Effectuer le clustering avec DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    # Déterminer le nombre de clusters formés
    n_clusters_formed = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    labels_true = np.zeros(n_samples)

    # Retourner les nouveaux labels et le nombre de clusters formés
    return X, labels, n_clusters_formed, n_noise_, labels_true
