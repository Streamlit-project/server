from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from back.clustering_back import perform_pca_and_kmeans
from front.menu import menu_with_redirect

# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

st.title('Clustering K-means')

st.sidebar.header('Paramètres de l\'algorithme')

df = st.session_state.df_normalized

if df is not None:
    # Afficher un aperçu des données
    st.write("Aperçu des données :", df.head())

    # Sélectionner les colonnes à utiliser pour le clustering
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    columns = st.multiselect("Sélectionner les colonnes à utiliser pour le clustering", numeric_columns)

    if columns:
        # Sidebar pour les paramètres de l'algorithme
        n_clusters = st.sidebar.slider('Nombre de clusters (k)', 2, 10, 3, key='n_clusters')
        init_method = st.sidebar.selectbox('Méthode d\'initialisation', ['k-means++', 'random'], key='init_method')
        max_iter = st.sidebar.slider('Nombre maximum d\'itérations', 100, 1000, 300, key='max_iter')
        n_init = st.sidebar.slider('Nombre d\'initialisations différentes', 1, 10, 10, key='n_init')

        # Vérifier le nombre de colonnes sélectionnées
        n_features = len(columns)

        # Déterminer le nombre de composantes à utiliser pour PCA
        if n_features < 3:
            n_components = n_features
        else:
            n_components = 2

        # Appliquer le clustering K-means avec PCA en utilisant la fonction backend
        selected_data = df[columns].values
        X, labels, centroids, points_count = perform_pca_and_kmeans(selected_data, n_clusters=n_clusters, n_components=n_components, init_method=init_method, max_iter=max_iter, n_init=n_init)

        # Compter le nombre point par cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        # st.write("Centroids des clusters:", st.table(np.unique(labels, return_counts=True)))
        number_point_ = {
            'Cluster': [],
            'Nombre de points': []
        ,}
        
        for i in unique_labels:
            number_point_['Cluster'].append(f"Cluster {i}")
            number_point_['Nombre de points'].append(counts[i])
        
        # Vérifier le nombre de composantes principales dans X
        if n_components < 2:
            st.write("Le nombre de composantes principales doit être supérieur ou égal à 2 pour afficher les résultats en 2D.")
            exit()
        else:
            # Affichage des résultats du clustering en 2D avec les centroids
            fig, ax = plt.subplots()
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            for i in range(n_clusters):
                points = X[labels == i]
                ax.scatter(points[:, 0], points[:, 1], c=colors[i % len(colors)], label=f'Cluster {i}')
            ax.scatter(centroids[:, 0], centroids[:, 1], s=100, c='yellow', marker='X', edgecolor='black', label='Centroids')
            ax.legend()
            ax.set_xlabel('Component 1')  # Légende de l'axe X
            ax.set_ylabel('Component 2')  # Légende de l'axe Y
            st.pyplot(fig)

            st.write("Centroids des clusters:", centroids)

            # Affichage du DataFrame dans Streamlit
            st.write("Nombre de points dans chaque cluster :")
            st.table(number_point_)

    else:
        st.write("Veuillez sélectionner les colonnes à utiliser pour le clustering.")
else:
    st.write("Veuillez télécharger un fichier CSV pour commencer le clustering.")