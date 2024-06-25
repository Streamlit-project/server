from matplotlib import colors, pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from back.clustering_back import perform_pca_and_kmeans
from front.menu import menu_with_redirect
import plotly.graph_objects as go

# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

st.title('Clustering K-means')

st.sidebar.header('Paramètres de l\'algorithme')

df = st.session_state.df_normalized

if df is not None:
    # Afficher un aperçu des données
    st.write("Aperçu des données :", df)

    # Sélectionner les colonnes à utiliser pour le clustering
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    columns = st.multiselect("Sélectionner les colonnes à utiliser pour le clustering", numeric_columns)

    if columns:
        # Sidebar pour les paramètres de l'algorithme
        n_components = st.sidebar.selectbox('Nombre de composantes principales', [2, 3], key='n_components')
        n_clusters = st.sidebar.slider('Nombre de clusters (k)', 2, 10, 3, key='n_clusters')
        init_method = st.sidebar.selectbox('Méthode d\'initialisation', ['k-means++', 'random'], key='init_method')
        max_iter = st.sidebar.slider('Nombre maximum d\'itérations', 100, 1000, 300, key='max_iter')
        n_init = st.sidebar.slider('Nombre d\'initialisations différentes', 1, 10, 10, key='n_init')

        # Vérifier le nombre de colonnes sélectionnées
        n_features = len(columns)

        # Appliquer le clustering K-means avec PCA en utilisant la fonction backend
        selected_data = df[columns].values
        n_features = selected_data.shape[1]

        if n_features < 2:
            st.error("Veuillez sélectionner au moins deux colonnes pour appliquer la PCA.")
        elif n_components > n_features:
            st.error(f"Le nombre de composantes principales doit être inférieur ou égal au nombre de caractéristiques (nombres de composantes principales : {n_components}, nombre de caractéristiques : {n_features}).")
        else:
            X, labels, centroids, points_count = perform_pca_and_kmeans(selected_data, n_clusters=n_clusters, n_components=n_components, init_method=init_method, max_iter=max_iter, n_init=n_init)

            # Vérifier le nombre de composantes principales dans X
            if n_components < 2:
                st.write("Le nombre de composantes principales doit être supérieur ou égal à 2 pour afficher les résultats en 2D.")
                exit()
            else:
                # Définir les couleurs à utiliser pour les clusters
                colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

                # Affichage des résultats du clustering en 2D ou 3D avec les centroids
                if n_components == 2:
                    fig, ax = plt.subplots()
                    for i in range(n_clusters):
                        points = X[labels == i]
                        ax.scatter(points[:, 0], points[:, 1], c=colors[i % len(colors)], label=f'Cluster {i}')
                    ax.scatter(centroids[:, 0], centroids[:, 1], s=100, c='yellow', marker='X', edgecolor='black', label='Centroids')
                    ax.legend()
                    ax.set_xlabel('Component 1')
                    ax.set_ylabel('Component 2')
                    st.pyplot(fig)
                else:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    for i in range(n_clusters):
                        points = X[labels == i]
                        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[i % len(colors)], label=f'Cluster {i}')
                    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=100, c='yellow', marker='X', edgecolor='black', label='Centroids')
                    ax.legend()
                    ax.set_xlabel('Component 1')
                    ax.set_ylabel('Component 2')
                    ax.set_zlabel('Component 3')

                    # Ajuster les marges du graphique pour éviter que les labels ne soient coupés
                    fig.tight_layout()
                    fig.subplots_adjust(bottom=0.5, left=0.5, right=2.5, top=2.5)

                    st.pyplot(fig)

                # Conversion des données en DataFrame pour un affichage sous forme de tableau
                clusters_data = {
                    'Cluster': [],
                }

                for i in range(n_clusters):
                    clusters_data['Cluster'].append(f"Cluster {i}")

                for j in range(n_components):
                    clusters_data[f'Component {j+1}'] = []

                for i in range(n_clusters):
                    for j in range(n_components):
                        clusters_data[f'Component {j+1}'].append(centroids[i][j])

                df = pd.DataFrame(data=clusters_data)

                # Affichage du DataFrame dans Streamlit
                st.write("Centroïds de chaque cluster :")
                st.table(df)

    else:
        st.write("Veuillez sélectionner les colonnes à utiliser pour le clustering.")
else:
    st.write("Veuillez télécharger un fichier CSV pour commencer le clustering.")