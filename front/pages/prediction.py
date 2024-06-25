import streamlit as st
from menu import menu_with_redirect
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
from back.prediction_back import SGDClassifier_prediction, RandomForest_prediction, KNN_prediction, RandomForestRegressor_prediction, RidgeRegressor_prediction, LassoRegressor_prediction


# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

st.header('Prediction')

# Classification algorithms
def SGDClassifier_front():
    loss = st.selectbox('Loss function:',('hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'))
    penalty = st.selectbox('Penalty:',('l2', 'l1', 'elasticnet'))
    alpha = st.number_input('Alpha:', value=0.0001)
    max_iter = st.number_input('Max iterations:', value=1000)
    tol = st.number_input('Tolerance:', value=1e-3)

    stats = SGDClassifier_prediction(target, loss, penalty, alpha, max_iter, tol, test_size)
    
    # Afficher le raport de classfication
    st.write('Rapport de classification :')
    st.dataframe(stats['dataframe_cr']) 
    
    # Créer un DataFrame à partir de la matrice de confusion
    st.write('Matrice de confusion :')
    df_cm = pd.DataFrame(stats['confusion_matrix'], columns=stats['y_unique'], index=stats['y_unique'])

    # Ajouter les noms des classes aux axes
    df_cm.index.name = 'Vérité'
    df_cm.columns.name = 'Prédiction'

    # Afficher la matrice de confusion avec Streamlit
    st.table(df_cm)


def RandomForest_front():
    n_estimators = st.number_input('Number of trees in the forest:', value=100)
    max_depth = st.number_input('Max depth of the tree:', value=None, step=1, placeholder='None')
    min_samples_split = st.number_input('Min samples split:', value=2)
    min_samples_leaf = st.number_input('Min samples leaf:', value=1)
    max_features = st.selectbox('Max features:', (None, 'sqrt', 'log2'))
    max_leaf_nodes = st.number_input('Max leaf nodes:', value=None, step=1, placeholder='None')

    stats = RandomForest_prediction(target, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, test_size)
    
    # Afficher le raport de classfication
    st.write('Rapport de classification :')
    st.dataframe(stats['dataframe_cr']) 
    
    # Créer un DataFrame à partir de la matrice de confusion
    st.write('Matrice de confusion :')
    df_cm = pd.DataFrame(stats['confusion_matrix'], columns=stats['y_unique'], index=stats['y_unique'])

    # Ajouter les noms des classes aux axes
    df_cm.index.name = 'Vérité'
    df_cm.columns.name = 'Prédiction'

    # Afficher la matrice de confusion avec Streamlit
    st.table(df_cm)


def KNN_front():
    n_neighbors = st.number_input('Number of neighbors:', value=5, step=1)
    weights = st.selectbox('Weight function:', ('uniform', 'distance'))
    algorithm = st.selectbox('Algorithm:', ('auto', 'ball_tree', 'kd_tree', 'brute'))
    leaf_size = st.number_input('Leaf size:', value=30, step=1)

    stats = KNN_prediction(target, n_neighbors, weights, algorithm, leaf_size, test_size)
    
    # Afficher le raport de classfication
    st.write('Rapport de classification :')
    st.dataframe(stats['dataframe_cr']) 
    
    # Créer un DataFrame à partir de la matrice de confusion
    st.write('Matrice de confusion :')
    df_cm = pd.DataFrame(stats['confusion_matrix'], columns=stats['y_unique'], index=stats['y_unique'])

    # Ajouter les noms des classes aux axes
    df_cm.index.name = 'Vérité'
    df_cm.columns.name = 'Prédiction'

    # Afficher la matrice de confusion avec Streamlit
    st.table(df_cm)


# Regression algorithms
def RandomForestRegressor_front():
    max_depth = st.number_input('max_depth:',value=None, step=1, placeholder='None')
    criterion = st.selectbox('criterion:',('squared_error', 'absolute_error', 'friedman_mse', 'poisson'))
    n_estimators = st.number_input('n_estimators:', value=100)

    stats = RandomForestRegressor_prediction(target, max_depth, criterion, n_estimators, test_size)
    st.write('R2 score : ' + str(stats['r2_score']))
    st.write('Mean squared error : ' + str(stats['mean_squared_error']))
    st.pyplot(stats['fig'])


def RidgeRegressor_front():
    alpha = st.number_input('Alpha:', value=1.0)
    fit_intercept = st.checkbox('Fit intercept', value=True)
    max_iter = st.number_input('Max iterations:', value=None, step=1, placeholder='None')
    tol = st.number_input('Tolerance:', value=1e-4)

    stats = RidgeRegressor_prediction(target, alpha, fit_intercept, max_iter, tol, test_size)
    st.write('R2 score : ' + str(stats['r2_score']))
    st.write('Mean squared error : ' + str(stats['mean_squared_error']))
    st.pyplot(stats['fig'])


def LassoRegressor_front():
    alpha = st.number_input('Alpha:', value=1.0)
    fit_intercept = st.checkbox('Fit intercept', value=True)
    max_iter = st.number_input('Max iterations:', value=1000)
    tol = st.number_input('Tolerance:', value=1e-4)

    stats = LassoRegressor_prediction(target, alpha, fit_intercept, max_iter, tol, test_size)
    st.write('R2 score : ' + str(stats['r2_score']))
    st.write('Mean squared error : ' + str(stats['mean_squared_error']))
    st.pyplot(stats['fig'])


if 'df_normalized' not in st.session_state:
    st.error('Please import, clean and standardize dataset before standardizing it.')
else:
    predictionType = st.radio('Choose a prediction type:', ('Classification', 'Regression'))
    if(predictionType == 'Classification'):
        prediction_algorithm = st.radio('Choose a prediction algorithm:', ('SGDClassifier', 'RandomForest', 'KNN'))
        st.subheader(prediction_algorithm)
        st.write('Config the '+str(prediction_algorithm)+' params')
        target = st.selectbox('Target:', st.session_state.df_normalized.columns)
        
        test_size = st.number_input('Test size (%) :', value=20, step=1, max_value=95, min_value=1) / 100

        if prediction_algorithm == 'SGDClassifier':
            SGDClassifier_front()
        elif prediction_algorithm == 'RandomForest':
            RandomForest_front()
        elif prediction_algorithm == 'KNN':
            KNN_front()
        else:
            st.error('Invalid prediction method')
            exit()
            
    elif(predictionType == 'Regression'):
        prediction_algorithm = st.radio('Choose a prediction algorithm:', ('RandomForestRegressor', 'RidgeRegressor', 'LassoRegressor'))
        st.subheader(prediction_algorithm)
        st.write('Config the '+str(prediction_algorithm)+' params')
        target = st.selectbox('Target:', st.session_state.df_normalized.columns)
        test_size = st.number_input('Test size (%) :', value=20, step=1, max_value=95, min_value=1) / 100

        if prediction_algorithm == 'RandomForestRegressor':
            RandomForestRegressor_front()
        elif prediction_algorithm == 'RidgeRegressor':
            RidgeRegressor_front()
        elif prediction_algorithm == 'LassoRegressor':
            LassoRegressor_front()
        else:
            st.error('Invalid prediction method')
            exit()
