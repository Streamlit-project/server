import streamlit as st
from menu import menu_with_redirect
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
from back.prediction_back import SGDClassifier_prediction, RandomForest_prediction, KNN_prediction, SGDRegressor_prediction

st.header('Prediction')

def SGDClassifier_front():
    loss = st.selectbox('Loss function:',('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'))
    penalty = st.selectbox('Penalty:',('l2', 'l1', 'elasticnet'))
    alpha = st.number_input('Alpha:', value=0.0001)
    max_iter = st.number_input('Max iterations:', value=1000)
    tol = st.number_input('Tolerance:', value=1e-3)

    stats = SGDClassifier_prediction(st.session_state.df_normalized, target, loss, penalty, alpha, max_iter, tol, test_size)
    st.write('Accuracy score : ' + str(stats['accuracy_score']))
    st.write('F1 score : ' + str(stats['f1_score']))
    st.write('Confusion matrix :')
    st.table(stats['confusion_matrix'])
    st.write(str(stats['confusion_matrix']))


def RandomForest_front():
    n_estimators = st.number_input('Number of trees in the forest:', value=100)
    max_depth = st.number_input('Max depth of the tree:', value=None, step=1, placeholder='None')
    min_samples_split = st.number_input('Min samples split:', value=2)
    min_samples_leaf = st.number_input('Min samples leaf:', value=1)
    max_features = st.selectbox('Max features:', (None, 'sqrt', 'log2'))
    max_leaf_nodes = st.number_input('Max leaf nodes:', value=None, step=1, placeholder='None')

    stats = RandomForest_prediction(st.session_state.df_normalized, target, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, test_size)
    st.write('Accuracy score : ' + str(stats['accuracy_score']))
    st.write('F1 score : ' + str(stats['f1_score']))
    st.write('Confusion matrix :')
    st.table(stats['confusion_matrix'])
    st.write(str(stats['confusion_matrix']))

def KNN_front():
    n_neighbors = st.number_input('Number of neighbors:', value=5, step=1)
    weights = st.selectbox('Weight function:', ('uniform', 'distance'))
    algorithm = st.selectbox('Algorithm:', ('auto', 'ball_tree', 'kd_tree', 'brute'))
    leaf_size = st.number_input('Leaf size:', value=30, step=1)

    stats = KNN_prediction(st.session_state.df_normalized, target, n_neighbors, weights, algorithm, leaf_size, test_size)
    st.write('Accuracy score : ' + str(stats['accuracy_score']))
    st.write('F1 score : ' + str(stats['f1_score']))
    st.write('Confusion matrix :')
    st.table(stats['confusion_matrix'])
    st.write(str(stats['confusion_matrix']))
    

def SGDRegressor_front():
    loss = st.selectbox('Loss function:',('huber', 'squared_error', 'epsilon_insensitive', 'squared_epsilon_insensitive'))
    penalty = st.selectbox('Penalty:',('l2', 'l1', 'elasticnet'))
    alpha = st.number_input('Alpha:', value=0.0001)
    max_iter = st.number_input('Max iterations:', value=1000)
    tol = st.number_input('Tolerance:', value=1e-3)

    stats = SGDRegressor_prediction(st.session_state.df_normalized, target, loss, penalty, alpha, max_iter, tol, test_size)
    st.write('R2 score : ' + str(stats['r2_score']))
    st.write('Mean squared error : ' + str(stats['mean_squared_error']))

if 'df_normalized' not in st.session_state:
    st.error('Please import, clean and standardize dataset before standardizing it.')
else:
    predictionType = st.radio('Choose a prediction type:', ('Classification', 'Regression'))
    if(predictionType == 'Classification'):
        prediction_algorithm = st.radio('Choose a prediction algorithm:', ('SGDClassifier', 'RandomForest', 'KNN'))
        st.subheader(prediction_algorithm)
        st.write('Config the '+str(prediction_algorithm)+' params')
        target = st.selectbox('Target:', st.session_state.df_normalized.columns)
        test_size = st.number_input('Test size:', value=0.2, step=0.1, max_value=0.9, min_value=0.01)
        
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
        prediction_algorithm = st.radio('Choose a prediction algorithm:', ('SGDRegressor', 'RandomForestRegressor', 'KNNRegressor'))
        st.subheader(prediction_algorithm)
        st.write('Config the '+str(prediction_algorithm)+' params')
        target = st.selectbox('Target:', st.session_state.df_normalized.columns)
        test_size = st.number_input('Test size:', value=0.2, step=0.1, max_value=0.9, min_value=0.01)
        
        if prediction_algorithm == 'SGDRegressor':
            SGDRegressor_front()
        else:
            st.error('Invalid prediction method')
            exit()
