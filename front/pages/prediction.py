import streamlit as st
from menu import menu_with_redirect
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
from back.prediction_back import SGDClassifier_prediction, RandomForest_prediction, KNN_prediction, SGDRegressor_prediction, RidgeRegressor_prediction, LassoRegressor_prediction

st.header('Prediction')

# Classification algorithms

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

# Regression algorithms

def SGDRegressor_front():
    loss = st.selectbox('Loss function:',('huber', 'squared_error', 'epsilon_insensitive', 'squared_epsilon_insensitive'))
    penalty = st.selectbox('Penalty:',('l2', 'l1', 'elasticnet'))
    alpha = st.number_input('Alpha:', value=0.0001)
    max_iter = st.number_input('Max iterations:', value=1000)
    tol = st.number_input('Tolerance:', value=1e-3)

    stats = SGDRegressor_prediction(st.session_state.df_normalized, target, loss, penalty, alpha, max_iter, tol, test_size)
    st.write('R2 score : ' + str(stats['r2_score']))
    st.write('Mean squared error : ' + str(stats['mean_squared_error']))

def RidgeRegressor_front():
    alpha = st.number_input('Alpha:', value=1.0)
    fit_intercept = st.checkbox('Fit intercept', value=True)
    max_iter = st.number_input('Max iterations:', value=None, step=1, placeholder='None')
    tol = st.number_input('Tolerance:', value=1e-4)

    stats = RidgeRegressor_prediction(st.session_state.df_normalized, target, alpha, fit_intercept, max_iter, tol, test_size)
    st.write('R2 score : ' + str(stats['r2_score']))
    st.write('Mean squared error : ' + str(stats['mean_squared_error']))

def LassoRegressor_front():
    alpha = st.number_input('Alpha:', value=1.0)
    fit_intercept = st.checkbox('Fit intercept', value=True)
    max_iter = st.number_input('Max iterations:', value=1000)
    tol = st.number_input('Tolerance:', value=1e-4)

    stats = LassoRegressor_prediction(st.session_state.df_normalized, target, alpha, fit_intercept, max_iter, tol, test_size)
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

        if st.session_state.df_normalized[target].nunique() > 2:
            st.error('The selected target is not compatible with a classification task. Please choose another target.')
            exit()

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
        prediction_algorithm = st.radio('Choose a prediction algorithm:', ('SGDRegressor', 'RidgeRegressor', 'LassoRegressor'))
        st.subheader(prediction_algorithm)
        st.write('Config the '+str(prediction_algorithm)+' params')
        target = st.selectbox('Target:', st.session_state.df_normalized.columns)
        test_size = st.number_input('Test size (%) :', value=20, step=1, max_value=95, min_value=1) / 100

        if prediction_algorithm == 'SGDRegressor':
            SGDRegressor_front()
        elif prediction_algorithm == 'RidgeRegressor':
            RidgeRegressor_front()
        elif prediction_algorithm == 'LassoRegressor':
            LassoRegressor_front()
        else:
            st.error('Invalid prediction method')
            exit()
