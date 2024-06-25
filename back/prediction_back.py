from sklearn.linear_model import SGDClassifier, SGDRegressor, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def SGDClassifier_prediction(target, loss, penalty, alpha, max_iter, tol, test_size):
    X = st.session_state.df_normalized.drop(columns=[target])
    y = st.session_state.df_cleaned[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter, tol=tol)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    dataframe_cr =  pd.DataFrame(
        classification_report(y_test, y_pred, target_names=y.unique(), output_dict=True)
    ).transpose()

    return {'confusion_matrix': confusion_matrix(y_test, y_pred), 'y_unique': y.unique(), 'dataframe_cr': dataframe_cr}

def RandomForest_prediction(target, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, test_size):
    X = st.session_state.df_normalized.drop(columns=[target])
    y = st.session_state.df_cleaned[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)   
    
    dataframe_cr =  pd.DataFrame(
        classification_report(y_test, y_pred, target_names=y.unique(), output_dict=True)
    ).transpose()
  
    return {'confusion_matrix': confusion_matrix(y_test, y_pred), 'y_unique': y.unique(), 'dataframe_cr': dataframe_cr}

def KNN_prediction(target, n_neighbors, weights, algorithm, leaf_size, test_size):
    X = st.session_state.df_normalized.drop(columns=[target])
    y = st.session_state.df_cleaned[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    dataframe_cr =  pd.DataFrame(
        classification_report(y_test, y_pred, target_names=y.unique(), output_dict=True)
    ).transpose()
    
    return {'confusion_matrix': confusion_matrix(y_test, y_pred), 'y_unique': y.unique(), 'dataframe_cr': dataframe_cr}

def RandomForestRegressor_prediction(target, max_depth, criterion, n_estimators, test_size):
    X = st.session_state.df_normalized.drop(columns=[target])
    y = st.session_state.df_cleaned[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    reg = RandomForestRegressor(max_depth=max_depth, criterion=criterion, n_estimators=n_estimators)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Valeurs réelles')
    ax.set_ylabel('Prédictions')
    ax.set_title('Prédictions par rapport aux valeurs réelles')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)
    
    return {'r2_score' : r2_score(y_test, y_pred), 'mean_squared_error': mean_squared_error(y_test, y_pred), 'fig': fig}

def RidgeRegressor_prediction(target, alpha, fit_intercept, max_iter, tol, test_size):
    X = st.session_state.df_normalized.drop(columns=[target])
    y = st.session_state.df_cleaned[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    reg = Ridge(alpha=alpha, max_iter=max_iter, tol=tol, fit_intercept=fit_intercept)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Valeurs réelles')
    ax.set_ylabel('Prédictions')
    ax.set_title('Prédictions par rapport aux valeurs réelles')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)
    
    return {'r2_score' : r2_score(y_test, y_pred), 'mean_squared_error': mean_squared_error(y_test, y_pred), 'fig': fig}

def LassoRegressor_prediction(target, alpha, fit_intercept, max_iter, tol, test_size):
    X = st.session_state.df_normalized.drop(columns=[target])
    y = st.session_state.df_cleaned[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    reg = Lasso(alpha=alpha, max_iter=max_iter, tol=tol, fit_intercept=fit_intercept)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Valeurs réelles')
    ax.set_ylabel('Prédictions')
    ax.set_title('Prédictions par rapport aux valeurs réelles')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)
    
    return {'r2_score' : r2_score(y_test, y_pred), 'mean_squared_error': mean_squared_error(y_test, y_pred), 'fig': fig}