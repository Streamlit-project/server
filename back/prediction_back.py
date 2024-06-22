import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
import scipy.stats
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

## Algorithmes de prédiction

## SGDClassifier
## RandomForest
## KNN

def SGDClassifier_prediction(df, target, loss, penalty, alpha, max_iter, tol):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter, tol=tol)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return accuracy_score(y_test, y_pred)

def RandomForest_prediction(df, target, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)    
    
    return accuracy_score(y_test, y_pred)

def KNN_prediction(df, target, n_neighbors, weights, algorithm, leaf_size):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return accuracy_score(y_test, y_pred)