################################
#####  Fonction Back-end  ######
################################
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
import numpy as np


