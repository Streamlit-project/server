import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def min_max_standardization(df):
    return (df - df.min()) / (df.max() - df.min())

def z_score_standardization(df):
    return (df - df.mean()) / df.std()