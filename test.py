import pandas as pd
import numpy as np

df = pd.read_csv('netflix.csv')

df.describe()

print(df.isnull())