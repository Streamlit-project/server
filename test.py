import pandas as pd
import numpy as np

df = pd.read_csv('netflix.csv')
df = df.replace('Unknown', np.nan)

print(df.isnull())