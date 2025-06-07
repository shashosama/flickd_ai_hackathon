# check_columns.py
import pandas as pd

df = pd.read_csv("data/catalog.csv")
print(df.columns.tolist())
