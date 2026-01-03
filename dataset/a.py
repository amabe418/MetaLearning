import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../original_datasets.csv")

meta_feature = [c for c in df.columns if c not in ["openml_task","flow","area_under_roc_curve"]]
scaler = MinMaxScaler()
df[meta_feature] = scaler.fit_transform(df[meta_feature])