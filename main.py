from sklearn.datasets import load_iris
import pandas as pd

ds = load_iris()
df = pd.DataFrame(data= ds["data"], columns=ds["feature_names"])
target_names = [ds.target_names[x] for x in ds.target]
df['species'] = target_names
print(df)