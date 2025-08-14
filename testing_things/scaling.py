import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Sample dataset
data = {
    'age': [18, 25, 40, 60],
    'income': [20000, 50000, 80000, 100000]
}
df = pd.DataFrame(data)
print(df)


scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(scaled, columns=df.columns)
print(df_scaled)

