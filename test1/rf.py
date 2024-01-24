import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 假设data是你的数据，包括特征和目标变量

num_iterations = 10
num_trees = 10000
r_squared_values = []
df_x = pd.read_excel(r'E:\RF-FeatureSelection-PowerAnalysis-master\Real data 1 (Acharjee et al., 2016)\xdata.xlsx')
X = df_x.iloc[:, 2:]
X1 = ['LPC(18:1)_[M+H]1+', 'LPC(18:0)_[M+H]1+ or LPE(21:0)_[M+H]1+', 'Cer(35:3)_[M+H]1+']
print(X.shape)
Xname = X.columns.tolist()
df_y = pd.read_excel(r'E:\RF-FeatureSelection-PowerAnalysis-master\Real data 1 (Acharjee et al., 2016)\ydata.xlsx')

y = df_y['Relative liver weight']

for j in range(1, num_iterations+1):
    print('Iteration: {}'.format(j))
    rf = RandomForestRegressor(n_estimators=num_trees)
    rf.fit(X[X1], y)  # 假设第一列是目标变量，其余列是特征
    y_pred = rf.predict(X[X1])
    r_squared_values.append(r2_score(y, y_pred))

# 计算平均值
average_r_squared = np.mean(r_squared_values)

print("Average R-squared: ", average_r_squared)
