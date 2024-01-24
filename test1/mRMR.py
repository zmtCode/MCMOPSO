import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pandas as pd
def get_10fold_cv_pls(X, y):
    """
    十折交叉验证，返回y_predicts, loo_RMSE, loo_R2
    :param pls: 模型
    :param X: 特征集合, 要求是矩阵类型或者ndarray
    :param y: 因变量
    :return: y_predicts, loo_RMSE, loo_R2
    """
    row, column = X.shape
    y_predicts = np.zeros((row, 1))  # 存放测试集的预测结果
    pls = PLSRegression(n_components=3)  # 实例化pls模型，成分为1
    kf = KFold(n_splits=10)  # 十折交叉验证
    for train_index, test_index in kf.split(X):
        x_train, y_train = X[train_index], y[train_index]
        x_test, y_test = X[test_index], y[test_index]
        pls.fit(x_train, y_train)  # 训练集建模
        y_predicts[test_index] = pls.predict(x_test)  # 预测
    return y_predicts
# df_x = pd.read_excel(r'C:\Users\Administrator\Desktop\data1\ExogenousSubstance.xlsx', sheet_name='Sheet1',index_col=0)  # 内源性物质
# df_y = pd.read_excel(r'C:\Users\Administrator\Desktop\data1\DrugEffectIndex(1).xlsx', sheet_name='Sheet1',index_col=0)
df_x = pd.read_excel(r'C:\Users\Administrator\Desktop\paper2回归任务\data\data big sample\ResidentialBuildingDataSet.xlsx',index_col=0)  # 内源性物质
df_y = pd.read_excel(r'C:\Users\Administrator\Desktop\paper2回归任务\data\data big sample\ResidentialBuildingDataSety.xlsx',index_col=0)

# X = df_x
# y = df_y['y2']
X = df_x.iloc[:,:-1]
y = df_y['y1']
m = X.shape[1]




# select top 10 features using mRMR
from mrmr import mrmr_regression

rmse = []
for i in range(int(m*0.1),m,round(m*0.01)):
# for i in range(791,799,1):
    rmse1 = np.sqrt(mean_squared_error(y, get_10fold_cv_pls(X[mrmr_regression(X=X, y=y, K=i)].values, y)))
    print(rmse1)
    rmse.append(rmse1)
print(min(rmse))
# selected_features = mrmr_regression(X=X, y=y, K=27)
# print(selected_features)
# print(np.sqrt(mean_squared_error(y, get_10fold_cv_pls(X[selected_features].values, y))))
