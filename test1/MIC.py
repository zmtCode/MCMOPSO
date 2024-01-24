import numpy as np
import pandas as pd
from minepy import MINE
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, KFold


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
    # pls = PLSRegression(n_components=3)  # 实例化pls模型，成分为1
    rf = RandomForestRegressor()
    kf = KFold(n_splits=10)  # 十折交叉验证
    #十折交叉验证
    for train_index, test_index in kf.split(X):
        x_train, y_train = X[train_index], y[train_index]
        x_test, y_test = X[test_index], y[test_index]
        # pls.fit(x_train, y_train)  # 训练集建模
        # y_predicts[test_index] = pls.predict(x_test)  # 预测
        rf.fit(x_train, y_train)
        y_predicts[test_index] = rf.predict(x_test).reshape((-1,1))
    # for train_index, test_index in kf.split(X):
    #     x_train, y_train = X[train_index], y[train_index]
    #     x_test, y_test = X[test_index], y[test_index]
    #     pls.fit(x_train, y_train)  # 训练集建模
    #     y_predicts[test_index] = pls.predict(x_test)  # 预测

    return y_predicts




def MIC(x, y):
    """
            计算最大信息系数
            :param x:
            :param y:
            :return:
            """
    m = MINE()
    m.compute_score(x, y)  # 搞不懂这一步干嘛用的
    return m.mic()


def sort_xname_by_MIC(X, y):
    """
            用最大信息系数MIC计算X中所有特征与y的相关性，根据这个相关性对特征进行排序
            :param X: Dataframe类型
            :param y:
            :return: 排序后的相关性序列，以及排序后的列名
            """
    MIC_Series = pd.Series()
    xname = X.columns.tolist()
    for i in xname:
        MIC_value = MIC(X[i], y)  # 计算X[i]这一列和y的MIC值
        MIC_Series[i] = MIC_value
    # 对序列进行排序
    MIC_Series = MIC_Series.sort_values(ascending=False)  # 降序排序
    xname_sorted = MIC_Series.index.tolist()
    return xname_sorted

def de_irrelevant( X_original, y, x_proportion=0.1, stepLength=0.01, cycles=89):
    """
            判断多少百分比的特征个数最合适
            :param X_original: 原始特征集合（根据特征的MIC得分排好序）
            :param x_proportion: 初始特征占比
            :param stepLength: 步长，特征占比一次加多少值
            :param cycles: 循环次数
            :return: 去除无关特征的特征名，特征百分比
            """
    xname_list = X_original.columns.values.tolist()
    X_original = X_original.values
    y = y.values
    series_MSE = pd.Series(name='MSE')
    # series_R2 = pd.Series(name='R2')
    # x_proportion = 0.2  # 取百分之20的特征
    proportion = x_proportion  # 取百分之10的特征
    # feature_number = len(x_name_MIC)  # 特征总数
    feature_number = X_original.shape[1]  # 特征总数
    for i in range(cycles):  # cycles循环次数
        sub_feature_number = round(feature_number * proportion)  # 取百分之十的特征, 四舍五入取整
        # sub_x_name_MIC = x_name_MIC[0: sub_feature_number]  # 前sub_feature_number个xname
        X = X_original.T[0:sub_feature_number, :]  # X_original.T:对X_original进行转置，列变行之后进行切片 (1028, 54)
        X = X.T  # (54, 1028)
        # print(X, X.shape)
        # 使用这部分特征进行回归建模（留一交叉验证）
        y_predicts = []  # 存放测试集的预测结果
        pls = PLSRegression(n_components=3)  # 实例化pls模型，成分为3
        # rf = RandomForestRegressor()
        loo = LeaveOneOut()  # 留一交叉验证
        for train_index, test_index in loo.split(X):
            x_train, y_train = X[train_index], y[train_index]
            x_test, y_test = X[test_index], y[test_index]  # 只有一个样本
            # print('x_train', x_train, 'x_test', x_test)
            pls.fit(x_train, y_train)  # 训练集建模
            y_test_predict = pls.predict(x_test)  # 对i这个样本进行预测
            y_predicts.append(y_test_predict[0][0])  # 只append数值
        y_predicts = np.ravel(y_predicts)
        loo_RMSE = np.sqrt(mean_squared_error(y_predicts, y))  # RMSE(y_predicts, y)
        # print('loo_RMSE', loo_RMSE)
        # loo_R2 = getR2(y_predicts, y)
        # print('loo_R2', loo_R2)
        series_MSE[proportion] = loo_RMSE
        # series_R2[x_proportion] = loo_R2

        # proportion = proportion + 0.01
        proportion = proportion + stepLength

    # 保存结果
    # result = pd.DataFrame()
    # result['MSE'] = series_MSE
    # result['R2'] = series_R2
    # result.to_excel('2.2result.xlsx', sheet_name='Sheet1')
    # 对series_MSE进行降序排序，取出得分最高的特征(特征是索引)
    series_MSE = series_MSE.sort_values()
    proportionOfX = series_MSE.index.tolist()[0]
    proportionOfFeature = series_MSE[series_MSE.values == min(series_MSE)].index
    xname_cut = int(round(feature_number * proportionOfX))
    xname_removedIrrelevantFeature = xname_list[0:int(round(feature_number * proportionOfX))]  # 根据百分比取出对应的特征列表
    return xname_removedIrrelevantFeature, proportionOfX  # 去除无关特征的特征名，特征百分比


if __name__ == "__main__":
    df_x = pd.read_excel(r'C:\Users\Administrator\Desktop\data1\ExogenousSubstance.xlsx', sheet_name='Sheet1',index_col=0)  # 内源性物质
    df_y = pd.read_excel(r'C:\Users\Administrator\Desktop\data1\DrugEffectIndex(1).xlsx', sheet_name='Sheet1',index_col=0)
    # df_x = pd.read_excel(r'C:\Users\Administrator\Desktop\paper2回归任务\data\blogData_test\blogData_test1.xlsx',index_col=0)  # 内源性物质
    X = df_x
    # print(X)
    y = df_y['y4']
    Xname = X.columns.tolist()
    xname_sorted = sort_xname_by_MIC(X, y)
    # print(xname_sorted)
    # print(np.sqrt(mean_squared_error(y, get_10fold_cv_pls(X[Xname].values, y))))
    print(np.sqrt(mean_squared_error(y, get_10fold_cv_pls(X[xname_sorted].values, y))))
    xname_removedIrrelevantFeature, proportionOfFeature = de_irrelevant(X[xname_sorted], y)
    print(np.sqrt(mean_squared_error(y, get_10fold_cv_pls(X[xname_removedIrrelevantFeature].values, y))))
    print(proportionOfFeature, len(xname_removedIrrelevantFeature))
