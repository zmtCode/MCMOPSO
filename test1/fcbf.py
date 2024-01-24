"""
实现FCBF
高维小样本特征选择算法
时间：20201007
lzq
"""
import numpy as np
import random
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import copy
from minepy import MINE

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

class AMB:
    def __init__(self):
        pass

    def MIC(self, x, y):
        """
        计算最大信息系数
        :param x:
        :param y:
        :return:
        """
        m = MINE()
        m.compute_score(x, y)  # 搞不懂这一步干嘛用的
        return m.mic()

    def sort_xname_by_MIC(self, X, y):
        """
        用最大信息系数MIC计算X中所有特征与y的相关性，根据这个相关性对特征进行排序
        :param X: Dataframe类型
        :param y:
        :return: 排序后的相关性序列，以及排序后的列名
        """
        MIC_Series = pd.Series()
        xname = X.columns.tolist()
        for i in xname:
            MIC_value = self.MIC(X[i], y)  # 计算X[i]这一列和y的MIC值
            MIC_Series[i] = MIC_value
        # 对序列进行排序
        MIC_Series = MIC_Series.sort_values(ascending=False)  # 降序排序
        xname_sorted = MIC_Series.index.tolist()
        return xname_sorted

    # 实现以下AMB
    def train(self, X, y):
        """
        根据近似马尔科夫毯进行特征选择：MIC(x1, x2) > MIC(x2, y) 且
                                        MIC(x1, y) > MIC(x2, y)  认为x1是x2的近似马尔科夫毯，所有以x1为近似马尔科夫毯的特征全部删除
                                        只留 x1
                                        即只需要近似马尔科夫毯，其他全部删除
        :param X: Dataframe类型的数据集
        :return: 特征选择sub_X
        """
        x_subset = []  # 用来存储特征选择的结果:只包括列名
        # 根据MIC相关性对特征进行排序
        # x_name = self.sort_xname_by_MIC(X, y)  # 这里可以满足MIC(x1, y) > MIC(x2, y)
        x_name = X.columns.tolist()  # FCBF的AMB这里不用排序
        while x_name:  # 由于一个空 list 本身等同于 False，所以可以直接使用
            first_xname = x_name[0]  # x_name中取出第一个值之后删除第一个值
            del x_name[0]  # 使用del删除对应下标的元素

            x_subset.append(first_xname)  # 该组中的第一个元素一定是某些特征的近似马尔科夫毯
            #  找出所有以first_xname为近似马尔科夫毯的特征， 然后删除
            for i in x_name:  # i是列名，x298这种类型
                # 1.计算MIC(x1, x2)
                mic_x1x2 = self.MIC(X[first_xname], X[i])
                # 2.计算MIC(x2, y)
                mic_x2y = self.MIC(X[i], y)
                # 3.如果MIC(x1, x2) > MIC(x2, y), 则x1是x2的近似马尔科夫毯，x1可以删除
                if mic_x1x2 > mic_x2y:
                    x_name.remove(i)  # 删除指定值的元素，这里没有重复的列名，因此可以这样删除

                    # 本次循环结束

        self.feature_selected = x_subset
        self.best_RMSE = np.sqrt(mean_squared_error(y, get_10fold_cv_pls(X[self.feature_selected].values, y)))

        # return x_subset, X[x_subset]  # x_subset, sub_X 列名，特征子集

    def predict(self):
        """
        获取被选择的特征名字
        :return:
        """
        return self.feature_selected

class FCBF:
    def __init__(self):
        pass

    def MIC(self, x, y):
        """
        计算最大信息系数
        :param x:
        :param y:
        :return:
        """
        m = MINE()
        m.compute_score(x, y)  # 搞不懂这一步干嘛用的
        return m.mic()

    def sort_xname_by_MIC(self, X, y):
        """
        用最大信息系数MIC计算X中所有特征与y的相关性，根据这个相关性对特征进行排序
        :param X: Dataframe类型
        :param y:
        :return: 排序后的相关性序列，以及排序后的列名
        """
        MIC_Series = pd.Series()
        xname = X.columns.tolist()
        for i in xname:
            MIC_value = self.MIC(X[i], y)  # 计算X[i]这一列和y的MIC值
            MIC_Series[i] = MIC_value
        # 对序列进行排序
        MIC_Series = MIC_Series.sort_values(ascending=False)  # 降序排序
        xname_sorted = MIC_Series.index.tolist()
        return xname_sorted

    # 第一阶段去无关
    def de_irrelevant(self, X_original, y, x_proportion=0.1, stepLength=0.01, cycles=89):
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


    def train(self, X, y):
        xname_sorted = self.sort_xname_by_MIC(X, y)  # 根据MIC对特征进行排序
        # 1、de-irrelevant：第一阶段（去无关）
        xname_removedIrrelevantFeature, proportionOfFeature = self.de_irrelevant(X[xname_sorted], y)
        print(proportionOfFeature, len(xname_removedIrrelevantFeature))
        self.de_irrelevant_RMSE = np.sqrt(mean_squared_error(y, get_10fold_cv_pls(X[xname_removedIrrelevantFeature].values, y)))
        # 2、AMB：第二阶段（去冗余）
        amb_model = AMB()
        amb_model.train(X[xname_removedIrrelevantFeature], y)
        self.feature_selected = amb_model.predict()
        self.best_RMSE = np.sqrt(mean_squared_error(y, get_10fold_cv_pls(X[self.feature_selected].values, y)))

    def predict(self):
        """
        获取被选择的特征名字
        :return:
        """
        return self.feature_selected




# 判断取几成的特征（这个步骤主要是去除无关特征）(X_sorted, y, 0.1, 0.01, 89)
def de_irrelevant(self, X_original, y, x_proportion=0.1, stepLength=0.01, cycles=89):
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

if __name__ == '__main__':
    df_x = pd.read_excel(r'C:\Users\Administrator\Desktop\data1\EndogenousSubstance.xlsx', sheet_name='Sheet1',index_col=0)  # 内源性物质
    df_y = pd.read_excel(r'C:\Users\Administrator\Desktop\data1\DrugEffectIndex.xlsx', sheet_name='Sheet1',index_col=0)
    X = df_x
    y = df_y['y1']
    Xname = X.columns.tolist()
    # ['x366', 'x512', 'x463', 'x529', 'x305', 'x465', 'x562', 'x208', 'x101', 'x10', 'x532', 'x130', 'x85', 'x172', 'x174']
    # Xname = ['x366', 'x512', 'x463', 'x529', 'x305', 'x465', 'x562', 'x208', 'x101', 'x10', 'x532', 'x130', 'x85', 'x172', 'x174']

    print(np.sqrt(mean_squared_error(y, get_10fold_cv_pls(X[Xname].values, y))))
    model = FCBF()
    model.train(X[Xname], y)
    feature_selected = model.predict()
    print(len(feature_selected), feature_selected)
    print(model.best_RMSE)



    # # 初始化森林
    # forest = model.initialize_forest_randomly(X[Xname[0:10]], y, 50, 10)
    # for i in forest:
    #     print(i.age, i.tree_vector, i.fitness_value)
    # stop_condition = 50
    # candidate_population = []
    # while stop_condition:
    #     forest = model.local_seeding(X[Xname[0:10]], y, forest, 10, 2)
    #     print(len(forest))
    #     print(get_forest_age(forest))
    #     forest, candidate_population = model.population_limiting(forest, life_time=5, area_limit=50)
    #     print(len(forest), len(candidate_population))
    #     print(get_forest_age(forest))
    #     forest, candidate_population = model.global_seeding(X[Xname[0:10]], y, forest, candidate_population, transfer_rate=0.1, GSC=3)
    #     print(len(forest), len(candidate_population))
    #     print(get_forest_age(forest))
    #     best_tree_now, forest = model.update_best_tree_now(forest)
    #     print(get_forest_age(forest))
    #     print(len(forest))
    #     print(best_tree_now.fitness_value)
    #     stop_condition -= 1
    # # 打印
    # for i in forest:
    #     print(i.age, i.tree_vector, i.fitness_value)
