import time

import numpy as np
import random

import pandas as pd
from mrmr import mrmr_regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


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

def fitness(pops,func):

    nPop = pops.shape[0]
    fits = np.array([func(pops[i]) for i in range(nPop)])
    return fits

def function(X):
    # 获取数据索引
    index = np.where(X == 1)[0]
    # 删除未选择的数据
    Xx = data.values[:, index]
    # print(Xx.shape[1])
    # if(Xx.shape[1]==1):
    #     return np.inf,np.inf
    # 特征数量
    num = np.sum(X == 1)
    f1 = num/len(X)
    if num == 0 or num == 1 or num == 2:
        return np.inf, np.inf
    # 初始化变量
    pr = get_10fold_cv_pls(Xx, data_label)
    RMSE = np.sqrt(mean_squared_error(data_label, pr))
    f2 = RMSE
    return [f1, f2]


def initPops(nPop, nChr, lb, rb):
    pops = np.zeros((nPop, nChr))
    for i in range(nPop):
        for j in range(nChr):
            pops[i, j] = random.randint(0, 1)
    return pops

def select1(pool, pops, fits, ranks, distances):
    # 一对一锦标赛选择
    # pool: 新生成的种群大小
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((pool, nChr))
    newFits = np.zeros((pool, nF))

    indices = np.arange(nPop).tolist()
    i = 0
    while i < pool:
        idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体
        idx = compare(idx1, idx2, ranks, distances)
        newPops[i] = pops[idx]
        newFits[i] = fits[idx]
        i += 1
    return newPops, newFits


def compare(idx1, idx2, ranks, distances):
    # return: 更优的 idx
    if ranks[idx1] < ranks[idx2]:
        idx = idx1
    elif ranks[idx1] > ranks[idx2]:
        idx = idx2
    else:
        if distances[idx1] <= distances[idx2]:
            idx = idx2
        else:
            idx = idx1
    return idx

def crossover(pops, pc, etaC, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构
    chrPops = pops.copy()
    nPop = chrPops.shape[0]
    for i in range(0, nPop, 2):
        if np.random.rand() < pc:
            SBX(chrPops[i], chrPops[i + 1], etaC, lb, rb)  # 交叉
    return chrPops


def SBX(chr1, chr2, etaC, lb, rb):
    # 模拟二进制交叉
    pos1, pos2 = np.sort(np.random.randint(0, len(chr1), 2))
    pos2 += 1
    u = np.random.rand()
    if u <= 0.5:
        gamma = (2 * u) ** (1 / (etaC + 1))
    else:
        gamma = (1 / (2 * (1 - u))) ** (1 / (etaC + 1))
    x1 = chr1[pos1:pos2]
    x2 = chr2[pos1:pos2]
    chr1[pos1:pos2], chr2[pos1:pos2] = 0.5 * ((1 + gamma) * x1 + (1 - gamma) * x2), \
                                       0.5 * ((1 - gamma) * x1 + (1 + gamma) * x2)
    # 检查是否符合约束
    chr1[chr1 < lb] = lb
    chr1[chr1 > rb] = rb
    chr2[chr2 < lb] = lb
    chr2[chr2 < rb] = rb

def nonDominationSort(pops, fits):
    """快速非支配排序算法
    Params:
        pops: 种群，nPop * nChr 数组
        fits: 适应度， nPop * nF 数组
    Return:
        ranks: 每个个体所对应的等级，一维数组
    """
    nPop = pops.shape[0]
    nF = fits.shape[1]  # 目标函数的个数
    ranks = np.zeros(nPop, dtype=np.int32)
    nPs = np.zeros(nPop)  # 每个个体p被支配解的个数
    sPs = []  # 每个个体支配的解的集合，把索引放进去
    for i in range(nPop):
        iSet = []  # 解i的支配解集
        for j in range(nPop):
            if i == j:
                continue
            isDom1 = fits[i] <= fits[j]
            isDom2 = fits[i] < fits[j]
            # 是否支配该解-> i支配j
            if sum(isDom1) == nF and sum(isDom2) >= 1:
                iSet.append(j)
                # 是否被支配-> i被j支配
            if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                nPs[i] += 1
        sPs.append(iSet)  # 添加i支配的解的索引
    r = 0  # 当前等级为 0， 等级越低越好
    indices = np.arange(nPop)
    while sum(nPs == 0) != 0:
        rIdices = indices[nPs == 0]  # 当前被支配数为0的索引
        ranks[rIdices] = r
        for rIdx in rIdices:
            iSet = sPs[rIdx]
            nPs[iSet] -= 1
        nPs[rIdices] = -1  # 当前等级的被支配数设置为负数
        r += 1
    return ranks


# 拥挤度排序算法
def crowdingDistanceSort(pops, fits, ranks):
    """拥挤度排序算法
    Params:
        pops: 种群，nPop * nChr 数组
        fits: 适应度， nPop * nF 数组
        ranks：每个个体对应的等级，一维数组
    Return：
        dis: 每个个体的拥挤度，一维数组
    """
    nPop = pops.shape[0]
    nF = fits.shape[1]  # 目标个数
    dis = np.zeros(nPop)
    nR = ranks.max()  # 最大等级
    indices = np.arange(nPop)
    for r in range(nR + 1):
        rIdices = indices[ranks == r]  # 当前等级种群的索引
        rPops = pops[ranks == r]  # 当前等级的种群
        rFits = fits[ranks == r]  # 当前等级种群的适应度
        rSortIdices = np.argsort(rFits, axis=0)  # 对纵向排序的索引
        rSortFits = np.sort(rFits, axis=0)
        fMax = np.max(rFits, axis=0)
        fMin = np.min(rFits, axis=0)
        n = len(rIdices)
        for i in range(nF):
            orIdices = rIdices[rSortIdices[:, i]]  # 当前操作元素的原始位置
            j = 1
            while n > 2 and j < n - 1:
                if fMax[i] != fMin[i]:
                    dis[orIdices[j]] += (rSortFits[j + 1, i] - rSortFits[j - 1, i]) / \
                                        (fMax[i] - fMin[i])
                else:
                    dis[orIdices[j]] = np.inf
                j += 1
            dis[orIdices[0]] = np.inf
            dis[orIdices[n - 1]] = np.inf
    return dis

def optSelect(pops, fits, chrPops, chrFits):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)

    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    while i + len(rIndices) <= nPop:
        newPops[i:i + len(rIndices)] = MergePops[rIndices]
        newFits[i:i + len(rIndices)] = MergeFits[rIndices]
        r += 1  # 当前等级+1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引

    if i < nPop:
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = MergePops[surIndices]
        newFits[i:] = MergeFits[surIndices]
    return (newPops, newFits)

def mutate(pops, pm, etaM, lb, rb):
    nPop = pops.shape[0]
    for i in range(nPop):
        if np.random.rand() < pm:
            polyMutation(pops[i], etaM, lb, rb)
    return pops

def polyMutation(chr, etaM, lb, rb):
    # 多项式变异
    pos1, pos2 = np.sort(np.random.randint(0,len(chr),2))
    pos2 += 1
    u = np.random.rand()
    if u < 0.5:
        delta = (2*u) ** (1/(etaM+1)) - 1
    else:
        delta = 1-(2*(1-u)) ** (1/(etaM+1))
    chr[pos1:pos2] += delta
    chr[chr < lb] = lb
    chr[chr > rb] = rb

def NSGA2(nIter, nChr, nPop, pc, pm, etaC, etaM, func, lb, rb):
    """非支配遗传算法主程序
    Params:
        nIter: 迭代次数
        nPop: 种群大小
        pc: 交叉概率
        pm: 变异概率
        func: 优化的函数
        lb: 自变量下界
        rb: 自变量上界
    """
    # 生成初始种群
    pops = initPops(nPop, nChr, lb, rb)
    fits = fitness(pops, func)

    # 开始第1次迭代
    iter = 1
    while iter <= nIter:
        # 进度条
        ranks = nonDominationSort(pops, fits)  # 非支配排序
        distances = crowdingDistanceSort(pops, fits, ranks)  # 拥挤度
        pops, fits = select1(nPop, pops, fits, ranks, distances)
        chrpops = crossover(pops, pc, etaC, lb, rb)  # 交叉产生子种群
        chrpops = mutate(chrpops, pm, etaM, lb, rb)  # 变异产生子种群
        chrfits = fitness(chrpops, func)
        # 从原始种群和子种群中筛选
        pops, fits = optSelect(pops, fits, chrpops, chrfits)
        iter += 1
        # 对最后一代进行非支配排序
    ranks = nonDominationSort(pops, fits)  # 非支配排序
    distances = crowdingDistanceSort(pops, fits, ranks)  # 拥挤度
    paretoPops = pops[ranks == 0]
    paretoFits = fits[ranks == 0]
    return paretoPops, paretoFits



def ga():
    nIter = 300
    nChr = 30
    nPop = 100
    pc = 0.6
    pm = 0.1
    etaC = 1
    etaM = 1
    lb = 0
    rb = 1
    func = function
    paretoPops, paretoFits = NSGA2(nIter, nChr, nPop, pc, pm, etaC, etaM, func, lb, rb)
    return paretoPops, paretoFits

def get_index(paretoPops, c):
    ans = []
    for item in paretoPops:
        tmp = []
        for k in range(len(item)):
            if item[k] == 1:
                tmp.append(c[k])
        ans.append(tmp)
    return ans

if __name__ == "__main__":

    # df_x = pd.read_excel(r'C:\Users\Administrator\Desktop\data1\EndogenousSubstance.xlsx', sheet_name='Sheet1',index_col=0)  # 内源性物质
    # df_y = pd.read_excel(r'C:\Users\Administrator\Desktop\data1\DrugEffectIndex.xlsx', sheet_name='Sheet1',index_col=0)
    # df_x = pd.read_excel(r'C:\Users\Administrator\Desktop\paper2回归任务\data\blogData_test\blogData_test1.xlsx',index_col=0)  # 内源性物质
    df_x = pd.read_excel(r'C:\Users\Administrator\Desktop\paper2回归任务\data\data big sample\Student Performance Data Set（数学）.xlsx',index_col=0)  # 内源性物质
    # X = df_x
    # # print(X)
    # y = df_y['y1']
    X = df_x.iloc[:,:-3]
    # print(X)
    y = df_x['y1']
    # print(y)
    # 1.mrmr
    selected_features = ['x6010', 'x4772', 'x6925', 'x630', 'x2104', 'x2671', 'x6017', 'x1279', 'x2078', 'x6621', 'x1247', 'x1037', 'x9399', 'x4633', 'x5472', 'x2684', 'x3695', 'x1768', 'x1251', 'x3589', 'x6162', 'x1011', 'x6859', 'x4752', 'x1166', 'x1065', 'x3635', 'x6714', 'x3593', 'x6019', 'x1072', 'x8084', 'x2658', 'x2286', 'x7634', 'x1237', 'x1075', 'x2101', 'x4632', 'x1059', 'x6021', 'x2600', 'x1904', 'x1038', 'x6018', 'x1299', 'x6709', 'x1281', 'x760', 'x6693', 'x1047', 'x4583', 'x2282', 'x5390', 'x928', 'x6011', 'x3850', 'x6350', 'x2063', 'x7834', 'x1230', 'x7387', 'x2108', 'x2284', 'x1074', 'x3636', 'x8074', 'x1034', 'x7640', 'x1071', 'x3864', 'x932', 'x6713', 'x1112', 'x3849', 'x1110', 'x4662', 'x1213', 'x5819', 'x1076', 'x6744', 'x2283', 'x1309', 'x2710', 'x6743', 'x5433', 'x1104', 'x2487', 'x2670', 'x7573', 'x4308', 'x1211', 'x6691', 'x1039', 'x1054', 'x2086', 'x4806', 'x664', 'x3856', 'x7045', 'x3017', 'x1032']
    # data = X[selected_features]
    data = X
    Xname = X.columns.tolist()

    # print(data)
    data_label = y.values
    action = time.time()
    print(action)
    # print(data_label)
    paretoPops, paretoFits = ga()
    print(np.unique(paretoFits,axis=0))
    #
    FS = get_index(paretoPops,Xname)
    # R2 = r2_score(y, get_10fold_cv_pls(X[FS[1]].values, y))
    # print('两阶段的R：', R2)
    FS = list(set([tuple(t) for t in FS]))
    FS = [list(s) for s in FS]
    FS = sorted(FS, key=lambda x: len(x))
    print(FS)
    #
    #
    end = time.time()
    print('时间', end - action)