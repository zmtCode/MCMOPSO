# MCMOPSO
Hybrid mRMR and multi-objective particle swarm feature selection methods
# 多目标粒子群算法应用在代谢组学上

## 1.使用说明

### 代码运行说明

一阶段采用mRMR过滤无关和部分冗余特征并采用MIC、Pearson、Spearman三种过滤式方法进行对比，运行时可直接运行mRMR.py、MIC.py、Pearson.py、Spearman.py。第二阶段使用三种多目标特征选择算法进行对比分别为改进的CMOPSO、MOPSO、NSGA-II，其中运行文件中的MCMOPSO.py即可运行CMOPSO算法，其他的对比算法也只要运行与命名相对应的.py文件。rf.py代码是采用与对比论文中一样的计算方式求R-square的，为公平起见，采用一样的随机森林回归器以及相同的参数设置。

### 数据说明

数据一共三个文件分别为：ExogenousSubstance.xlsx、EndogenousSubstance.xlsx以及DrugEffectIndex(1).xlsx。

ExogenousSubstance.xlsx：参物注射液中的外源性物质，其中横项为特征，纵项为样本。

EndogenousSubstance.xlsx：参物注射液中的内源性物质，其中横项为特征，纵项为样本。

DrugEffectIndex(1).xlsx：药效指标，其中横项为药效指标，纵项为样本。

## 2.参数调整

本文对MOPSO中速度公式中的c1，c2以及w三个参数进行动态调整。若有其他的参数变化函数可直接更改这个三个函数。参数变化函数为：

```
c1 = c1f + (c1i-c1f) * (iter / nIter)
c2 = c2f + (c2f - c2i) * iter / nIter
w = (wStart-wEnd)*(iter/nIter-1)**2+wEnd
```

## 3.结果说明

![189089308a9c9ac7a8f16c0d1ab0080](https://github.com/zmtCode/MCMOPSO/assets/145536163/aa83936d-9eb4-4610-a813-a44db9b6c968)


这是mRMR运行结果图，选出最优个数的特征，最优RMSE为316.0866对应的特征个数是37个。然后再把最优的37个特征输出放在第二阶段去筛选最终的特征。

![39781215e559daab6c7c9d9c7255d81](https://github.com/zmtCode/MCMOPSO/assets/145536163/f2be0212-88bd-4eea-b858-4ebef021b79a)


这是三种对比过滤式运行的结果图，其中第一个数据是原始没有进行特征选择前的RMSE为3.482069，第二行数据是进行MIC筛选过滤得到的RMSE为3.26906，第三行两个数据分别是选出了特征数量的比例为排序后的前14%，一共112个特征。

![ca1f2f73d3a5530919982a96050b100](https://github.com/zmtCode/MCMOPSO/assets/145536163/9b1aa6cb-6bb8-4469-bcd7-521b185dd07d)

第二阶段运行得到的结果如图所示，其中第一个数组包含两个目标的结果，中间用.隔开，第一个是特征数量，第二个是RMSE。如上所示：得到3个特征，RMSE为152.8524。第二个数据是得到的特征具体有哪些，如图所示得到的3个特征分别为x1，x49，x58。



