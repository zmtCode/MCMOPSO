# Hybrid mRMR and multi-objective particle swarm feature selection methods and metabolomics applications

## 1. Instructions for use

### Code Run Description

In the first stage, mRMR is used to filter irrelevant and partially redundant features and three filtered methods, MIC, Pearson and Spearman, are used for comparison, which can be run directly by running mRMR.py, MIC.py, Pearson.py and Spearman.py. In the second stage, three multi-objective feature selection algorithms are used for comparison respectively the improved CMOPSO, MOPSO, and NSGA-II, where the CMOPSO algorithm can be run by running MCMOPSO.py in the paper, and the other comparison algorithms just run the .py file corresponding to the naming. rf.py code is used to find the R-square using the same computation as in the comparison paper, and for fairness, the same random forest regressor is used and the same parameter settings.

### Data description

There are a total of three files of data are: ExogenousSubstance.xlsx、EndogenousSubstance.xlsx以及DrugEffectIndex(1).xlsx。

ExogenousSubstance.xlsx：Exogenous substances in ginseng injections, where the horizontal items are characteristics and the vertical items are samples.

EndogenousSubstance.xlsx：Endogenous substances in ginseng injections, where the horizontal items are characteristics and the vertical items are samples.

DrugEffectIndex(1).xlsx：Pharmacodynamic indicators, where the horizontal term is the pharmacodynamic indicator and the vertical term is the sample.

## 2. Parameter adjustment

This paper dynamically adjusts the three parameters c1, c2 and w in the velocity equation in MOPSO. If there are other parameter change functions, these three functions can be changed directly. The parameter change function is:

```
c1 = c1f + (c1i-c1f) * (iter / nIter)
c2 = c2f + (c2f - c2i) * iter / nIter
w = (wStart-wEnd)*(iter/nIter-1)**2+wEnd
```

## 3. Description of results

![1e0d2bfe117a76be96a8bab56fdc5f1](https://github.com/zmtCode/MCMOPSO1/assets/145536163/3d341a9b-eb14-4e25-976b-c5efdeeb33fc)



This is the result graph of the mRMR run to select the optimal number of features, the optimal RMSE is 316.0866 corresponding to 37 features. Then the optimal 37 feature outputs are put in the second stage to screen the final features.

![5f4581e8ac829448ebd9acdd51ab64e](https://github.com/zmtCode/MCMOPSO1/assets/145536163/8fe72fc6-a8b4-44bd-9cbc-41f224ab2359)



This is a graph of the results of the three comparative filtering runs, where the first data is the RMSE of 3.482069 before the original no feature selection, the second row of data is the RMSE of 3.26906 obtained by performing the MIC screening and filtering, and the two third rows of data are the proportion of the number of features selected to be in the top 14% of the sorted list, for a total of 112 features, respectively.

![e347369a62d7e3527d023136baea65f](https://github.com/zmtCode/MCMOPSO1/assets/145536163/7da7cd81-11a1-4c5c-ae6f-71686dcd9809)



The results obtained from the second stage of the run are shown in the figure, where the first array contains the results of the two objectives, separated by . The first is the number of features and the second is the RMSE. as shown above: 3 features were obtained and the RMSE is 152.8524. the second data is what are the specific features obtained, as shown in the figure the 3 features obtained are x1, x49, x58.



