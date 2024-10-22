import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib 
from matplotlib.pyplot import subplots
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize , poly)

Boston = load_data("Boston")
X1 = pd.DataFrame({'intercept': np.ones(Boston.shape[0]), 'lstat': Boston['lstat']})

def OLS_data(y: pd.Series, X: pd.DataFrame):
    n = len(y)
    x_mean = X.iloc[:,1].mean()
    y_mean = y.mean()
    s1 = sum((X.iloc[i, 1] - x_mean)*(y.iloc[i] - y_mean) for i in range(n))
    s2 = sum((X.iloc[i, 1] - x_mean)**2 for i in range(n))
    beta1 = s1/s2
    beta0 = y_mean - beta1*x_mean

    residuals = pd.Series([y.iloc[i] - beta0 - beta1*X.iloc[i, 1] for i in range(n)])
    RSS = sum(residuals**2)

    residual_var = RSS/(n - 2)

    SE_beta0 = (residual_var*(1/n + ((x_mean)**2)/s2))**0.5
    SE_beta1 = (residual_var/s2)**0.5
    return beta0, beta1, RSS, SE_beta0, SE_beta1, n

#Standard Errors: these tell us by how much beta0 and beta1 differ from the actual value

#Confidence Intervals
def CI_of_coeffs(width = 95, y: pd.Series, X: pd.DataFrame):
    value = width + (100 - width)/2
    print(f"{width}% CI for beta0:",(OLS_data(y,X)[0] - t.ppf(value/ 100, len(y)-2)*OLS_data(y,X)[3],
                                     OLS_data(y,X)[0] + t.ppf(value/ 100, len(y)-2)*OLS_data(y,X)[3]))
    print(f"{width}% CI for beta1:", (OLS_data(y,X)[1] - t.ppf(value/ 100, len(y)-2)*OLS_data(y,X)[4],
                                      OLS_data(y,X)[1] + t.ppf(value/ 100, len(y)-2)*OLS_data(y,X)[4]))
    return

#Hypothesis Testing
def hyp_test(y: pd.Series, X: pd.DataFrame):
    t_stat_beta0 = OLS_data(y, X)[0] / OLS_data(y, X)[3]
    t_stat_beta1 = OLS_data(y, X)[1] / OLS_data(y, X)[4]
    p_value_beta0 = 2 * t.sf(abs(t_stat_beta0), len(y))
    p_value_beta1 = 2 * t.sf(abs(t_stat_beta1), len(y))
    return t_stat_beta0, t_stat_beta1, p_value_beta0 , p_value_beta1

def regression(y: pd.Series, X: pd.DataFrame):
    print('Residual Sum of Squares (RSS):', OLS_data(y, X)[2])
    print('Intercept:', OLS_data(y, X)[1])
    print('Slope:', OLS_data(y, X)[0])
    print('Standard Error in Intercept:', OLS_data(y, X)[3])
    print('Standard Error in Slope:', OLS_data(y, X)[4])
    print('t-stat in Intercept:', hyp_test(y, X)[0])
    print('t-stat in Slope:', hyp_test(y, X)[1])
    print('P-Value in Intercept:', hyp_test(y, X)[2])
    print('P-Value in Slope:', hyp_test(y, X)[3])
    return

regression(y1,X1)
CI_of_coeffs(95, y1,X1)
y1 = Boston['medv']
model = sm.OLS(y1, X1)
results = model.fit()

print(summarize(results))