import numpy as np
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import summarize

Boston = load_data("Boston")
y1 = Boston['medv']
X1 = pd.DataFrame({'lstat': Boston['lstat'], 'age': Boston['age']})

model1 = sm.OLS(y1, sm.add_constant(X1))
results1 = model1.fit()
print(summarize(results1))

class Multiple_Regression_Model:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.predictor_means = self.X.mean(axis=0)
        self.output_mean = self.y.mean()
        self.data_matrix = np.hstack([np.ones((X.shape[0], 1)), X.to_numpy()])
        self.output_vector = y.to_numpy()
        self.number_observations = len(y)
        self.number_features = self.data_matrix.shape[1]
        self.df = self.number_observations - self.number_features - 1
        self.coefficients = None
        self.residuals = None
        self.RSS = None
        self.residual_variance = None
        self.SE_coefficients = None
        self.r2 = None
        self.adjusted_r2 = None
        self.t_stat_coefficients = None
        self.p_value_coefficients = None
        self.fitted = False

    def fit(self):
        inverse = np.linalg.inv(self.data_matrix.T @ self.data_matrix)
        self.coefficients = inverse @ (self.data_matrix.T @ self.output_vector)
        self.residuals = self.output_vector - self.data_matrix @ self.coefficients
        self.RSS = sum(self.residuals ** 2)
        self.TSS = sum((self.y - self.output_mean) ** 2)
        self.residual_variance = self.RSS / (self.df + 1)
        self.SE_coefficients = np.sqrt(np.diag(self.residual_variance * inverse))
        self.r2 = 1 - (self.RSS / self.TSS)
        self.adjusted_r2 = 1 - (1 - self.r2) * (self.number_observations - 1) / self.df
        self.t_stat_coefficients = self.coefficients / self.SE_coefficients
        self.p_value_coefficients = 2 * (1 - t.cdf(abs(self.t_stat_coefficients), self.df))
        self.fitted = True

    def predict(self, x: pd.Series):
        return np.dot([1] + list(x), self.coefficients)

model = Multiple_Regression_Model(X1, y1)
model.fit()
print("Coefficients:", model.coefficients)
print("T-Statistics:", model.t_stat_coefficients)
print("P-Values:", model.p_value_coefficients)

def covariance_matrix(X):
    n_features = X.shape[1]
    cov_matrix = np.zeros((n_features, n_features))
    means = np.mean(X, axis=0)

    for i in range(n_features):
        for j in range(n_features):
            cov_matrix[i, j] = np.sum((X[:, i] - means[i]) * (X[:, j] - means[j])) / (X.shape[0] - 1)

    return cov_matrix

def correlation_matrix(X):
    cov_matrix = covariance_matrix(X)
    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

    return corr_matrix

print("Covariance Matrix:\n", covariance_matrix(X1))
print("Correlation Matrix:\n", correlation_matrix(X1))