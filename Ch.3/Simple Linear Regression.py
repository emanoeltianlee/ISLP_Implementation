import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt

from ISLP import load_data

class Simple_Regression_Model:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.number_observations = len(y)
        self.coef0 = None
        self.coef1 = None
        self.RSS = None
        self.TSS = None
        self.residuals = None
        self.SE_coef0 = None
        self.SE_coef1 = None
        self.r2 = None
        self.adjusted_r2 = None
        self.t_stat_coef0 = None
        self.t_stat_coef1 = None
        self.p_value_coef0 = None
        self.p_value_coef1 = None

        self.fitted = False

        return

    def myfit(self):
        x_mean = self.X.iloc[:, 1].mean()
        y_mean = self.y.mean()

        s1 = sum((self.X.iloc[i, 1] - x_mean) * (self.y.iloc[i] - y_mean)
                 for i in range(self.number_observations))
        s2 = sum((self.X.iloc[i, 1] - x_mean) ** 2 for i in range(self.number_observations))

        self.coef1 = s1 / s2
        self.coef0 = y_mean - self.coef1 * x_mean

        self.residuals = pd.Series([self.y.iloc[i] - self.coef0 - self.coef1 * self.X.iloc[i, 1]
                                    for i in range(self.number_observations)])
        self.RSS = sum(self.residuals ** 2)
        self.TSS = sum((self.y - y_mean) ** 2)

        residual_var = self.RSS / (self.number_observations - 2)
        self.SE_coef0 = (residual_var * (1 / self.number_observations + (x_mean ** 2) / s2)) ** 0.5
        self.SE_coef1 = (residual_var / s2) ** 0.5


        self.r2 = 1 - (self.RSS / self.TSS)
        self.adjusted_r2 = 1 - ((1 - self.r2) * (self.number_observations - 1)
                                / (self.number_observations - 3))

        self.t_stat_coef0 = self.coef0 / self.SE_coef0
        self.t_stat_coef1 = self.coef1 / self.SE_coef1

        self.p_value_coef0 = 2 * t.sf(abs(self.t_stat_coef0), self.number_observations - 2)
        self.p_value_coef1 = 2 * t.sf(abs(self.t_stat_coef1), self.number_observations - 2)

        self.fitted = True

        return

    def hypothesis_test(self):
        if not self.fitted:
            raise Exception("Model not yet fitted."
                            "Please call the 'fit' method before performing hypothesis test.")

        return


    def confidence_intervals(self, width=95):
        if not self.fitted:
            raise Exception("Model not yet fitted."
                            "Please call the 'fit' method before computing confidence intervals.")

        value = width + (100 - width) / 2
        t_value = t.ppf(value / 100, self.number_observations - 2)

        ci_coef0 = (self.coef0 - t_value * self.SE_coef0, self.coef0 + t_value * self.SE_coef0)
        ci_coef1 = (self.coef1 - t_value * self.SE_coef1, self.coef1 + t_value * self.SE_coef1)

        return ci_coef0, ci_coef1

    def plot_residuals(self):
        if not self.fitted:
            raise Exception("Model not yet fitted."
                            "Please call the 'fit' method before plotting residuals.")

        fitted_values = self.coef0 + self.coef1 * self.X.iloc[:, 1]
        plt.scatter(fitted_values, self.residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residuals vs Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.show()

    def plot_regression_line(self):
        if not self.fitted:
            raise Exception("Model not yet fitted."
                            "Please call the 'fit' method before plotting the regression line.")

        plt.scatter(self.X.iloc[:, 1], self.y, color='blue', label="Data points")

        regression_line = self.coef0 + self.coef1 * self.X.iloc[:, 1]
        plt.plot(self.X.iloc[:, 1], regression_line, color='red', label="Line of best fit")

        plt.title("Regression Line with Data Points")
        plt.xlabel("X values (Predictor)")
        plt.ylabel("Y values (Response)")
        plt.legend()
        plt.show()

        return

    def summary(self):
        if not self.fitted:
            raise Exception("Model not fitted yet. Please call the 'fit' method before summary.")

        print(f"""
        --------- Model Summary ---------
        Intercept (Coef 0):              {self.coef0:.4f}
        Slope (Coef 1):                  {self.coef1:.4f}
        Residual Sum of Squares (RSS):   {self.RSS:.4f}
        Standard Error of Intercept:     {self.SE_coef0:.4f}
        Standard Error of Slope:         {self.SE_coef1:.4f}
        R-squared:                      {self.r2:.4f}
        Adjusted R-squared:             {self.adjusted_r2:.4f}
        t-statistic for Intercept (Coef 0):   {self.t_stat_coef0:.4f}
        t-statistic for Slope (Coef 1):       {self.t_stat_coef1:.4f}
        p-value for Intercept (Coef 0):       {self.p_value_coef0:.4e}
        p-value for Slope (Coef 1):           {self.p_value_coef1:.4e}
        -----------------------------------
        """)
        return

def main():
    Boston = load_data("Boston")
    X = pd.DataFrame({'intercept': np.ones(Boston.shape[0]), 'lstat': Boston['lstat']})
    y = Boston['medv']

    model = Simple_Regression_Model(X, y)
    model.myfit()

    model.summary()

    print(model.hypothesis_test())
    print(model.confidence_intervals(95))

    model.plot_residuals()
    model.plot_regression_line()
    return


if __name__ == "__main__":
    main()