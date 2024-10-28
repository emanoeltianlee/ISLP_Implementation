import numpy as np
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import summarize

Boston = load_data("Boston")
y1 = Boston['medv']
X1 = pd.DataFrame({'lstat': Boston['lstat'], 'age': Boston['age']})

class KNNRegressor:
    def __init__(self, X: pd.DataFrame, y: pd.Series, k: int,
                 distance_metric='euclidean', p: int = 2, weights='uniform'):
        self.X = X
        self.y = y
        self.k = k
        self.distance_metric = distance_metric
        self.p = p
        self.weights = weights

        self.predictor_means = self.X.mean(axis=0)
        self.output_mean = self.y.mean()
        self.number_observations = len(y)
        self.number_features = self.X.shape[1]
        self.df = self.number_observations - self.number_features - 1

        self.coefficients = None
        self.residuals_values = None
        self.RSS = None
        self.residual_variance = None
        self.SE_coefficients = None
        self.r2 = None
        self.adjusted_r2 = None
        self.t_stat_coefficients = None
        self.p_value_coefficients = None

        self.fitted = False

    def _calculate_distances(self) -> np.ndarray:
        if self.distance_metric == 'euclidean':
            distances = np.sum((self.X.to_numpy()[:, np.newaxis, :] -
                                self.X.to_numpy()[np.newaxis, :, :]) ** 2, axis=2) ** 0.5
        elif self.distance_metric == 'manhattan':
            distances = np.sum(np.abs(self.X.to_numpy()[:, np.newaxis, :] -
                                      self.X.to_numpy()[np.newaxis, :, :]), axis=2)
        elif self.distance_metric == 'minkowski':
            distances = np.sum(np.abs(self.X.to_numpy()[:, np.newaxis, :] -
                                      self.X.to_numpy()[np.newaxis, :, :]) ** self.p, axis=2) ** (1 / self.p)
        else:
            raise ValueError(f"Unsupported metric: {self.distance_metric}")
        return distances

    def fit(self) -> pd.Series:
        distances = self._calculate_distances()
        predictions = []

        for i in range(distances.shape[0]):
            nearest_indices = np.argsort(distances[i])[1:self.k + 1]
            nearest_outputs = self.y.iloc[nearest_indices].tolist()
            nearest_distances = distances[i][nearest_indices]
            prediction = self._weighted_average(nearest_outputs, nearest_distances)
            predictions.append(prediction)

        predictions = pd.Series(predictions)

        self.residuals_values = self.y - predictions
        self.RSS = np.sum(self.residuals_values ** 2)

        self.residual_variance = self.RSS / self.df

        total_sum_of_squares = np.sum((self.y - self.output_mean) ** 2)
        self.r2 = 1 - (self.RSS / total_sum_of_squares)
        self.adjusted_r2 = 1 - (1 - self.r2) * (self.number_observations - 1) / (self.df)

        self.fitted = True

        return predictions

    def residuals(self) -> pd.Series:
        if not self.fitted:
            raise ValueError("Model not yet fitted")
        return self.residuals_values

    def predict(self, x):
        if isinstance(x, pd.Series):
            x = x.values

        distances = self.X.apply(lambda row: self._calculate_distance(row.values, x), axis=1)
        nearest_indices = distances.nsmallest(self.k).index
        nearest_outputs = self.y.iloc[nearest_indices]
        nearest_distances = distances.iloc[nearest_indices]

        return self._weighted_average(nearest_outputs.tolist(), nearest_distances.tolist())

    def _weighted_average(self, neighbors: list, distances: list) -> float:
        if self.weights == 'uniform':
            return sum(neighbors) / self.k
        elif self.weights == 'distance':
            weights = np.array([1 / d if d != 0 else 0 for d in distances])
            if np.any(weights == 0):
                weights = np.where(weights == 0, 1, 0)
            weights = weights / np.sum(weights)
            return np.dot(weights, neighbors)

    def prediction_data(self, x):
        if isinstance(x, pd.Series):
            x = x.values

        distances = self.X.apply(lambda row: self._calculate_distance(row.values, x), axis=1)
        nearest_indices = distances.nsmallest(self.k).index
        nearest_outputs = self.y.iloc[nearest_indices]
        nearest_distances = distances.iloc[nearest_indices]

        return self._weighted_average(nearest_outputs.tolist(),
                                      nearest_distances.tolist()), nearest_distances, nearest_outputs

    def _calculate_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((vec1 - vec2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(vec1 - vec2))
        elif self.distance_metric == 'minkowski':
            return np.sum(np.abs(vec1 - vec2) ** self.p) ** (1 / self.p)
        else:
            raise ValueError(f"Unsupported metric: {self.distance_metric}")

knn_regressor = KNNRegressor(X1, y1, k=5)

predictions = knn_regressor.fit()
#Examples and Comparison
a = knn_regressor.predict([33,5])
residuals = knn_regressor.residuals()
print(a)
print(predictions)

model1 = sm.OLS(y1, sm.add_constant(X1))
results1 = model1.fit()
print(summarize(results1))
