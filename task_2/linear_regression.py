from matplotlib import pyplot
import numpy as np


class LinearRegression:
    def __init__(self, X: np.ndarray, y: np.ndarray, ridge_alpha: float = 0) -> None:
        """
        Initialize and fit the model.
        """
        self.ridge_alpha = ridge_alpha
        self.weights = self.__fit(X.T, y)
        #for weight in self.weights:
            #print(self.weights)


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the target for X.
        """
        # Assume single vector of explanatory variables
        return np.dot(x, self.weights)


    def __fit(self, X: np.ndarray, y=np.ndarray) -> np.ndarray:
        """
        Fit the model with ridge regularization.
        """
        inverse = np.linalg.inv(X.T @ X + self.ridge_alpha * np.identity(X.shape[1]))
        # return np.linalg.inv(X.T @ X + np.eye(X.shape[0]) * (self.ridge_alpha ** 2)) @ X.T @ y
        inv_times_XT = inverse @ X.T
        return inv_times_XT @ y

