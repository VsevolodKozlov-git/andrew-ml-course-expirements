from abc import ABC, abstractmethod
import numpy as np


class MlModel(ABC):
    @abstractmethod
    def get_error_and_grad(self, theta, X, Y):
        pass

    @abstractmethod
    def get_error(self, X, Y, theta):
        pass

    @abstractmethod
    def get_grad(self, X, Y, theta):
        """
    :return: [Y.size] gradient of given parameters
    """
        pass

    @abstractmethod
    def train_theta(self, X, Y, current_theta, lr):
        pass

    @abstractmethod
    def predict(self, X, theta):
        pass


class LinearRegressionCore(MlModel):
    def get_error_and_grad(self, theta, X, Y):
        m = Y.size  # number of training examples
        h = self.predict(X, theta)
        error = self.get_error(X, Y, theta)
        grad = self.get_grad(X, Y, theta)
        return error, grad

    def get_error(self, X, Y, theta):
        m = Y.size
        h = self.predict(X, theta)
        return (1 / (2 * m)) * np.sum((h - Y) ** 2)

    def get_grad(self, X, Y, theta):
        """
    :return: [1 x Y.size] gradient of given parameters
    """
        m = Y.size
        h = self.predict(X, theta)
        return (1 / m) * np.sum((h - Y) * X, axis=0)

    def train_theta(self, X, Y, current_theta, lr):
        trained_theta = current_theta - lr * self.get_grad(X, Y, current_theta)
        return trained_theta

    def predict(self, X, theta):
        return np.dot(X, theta)[:, np.newaxis]
