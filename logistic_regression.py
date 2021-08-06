from linear_regression import  LinearRegression
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression(LinearRegression):

    def predict(self, X):
        return LogisticRegression.sigmoid(super().predict(X))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.e ** (-z))

    def error(self):
        cost =  -np.sum((self.Y * np.log2(self.predict_with_transformation(self.X))
                 + (1 - self.Y) * np.log2(1 - self.predict_with_transformation(self.X)) ))
        return (1 / self.m) * cost

    def plot(self, x_range: list, y_range:list, axis: plt.Axes=None):
        # working with axis and fig
        if axis is None:
            axis = plt.gca()
        fig = plt.gcf()
        # preparation work
        linspace_values = 100
        x = np.linspace(*x_range, num=linspace_values)
        y = np.linspace(*y_range, num=linspace_values)
        xx, yy =np.meshgrid(x, y)
        X = np.hstack([np.ravel(xx)[:, np.newaxis], np.ravel(yy)[:, np.newaxis]])
        predicted_values = self.predict_with_transformation(X).reshape(linspace_values, linspace_values)
        # plotting
        cs = axis.contourf(xx, yy, predicted_values, 20)
        fig.colorbar(cs )