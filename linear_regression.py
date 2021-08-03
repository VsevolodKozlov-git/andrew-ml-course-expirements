import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, X, Y, pipeline, lr):
        if (lr < 0):
            raise Exception()
        self.X = X
        self.Y = Y
        self.pipeline = pipeline
        self.lr = lr
        self.m = X.shape[0]
        self.transformed_X = pipeline(X)
        self.parameters = np.random.rand(pipeline(X).shape[1])[:, np.newaxis]

    def learn_and_plot_error(self, iterations, lr=None, axis=None):
        if lr is None:
            lr = self.lr
        if axis is None:
            axis = plt.gca()
        errors_arr = np.zeros(iterations)
        self.change_lr(lr)
        for iteration in range(iterations):
            self.update_parameters()
            errors_arr[iteration] = self.error()

        axis.plot(range(1, iterations + 1), errors_arr)
        axis.set_xlabel('№ of iteration')
        axis.set_ylabel('Error')

    def update_parameters(self):
        transformed_X = self.pipeline(self.X)
        self.parameters -= self.lr * self.get_partial_derivatives()

    def get_partial_derivatives(self):
        return 1 / self.m * \
               np.sum((self.predict_with_transformation(self.X) - self.Y) * self.transformed_X, axis=0)[:, np.newaxis]

    def predict_with_transformation(self, X):
        transfotmed_X = self.pipeline(X)
        return self.predict(transfotmed_X)

    def predict(self, X):
        return np.matmul(X, self.parameters)

    def plot(self, x_start, x_end, axis=None):
        if axis is None:
            axis = plt.gca()
        X = np.linspace(x_start, x_end, 100)[:, np.newaxis]
        Y = self.predict_with_transformation(X)
        axis.scatter(self.X, self.Y)
        axis.plot(X, Y)
        axis.set_xlim(x_start, x_end)

    def error(self):
        return 1 / (2 * self.m) * np.sum((self.predict_with_transformation(self.X) - Y) ** 2)

    def change_lr(self, new_lr):
        if (new_lr <= 0):
            raise Exception('lr must be positive number')
        self.lr = new_lr

    def __str__(self):
        linear_regression_info = \
            f'''
    parameters: {self.parameters}
    partial_derivatives: {self.get_partial_derivatives()}
    error: {self.error()}
    learning rate: {self.lr}
    '''

        return linear_regression_info
