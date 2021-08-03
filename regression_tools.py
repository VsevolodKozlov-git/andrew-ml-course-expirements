import numpy as np


class PipeLine:
    def __init__(self):
        self.pipeline = []

    def add(self, func):
        self.pipeline.append(func)

    def __call__(self, values_to_tranform):
        for transformation_function in self.pipeline:
            values_to_tranform = transformation_function(values_to_tranform)
        return values_to_tranform


def n_degree_polynomial(n, fit_intercept=True):
    def transfromtaion_function(X):
        if fit_intercept:
            transformed_X = np.ones(X.shape[0])[:, np.newaxis]
        else:
            transformed_X = X
        for power in range(1, n + 1):
            if not fit_intercept and power == 1:
                continue
            transformed_X = np.hstack([transformed_X, X ** power])
        return transformed_X

    return transfromtaion_function


def scale_data(X):
    max_column_value = np.max(X, axis=0)
    max_column_value[max_column_value == 0] = 1
    return X / np.max(X, axis=0)


def normalize_data(X):
    std = np.std(X)
    mean = np.mean(X)
    return (X - mean) / std