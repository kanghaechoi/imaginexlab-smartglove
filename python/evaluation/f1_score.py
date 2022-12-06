import numpy as np


class F1Score:
    def __init__(self, matrix: np.ndarray) -> None:
        self.true_negative = matrix[0, 0]
        self.false_positive = matrix[0, 1]
        self.false_negative = matrix[1, 0]
        self.true_positive = matrix[1, 1]

        self.number_of_samples = np.sum(matrix)

    def accuracy(self):
        _accuracy = (
            (self.true_negative + self.true_positive) / self.number_of_samples
        ) * 100

        return _accuracy

    def recall(self):
        _recall = (
            self.true_positive / (self.true_positive + self.false_negative)
        ) * 100

        return _recall

    def precision(self):
        _precision = (
            self.true_positive / (self.true_positive + self.false_positive)
        ) * 100

        return _precision

    def f1_score(self):
        _f1_score = (
            (2 * self.precision + self.recall) / (self.precision + self.recall) * 100
        )

        return _f1_score


class F1Score3D(F1Score):
    def __init__(self, matrix: np.ndarray) -> None:
        super().__init__(matrix)
