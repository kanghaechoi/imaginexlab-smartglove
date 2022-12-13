import math
import random
from typing import Tuple


import numpy as np


class Divide:
    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.data = data
        self.labels = labels

        self.training_data_index = None
        self.test_data_index = None

    def fit(self, test_dataset_ratio: float = 0.2) -> None:
        number_of_data = self.data.shape[0]

        data_index = np.linspace(
            0,
            number_of_data,
            number_of_data,
            endpoint=False,
            dtype=int,
        )

        number_of_test_dataset = math.floor(number_of_data * test_dataset_ratio)

        self.test_data_index = np.array(
            random.sample(data_index.tolist(), number_of_test_dataset)
        )

        self.training_data_index = np.delete(
            data_index, self.test_data_index, axis=None
        )

        return 0

    def training_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        training_data = self.data[self.training_data_index, :, :]
        training_labels = self.labels[self.training_data_index, :]

        return training_data, training_labels

    def test_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        test_data = self.data[self.test_data_index, :, :]
        test_labels = self.labels[self.test_data_index, :]

        return test_data, test_labels
