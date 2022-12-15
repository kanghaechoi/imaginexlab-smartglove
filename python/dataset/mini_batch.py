from typing import Tuple

import numpy as np


class MiniBatch:
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        number_of_batches: int,
        batch_size: int,
    ) -> None:
        self.data = data
        self.labels = labels
        self.number_of_batches = number_of_batches
        self.batch_size = batch_size

    def create_mini_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        number_of_data = self.data.shape[0]

        mini_batch_index = np.random.choice(number_of_data, self.batch_size)

        mini_data = self.data[mini_batch_index]
        mini_labels = self.labels[mini_batch_index]

        return mini_data, mini_labels

    def get_mini_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        first_mini_data, first_mini_labels = self.create_mini_dataset()
        second_mini_data, second_mini_labels = self.create_mini_dataset()

        mini_batch_data = np.stack((first_mini_data, second_mini_data))
        mini_batch_labels = np.stack((first_mini_labels, second_mini_labels))

        for _ in range(self.number_of_batches - 2):
            other_mini_data, other_mini_labels = self.create_mini_dataset()

            other_mini_data = np.expand_dims(other_mini_data, axis=0)
            other_mini_labels = np.expand_dims(other_mini_labels, axis=0)

            mini_batch_data = np.append(mini_batch_data, other_mini_data, axis=0)
            mini_batch_labels = np.append(mini_batch_labels, other_mini_labels, axis=0)

        return mini_batch_data, mini_batch_labels
