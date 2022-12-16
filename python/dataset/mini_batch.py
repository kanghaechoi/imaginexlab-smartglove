from typing import Tuple

import tensorflow as tf
import numpy as np


class MiniBatch:
    def __init__(
        self,
        data: tf.Tensor,
        labels: tf.Tensor,
        batch_size: int,
    ) -> None:
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.number_of_mini_batches = tf.math.ceil(data.shape[0] / batch_size)

    def create_small_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        number_of_data = self.data.shape[0]

        breakpoint()
        shuffle_index = tf.random.categorical(
            tf.math.log([[0.5, 0.5]]),
            number_of_data,
        )
        breakpoint()
        shuffle_index = np.random.choice(number_of_data, self.batch_size)

        small_data = self.data[shuffle_index]
        small_labels = self.labels[shuffle_index]

        return small_data, small_labels

    def get_mini_batch_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        first_small_data, first_small_labels = self.create_small_dataset()
        second_small_data, second_small_labels = self.create_small_dataset()

        mini_batch_data = np.stack((first_small_data, second_small_data))
        mini_batch_labels = np.stack((first_small_labels, second_small_labels))

        for _ in range(self.number_of_mini_batches - 2):
            other_mini_data, other_mini_labels = self.create_small_dataset()

            other_mini_data = np.expand_dims(other_mini_data, axis=0)
            other_mini_labels = np.expand_dims(other_mini_labels, axis=0)

            mini_batch_data = np.append(mini_batch_data, other_mini_data, axis=0)
            mini_batch_labels = np.append(mini_batch_labels, other_mini_labels, axis=0)

        return mini_batch_data, mini_batch_labels
