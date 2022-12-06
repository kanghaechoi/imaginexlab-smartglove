import numpy as np
import tensorflow as tf


class Tensor:
    def __init__(self) -> None:
        pass

    def array_to_tensor(self, array: np.ndarray) -> tf.Tensor:
        tensor = tf.constant(array)

        return tensor
