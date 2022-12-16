import numpy as np
import tensorflow as tf


class AsTensor:
    def __init__(self) -> None:
        pass

    def array_to_tensor(self, array: np.ndarray) -> tf.Tensor:
        tensor = tf.Variable(array)

        return tensor
