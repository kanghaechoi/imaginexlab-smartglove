import tensorflow as tf


class _BottleneckBase:
    def __init__(self, _filters: int, _strides: int = 1):
        super(_BottleneckBase, self).__init__()

        self.convolution_layer_type1 = tf.keras.layers.Conv2D(
            filters=_filters,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
        )
        self.convolution_layer_type2 = tf.keras.layers.Conv2D(
            filters=_filters,
            kernel_size=(3, 3),
            strides=_strides,
            padding="same",
        )
        self.convolution_layer_type3 = tf.keras.layers.Conv2D(
            filters=_filters * 4,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
        )

        self.batch_normalization = tf.keras.layers.BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        self.shortcut.add(
            tf.keras.layers.Conv2D(
                filters=_filters,
                kernel_size=(1, 1),
                strides=_strides,
            )
        )
        self.shortcut.add(tf.keras.layers.BatchNormalization())


class BottleneckType1(_BottleneckBase):
    def call(self, input, **kwargs):
        residual = self.shortcut(input)

        x = self.convolution_layer_type1(input)
        x = self.batch_normalization(x)
        x = tf.nn.relu(x)
        x = self.convolution_layer_type2(x)
        x = self.batch_normalization(x)
        x = tf.nn.relu(x)
        x = self.convolution_layer_type3(x)
        x = self.batch_normalization(x)

        output = tf.keras.layers.add([residual, x])
        output = tf.nn.relu(output)

        return output


class BottleneckType2(_BottleneckBase):
    def __init__(self):
        super(BottleneckType2, self).__init__()
        self.max_pooling = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=2,
            padding="same",
        )

    def call(self, input, **kwargs):
        residual = self.shortcut(input)

        x = self.max_pooling(x)
        x = self.convolution_layer_type1(input)
        x = self.batch_normalization(x)
        x = tf.nn.relu(x)
        x = self.convolution_layer_type2(x)
        x = self.batch_normalization(x)
        x = tf.nn.relu(x)
        x = self.convolution_layer_type3(x)
        x = self.batch_normalization(x)

        output = tf.keras.layers.add([residual, x])
        output = tf.nn.relu(output)

        return output
