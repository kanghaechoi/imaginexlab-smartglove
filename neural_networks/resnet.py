import tensorflow as tf

from neural_networks._residual_block import BaseBlock, BottleneckType1, BottleneckType2


def __build_base_block_layer(filters: int, blocks: int, strides: int = 1):
    layer = tf.keras.Sequential()
    layer.add(BaseBlock(_filters=filters, _strides=strides))

    for _ in range(1, blocks):
        layer.add(BaseBlock(_filters=filters, _strides=1))

    return layer


def __build_bottleneck_type1_convolution_layer(
    filters: int,
    blocks: int,
    strides: int = 1,
):
    layer = tf.keras.Sequential()
    layer.add(BottleneckType1(_filters=filters, _strides=strides))

    for _ in range(1, blocks - 1):
        layer.add(BottleneckType1(_filters=filters, _strides=1))

    return layer


def __build_bottleneck_type2_convolution_layer(
    filters: int,
    blocks: int,
    strides: int = 1,
):
    layer = tf.keras.Sequential()
    layer.add(BottleneckType2(_filters=filters, _strides=strides))

    for _ in range(1, blocks - 1):
        layer.add(BottleneckType1(_filters=filters, _strides=1))

    return layer


class ResNet50(tf.keras.Model):
    def __init__(self, _units: int):
        super(ResNet50, self).__init__()

        self.convolution_layer_type_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            padding="same",
        )
        self.batch_normalization = tf.keras.layers.BatchNormalization()

        self.convolution_layer_type2 = __build_bottleneck_type2_convolution_layer(
            filters=64,
            blocks=5,
        )
        self.convolution_layer_type3 = __build_bottleneck_type1_convolution_layer(
            filters=128,
            blocks=5,
        )
        self.convolution_layer_type4 = __build_bottleneck_type1_convolution_layer(
            filters=256,
            blocks=5,
        )
        self.convolution_layer_type5 = __build_bottleneck_type1_convolution_layer(
            filters=512,
            blocks=5,
        )

        self.average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.fully_connected_layer = tf.keras.layers.Dense(
            units=_units,
            activation=tf.keras.activations.softmax,
        )

    def call(self, input, training=None, mask=None) -> any:
        x = self.convolution_layer_type_1(input)
        x = self.batch_normalization(x)
        x = tf.nn.relu(x)

        x = self.convolution_layer_type2(x)
        x = self.convolution_layer_type3(x)
        x = self.convolution_layer_type4(x)
        x = self.convolution_layer_type5(x)

        x = self.average_pooling(x)

        output = self.fully_connected_layer(x)

        return output
