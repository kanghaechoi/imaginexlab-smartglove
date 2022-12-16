import tensorflow as tf


loss_object = tf.keras.losses.BinaryCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.BinaryCrossentropy(name="train_loss")
train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

test_loss = tf.keras.metrics.BinaryCrossentropy(name="test_loss")
test_accuracy = tf.keras.metrics.BinaryAccuracy(name="test_accuracy")


@tf.function
def training_step(model: tf.keras.Model, data: tf.Tensor, labels: tf.Tensor):
    with tf.GradientTape(persistent=True) as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(data, training=True)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(labels, predictions)
    train_accuracy(labels, predictions)


@tf.function
def test_step(model, data: tf.Tensor, labels: tf.Tensor):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(data, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


class NNTraining:
    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam,
        epochs: int = 256,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs

    def train_model(self, training_data: tf.Tensor, training_labels: tf.Tensor) -> None:
        for epoch in range(self.epochs):
            # Reset the metrics at the start of the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

            for batch_count, (mini_training_data, mini_training_labels) in enumerate(
                zip(training_data, training_labels)
            ):
                mini_training_one_hot_labels = tf.one_hot(
                    tf.squeeze(mini_training_labels),
                    2,
                )
                training_step(
                    self.model, mini_training_data, mini_training_one_hot_labels
                )
                print("Batch index: ", batch_count + 1)

            # for test_images, test_labels in test_ds:
            #     test_step(test_images, test_labels)

            print(
                f"Epoch {epoch + 1}, "
                f"Loss: {train_loss.result()}, "
                f"Accuracy: {train_accuracy.result() * 100}, "
                # f"Test Loss: {test_loss.result()}, "
                # f"Test Accuracy: {test_accuracy.result() * 100}"
            )
