import tensorflow as tf
import numpy as np

from dataset.extraction import Extraction
from dataset.divide import Divide

from filters.normalization import MinMaxNormalization
from filters.relieff import ReliefF

from utilities.dimension import Dimension
from utilities.fetch import Fetch

from neural_networks.resnet import ResNet
from neural_networks.nn_training import NNTraining


if __name__ == "__main__":
    print("======================")
    print("       MANOVIVO       ")
    print("======================")

    research_question = input(
        "Please select research question.\n(1) Research Question 1\n(2) Research Question 2\n(3) Research Question 3\nPress [Ctrl + C] to exit this program.\n"
    )
    research_question = int(research_question)

    """
    Data fetch configuration
    """
    age_20s: int = 20
    age_50s: int = 50
    age_70s: int = 70

    is_not_authorized: int = 0
    is_authorized: int = 1

    if research_question == 1:
        ages = [age_20s, age_50s, age_70s]
        authentication_classes = None
        authentication_flag: bool = False
        number_of_classes = len(ages)
    elif research_question == 2 or research_question == 3:
        ages = [age_20s]
        authentication_classes = [is_authorized, is_not_authorized]
        authentication_flag: bool = True
        number_of_classes = len(authentication_classes)
    else:
        raise ValueError

    """
    Data fetch
    """
    fetch = Fetch(research_question, ages, authentication_flag, authentication_classes)
    data_chunks: zip = fetch.fetch_chunks()

    sample_length: int = 150
    extraction = Extraction(
        research_question,
        ages,
        data_chunks,
        sample_length,
        authentication_classes,
    )
    data, labels = extraction.extract_dataset()

    data_depth: int = data.shape[2]
    data_width: int = data.shape[1]
    data_height: int = data.shape[0]

    """
    Array dimension manipulation (Temporal)
    """
    dimension = Dimension()
    # if research_question == 2 or research_question == 3:
    data = dimension.numpy_squeeze(
        data,
        data_depth,
        data_width,
        data_height,
    )

    """
    Feature Normalization
    """
    normalization = MinMaxNormalization(data)
    normalized_data = normalization.transform(data)

    normalized_data = dimension.numpy_unsqueeze(
        normalized_data,
        data_depth,
        data_width,
        data_height,
    )

    divide = Divide(normalized_data, labels)
    divide.fit(test_dataset_ratio=0.2)

    training_data, training_labels = divide.training_dataset()
    test_data, test_labels = divide.test_dataset()

    """
    Feature selection using ReliefF algorithm
    """
    number_of_feature_reduction = input(
        "Please insert number of features to be reduced.\n"
    )
    number_of_feature_reduction = int(number_of_feature_reduction)

    if research_question == 2 or research_question == 3:
        training_data_for_relieff = np.sum(training_data, axis=1)

    relieff = ReliefF(
        n_neighbors=normalized_data.shape[2],
        n_features_to_keep=normalized_data.shape[2] - number_of_feature_reduction,
    )
    relieff.fit_transform(training_data_for_relieff, np.squeeze(training_labels))
    top_feature_indices = relieff.top_features[0 : relieff.n_features_to_keep]

    training_data = training_data[:, :, top_feature_indices]
    training_data = np.expand_dims(training_data, axis=3)
    test_data = test_data[:, :, top_feature_indices]
    test_data = np.expand_dims(test_data, axis=3)

    """
    Model training
    """
    epochs: str = input("Please insert the number of epochs: ")
    epochs: int = int(epochs)

    batch_size: str = input("Please insert batch size: ")
    batch_size: int = int(batch_size)

    chosen_model: str = input(
        "Please select a model to train.\n(1) ResNet-50\n(2) ResNet-101\n(3) ResNet-152\n"
    )
    chosen_model: int = int(chosen_model)

    if chosen_model == 1:
        resnet_block_parameters: list = [3, 4, 6, 3]
        saved_models_path: str = "./python/saved_models/resnet50"
    elif chosen_model == 2:
        resnet_block_parameters: list = [3, 4, 23, 3]
        saved_models_path: str = "./python/saved_models/resnet101"
    elif chosen_model == 3:
        resnet_block_parameters: list = [3, 8, 36, 3]
        saved_models_path: str = "./python/saved_models/resnet152"
    else:
        ValueError()

    resnet = ResNet(resnet_block_parameters, number_of_classes)

    adam_optimizer = tf.keras.optimizers.Adam
    rms_prop_optimizer = tf.keras.optimizers.RMSprop

    nn_training = NNTraining(
        resnet,
        adam_optimizer,
        epochs,
        batch_size,
        saved_models_path,
    )
    nn_training.train_model(training_data, training_labels)
    nn_training.save_trained_model()
    nn_training.test_trained_model(test_data, test_labels)

    new_resnet = tf.keras.models.load_model(saved_models_path)
    new_resnet.summary()

    breakpoint()

    if KeyboardInterrupt:
        exit(0)
