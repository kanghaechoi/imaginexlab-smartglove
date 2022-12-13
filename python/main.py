from dataset.extraction import Extraction
from dataset.divide import Divide
from filters.normalization import MinMaxNormalization
from filters.relieff import ReliefF
from utilities.dimension import Dimension
from utilities.fetch import Fetch
from utilities.read import Read
from type.as_tensor import AsTensor

from neural_networks.resnet import ResNet50
from neural_networks.nn_training import NNTraining

import numpy as np

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
    elif research_question == 2 or research_question == 3:
        ages = [age_20s]
        authentication_classes = [is_authorized, is_not_authorized]
        authentication_flag: bool = True
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
    if research_question == 2 or research_question == 3:
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
    Data type conversion from Numpy array to tensor
    """
    as_tensor = AsTensor()
    training_data_as_tensor = as_tensor.array_to_tensor(training_data)
    training_labels_as_tensor = as_tensor.array_to_tensor(training_labels)

    test_data_as_tensor = as_tensor.array_to_tensor(test_data)
    test_labels_as_tensor = as_tensor.array_to_tensor(test_labels)

    breakpoint()
    nn_model = ResNet50([3, 4, 6, 3], 2)
    nn_training = NNTraining(nn_model)
    nn_training.train_model(training_data_as_tensor, training_labels_as_tensor)

    breakpoint()

    if KeyboardInterrupt:
        exit(0)
