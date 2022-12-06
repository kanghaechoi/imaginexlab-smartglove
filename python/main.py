from typing import Union

from filters.dimension import Dimension
from filters.normalization import MinMaxNormalization
from filters.type_conversion import Tensor

from utilities.fetch import Fetch
from filters.extraction import Extraction
from utilities.read import Read

if __name__ == "__main__":
    print("======================")
    print("       MANOVIVO       ")
    print("======================")

    research_question = input(
        "Please select research question.\n(1) Research Question 1\n(2) Research Question 2\n(3) Research Question 3\nPress [Ctrl + C] to exit this program.\n"
    )
    research_question = int(research_question)

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
    features, labels = extraction.extract_dataset()

    features_depth = features.shape[2]
    features_width = features.shape[1]
    features_height = features.shape[0]

    dimension = Dimension()
    if research_question == 2 or research_question == 3:
        features = dimension.numpy_squeeze(
            features,
            features_depth,
            features_width,
            features_height,
        )

    normalization = MinMaxNormalization(features)
    normalized_features = normalization.transform(features)

    normalized_features = dimension.numpy_unsqueeze(
        normalized_features,
        features_depth,
        features_width,
        features_height,
    )

    type_conversion = Tensor()
    normalized_features_as_tensor = type_conversion.array_to_tensor(normalized_features)

    breakpoint()

    if KeyboardInterrupt:
        exit(0)