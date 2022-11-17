from typing import Union

from utilities.fetch import Fetch
from utilities.dataset import Dataset
from utilities.read import Read

if __name__ == "__main__":
    selected_question = input(
        "Please select research question.\n(1) Research Question 1\n(2) Research Question 2\n(3) Research Question 3\nPress [Ctrl + C] to exit this program.\n"
    )
    selected_question = int(selected_question)

    age_20s: int = 20
    age_50s: int = 50
    age_70s: int = 70

    is_not_authorized = 0
    is_authorized = 1

    if selected_question == 1:
        ages = [age_20s, age_50s, age_70s]
        authentication_classes = None
        authentication_flag = False
    elif selected_question == 2 or selected_question == 3:
        ages = [age_20s]
        authentication_classes = [is_authorized, is_not_authorized]
        authentication_flag = True
    else:
        raise ValueError

    fetch = Fetch(selected_question, ages, authentication_flag, authentication_classes)
    data_chunks: zip = fetch.fetch_chunks()

    sample_length: int = 150
    dataset = Dataset(
        selected_question,
        ages,
        data_chunks,
        sample_length,
        authentication_classes,
    )
    features = dataset.create_dataset()

    breakpoint()

    if KeyboardInterrupt:
        exit(0)
