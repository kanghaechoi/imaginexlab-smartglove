import glob

from typing import Optional


def read_paths(
    _research_question: int,
    _body_part: str,
    _age: int,
    _auth_tag: Optional[int] = None,
) -> None:
    if _research_question == 1:
        paths = sorted(
            glob.glob(
                "./data/research_question"
                + str(_research_question)
                + "/"
                + _body_part
                + "_IMU_"
                + str(_age)
                + "_*.txt"
            )
        )
    else:
        paths = sorted(
            glob.glob(
                "./data/research_question"
                + str(_research_question)
                + "/"
                + _body_part
                + "_IMU_"
                + str(_age)
                + "_"
                + _auth_tag
                + "_*.txt"
            )
        )

    return paths


class Path:
    def __init__(
        self,
        _research_question: int,
        _age: int,
        _auth_tag: Optional[int] = None,
    ) -> None:
        self.hand_paths = read_paths(_research_question, "Hand", _age, _auth_tag)
        self.wrist_paths = read_paths(_research_question, "Wrist", _age, _auth_tag)
        self.helical_paths = read_paths(_research_question, "Helical", _age, _auth_tag)

    def zip(self) -> zip:
        zipped_paths = zip(self.hand_paths, self.wrist_paths, self.helical_paths)

        return zipped_paths
