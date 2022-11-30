from typing import Optional, Tuple

import numpy as np

from scipy import integrate


class Extraction:
    def __init__(
        self,
        _research_question: int,
        _ages: list,
        _data_chunks: zip,
        _sample_length: int = 150,
        _authentication_classes: Optional[list] = None,
    ) -> None:
        self.research_question = _research_question
        self.ages = _ages
        self.data_chunks = _data_chunks
        self.sample_length = _sample_length
        self.authentication_classes = _authentication_classes

    def _create_array_default(self, data: np.ndarray, array_length: int) -> np.ndarray:
        data = data.reshape(1, -1)

        number_of_samples = data.shape[1]

        if number_of_samples >= array_length:
            array = data[0, :array_length].reshape(1, -1)
        else:
            array = np.block([data, np.zeros((1, array_length - number_of_samples))])

        return array

    def _create_array_for_integral(
        self, data: np.ndarray, array_length: int
    ) -> np.ndarray:
        data = data.reshape(1, -1)

        number_of_samples = data.shape[1]

        data_integral = integrate.cumtrapz(data)

        if number_of_samples > array_length:
            array = data_integral[0, :array_length].reshape(1, -1)
        else:
            array = np.block(
                [
                    data_integral,
                    np.zeros((1, array_length - number_of_samples + 1)),
                ]
            )

        return array

    def extract_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        is_first_label_stack: bool = True
        is_first_feature_stack_major: bool = True

        for chunk_index, (hand_data, wrist_data, helical_data, labels) in enumerate(
            self.data_chunks
        ):
            number_of_samples = min(
                len(hand_data),
                len(wrist_data),
                len(helical_data),
            )

            if is_first_label_stack is True:
                labels_stack = np.ones((number_of_samples, 1)) * labels
                is_first_label_stack = False
            else:
                _labels = np.ones((number_of_samples, 1)) * labels
                labels_stack = np.concatenate((labels_stack, _labels), axis=0)

            is_first_feature_stack_minor: bool = True

            for sample_index in range(number_of_samples):
                hand_array = np.array(hand_data[sample_index]).astype(float)
                hand_array = hand_array.transpose()

                wrist_array = np.array(wrist_data[sample_index]).astype(float)
                wrist_array = wrist_array.transpose()

                helical_array = np.array(helical_data[sample_index]).astype(float)
                helical_array = helical_array.transpose()

                hand_x_angle = self._create_array_default(
                    hand_array[0, :], self.sample_length
                )
                hand_y_angle = self._create_array_default(
                    hand_array[1, :], self.sample_length
                )
                hand_z_angle = self._create_array_default(
                    hand_array[2, :], self.sample_length
                )

                thumb_x_angle = self._create_array_default(
                    hand_array[3, :], self.sample_length
                )
                index_x_angle = self._create_array_default(
                    hand_array[4, :], self.sample_length
                )

                thumb_x_acceleration = self._create_array_default(
                    hand_array[6, :], self.sample_length
                )
                thumb_x_velocity = self._create_array_for_integral(
                    hand_array[6, :], self.sample_length
                )
                thumb_y_acceleration = self._create_array_default(
                    hand_array[7, :], self.sample_length
                )
                thumb_y_velocity = self._create_array_for_integral(
                    hand_array[7, :], self.sample_length
                )
                thumb_z_acceleration = self._create_array_default(
                    hand_array[8, :], self.sample_length
                )
                thumb_z_velocity = self._create_array_for_integral(
                    hand_array[8, :], self.sample_length
                )

                index_x_acceleration = self._create_array_default(
                    hand_array[9, :], self.sample_length
                )
                index_x_velocity = self._create_array_for_integral(
                    hand_array[9, :], self.sample_length
                )
                index_y_acceleration = self._create_array_default(
                    hand_array[10, :], self.sample_length
                )
                index_y_velocity = self._create_array_for_integral(
                    hand_array[10, :], self.sample_length
                )
                index_z_acceleration = self._create_array_default(
                    hand_array[11, :], self.sample_length
                )
                index_z_velocity = self._create_array_for_integral(
                    hand_array[11, :], self.sample_length
                )

                wrist_x_angle = self._create_array_default(
                    wrist_array[0, :], self.sample_length
                )
                wrist_y_angle = self._create_array_default(
                    wrist_array[1, :], self.sample_length
                )
                wrist_z_angle = self._create_array_default(
                    wrist_array[2, :], self.sample_length
                )

                wrist_x_acceleration = self._create_array_default(
                    wrist_array[3, :], self.sample_length
                )
                wrist_x_velocity = self._create_array_for_integral(
                    wrist_array[3, :], self.sample_length
                )
                wrist_y_acceleration = self._create_array_default(
                    wrist_array[4, :], self.sample_length
                )
                wrist_y_velocity = self._create_array_for_integral(
                    wrist_array[4, :], self.sample_length
                )
                wrist_z_acceleration = self._create_array_default(
                    wrist_array[5, :], self.sample_length
                )
                wrist_z_velocity = self._create_array_for_integral(
                    wrist_array[5, :], self.sample_length
                )

                helical_x_angle = self._create_array_default(
                    helical_array[0, :], self.sample_length
                )
                helical_y_angle = self._create_array_default(
                    helical_array[1, :], self.sample_length
                )
                helical_z_angle = self._create_array_default(
                    helical_array[2, :], self.sample_length
                )

                feature = np.concatenate(
                    (
                        hand_x_angle,
                        hand_y_angle,
                        hand_z_angle,
                        thumb_x_angle,
                        index_x_angle,
                        thumb_x_acceleration,
                        thumb_y_acceleration,
                        thumb_z_acceleration,
                        thumb_x_velocity,
                        thumb_y_velocity,
                        thumb_z_velocity,
                        index_x_acceleration,
                        index_y_acceleration,
                        index_z_acceleration,
                        index_x_velocity,
                        index_y_velocity,
                        index_z_velocity,
                        wrist_x_angle,
                        wrist_y_angle,
                        wrist_z_angle,
                        wrist_x_acceleration,
                        wrist_y_acceleration,
                        wrist_z_acceleration,
                        wrist_x_velocity,
                        wrist_y_velocity,
                        wrist_z_velocity,
                        helical_x_angle,
                        helical_y_angle,
                        helical_z_angle,
                    )
                )

                if is_first_feature_stack_minor is True:
                    feature_stack_minor = feature
                    is_first_feature_stack_minor = False
                else:
                    feature_stack_minor = np.dstack((feature_stack_minor, feature))

                if sample_index == number_of_samples - 1:
                    is_first_feature_stack_minor = True

                print(
                    "Subject {:d}'s {:d} feature is add.".format(
                        chunk_index,
                        sample_index,
                    )
                )

            if is_first_feature_stack_major is True:
                feature_stack_major = feature_stack_minor
                is_first_feature_stack_major = False
            else:
                feature_stack_major = np.dstack(
                    (feature_stack_major, feature_stack_minor)
                )

            print("***Subject {:d}'s feature stack is added.".format(chunk_index))

        return feature_stack_major, labels_stack
