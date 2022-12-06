class Read:
    def read(self, file_path: str) -> list:
        with open(file_path, "r") as file:
            data = []
            buffer = []

            previous_z_axis_euler_angle = 0
            current_z_axis_euler_angle = 0

            swap_count = 0

            for row in file:
                row_data = row.split()
                current_z_axis_euler_angle = float(row_data[0])

                if abs(current_z_axis_euler_angle + previous_z_axis_euler_angle) <= 2:
                    data.append(buffer)

                    previous_z_axis_euler_angle = current_z_axis_euler_angle
                    buffer = row_data
                    # swap_count += 1

                else:
                    buffer.append(row_data)

                    # if swap_count == 17:
                    #     break

        return data
