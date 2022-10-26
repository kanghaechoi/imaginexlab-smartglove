from utilities.path import Path
from utilities.read import Read

if __name__ == "__main__":
    research_question1_20s_path_instance = Path(1, 20)
    research_question1_20s_zipped_paths = research_question1_20s_path_instance.zip()
    for hand_path, wrist_path, helical_path in research_question1_20s_zipped_paths:
        reader = Read()

        breakpoint()
        hand_data = reader.read(hand_path)
        breakpoint()
    _20s_helical_read = Read(1, "Helical", 20)
