import os
import re


def read_cfg(file):
    block_list = []
    block_map = {}
    with open(file_directory, 'r') as file:
        for i, line in enumerate(file):
            line_word_list = line.split("=")
            line_length = len(line_word_list)
            if line[0] == '[':  # Check if you're at a new block => you need to create a new dictionary
                if i != 0:  # Avoid adding an empty dictionary
                    block_list.append(block_map.copy())
                block_map = {}
                value = re.findall(r"\[([A-Za-z0-9_]+)\]", line_word_list[0])[0]
                block_map["model_type"] = value
            elif line_length > 1:  # If the string is not a new model type, you have the key and values
                # Check if the string is a particular length:
                # If it is, you're going to have to add the other elements as  "values" in key-value pairs
                key = line_word_list[0].replace("# ", "")
                value = line_word_list[1].replace("\n", "")
                block_map[key] = value
    return block_list


if __name__ == "__main__":
    file_directory = os.getcwd() + '\cfg\yolov3.cfg.txt'
    block_list = read_cfg(file_directory)
