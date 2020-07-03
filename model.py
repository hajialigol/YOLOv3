import os
import re
import torch.nn as nn


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


def create_network(network_blocks):
    # network_info = network_blocks[0]
    in_channels = int(network_info['channels'])
    out_channels = 0
    kernel_size = 0
    stride = 1
    pad = 1
    batch_norm = 0
    module_list = nn.ModuleList()

    for i, network in enumerate(network_blocks[1:]):
        sequential_module = nn.Sequential()
        x = network['model_type']

        try:
            out_channels = int(network['filters'])
            kernel_size = int(network['size'])
            stride = int(network['stride'])
            padding = int(network['pad'])
            batch_norm = int(network['batch_normalize'])
            activation_name = network['activation']
        except:
            print("Something not there")

        if x == "convolutional":
            # make a convolutional layer
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=False)
            sequential_module.add_module("conv" + str(i), conv_layer)

        if batch_norm == 1:
            batch_layer = nn.BatchNorm2d(num_features=out_channels)
            sequential_module.add_module("batch" + str(i), batch_layer)
            channels = out_channels

        if activation_name == "leaky":
            activation_layer = nn.LeakyReLU(inplace=True)
            sequential_module.add_module(activation_name + str(i), activation_layer)

        in_channels = out_channels
        print(sequential_module)
        module_list.append(sequential_module)

    # elif x == "shortcut":
    # make a shortcut layer

    # elif x == "yolo":
    # make a yolo layer

    # elif x == "route":
    # make a route layer

    # elif x == "upsample":
    # make an upsample layer

    print("Module list:", module_list)
    return network_info


if __name__ == "__main__":
    file_directory = os.getcwd() + '\cfg\yolov3.cfg.txt'
    block_list = read_cfg(file_directory)
    network_info = create_network(block_list)
