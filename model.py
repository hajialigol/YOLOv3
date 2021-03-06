import os
import re
import torch.nn as nn


class VacantLayer(nn.Module):
    def __init__(self):
        super(VacantLayer, self).__init__()



class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        self.anchors = anchors



def read_cfg(file):
    '''
    Desc:
        This method reads the YoloV3 configuration file and maps the network name to network's architecture.
    Inpt:
        file [file]: The configuration file to read and map data from.
    Oupt:
        block_list [list]: List of dictionaries, where each dictionary has keys corresponding to feature names
        of an architecture and the values are the feature's values.
    '''
    block_list = []
    block_map = {}
    with open(file_directory, 'r') as file:
        for i, line in enumerate(file):
            line_word_list = line.split("=")
            line_length = len(line_word_list)
            if line[0] == '[':  # Check if you're at a new block. If you are, you need to create a new dictionary
                if i != 0:  # Avoid adding an empty dictionary
                    block_list.append(block_map.copy())
                block_map = {}
                value = re.findall(r"\[([A-Za-z0-9_]+)\]", line_word_list[0])[0]
                block_map["model_type"] = value
            elif line_length > 1:  # If the string is not a new model type, you have the key and values
                key = line_word_list[0].replace("# ", "").strip()
                value = line_word_list[1].replace("\n", "")
                block_map[key] = value
    return block_list



def create_network(network_blocks):
    '''
    Desc:
        This method creates sequential models from a list of dictionaries corresponding to the model's
        architecture.
    Inpt:
        network_blocks [list]: List YoloV3's architecture as a list of dictionaries.
    Oupt:
        network_info [list]: Information about the YoloV3 network.
        module_list [list]: List of sequential models.

    '''
    network_info = network_blocks[0]
    in_channels = int(network_info['channels'])
    out_channels = 0
    kernel_size = 0
    stride = 1
    padding = 1
    module_list = nn.ModuleList()

    for i, network in enumerate(network_blocks[1:]):
        sequential_module = nn.Sequential()
        x = network['model_type']

        try:
            out_channels = int(network['filters'])
            kernel_size = int(network['size'])
            stride = int(network['stride'])
            padding = int(network['pad'])
        except:
            pass

        if x == "convolutional":
            bias_parameter = (False if "batch_normalize" in network.keys() else True)

            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=bias_parameter)
            sequential_module.add_module("conv" + str(i), conv_layer)

            activation_name = network['activation']
            if activation_name == "leaky":
                activation_layer = nn.LeakyReLU(inplace=True)
                sequential_module.add_module(activation_name + str(i), activation_layer)

            if not bias_parameter:
                batch_layer = nn.BatchNorm2d(num_features=out_channels)
                sequential_module.add_module("batch" + str(i), batch_layer)

        elif x == "upsample":
            upsample_layer = nn.Upsample(scale_factor=2, mode="bilinear")
            sequential_module.add_module("upsample" + str(i), upsample_layer)

        elif x == "route":
            route = VacantLayer()
            sequential_module.add_module("route" + str(i), route)
            layers = network['layers'].split(",")
            layer_list = [int(layer) for layer in layers]
            first_layer = layer_list[0]

            if len(layer_list) == 1:
                new_layer = module_list[i + first_layer]
                out_channels = new_layer[0].out_channels

            elif len(layer_list) == 2:
                second_layer = layer_list[1]
                first_layer_filters = module_list[i - 1 + first_layer][0].out_channels
                second_layer_filters = module_list[second_layer - 1][0].out_channels
                out_channels = first_layer_filters + second_layer_filters


        elif x == "shortcut":
            shortcut = VacantLayer()
            sequential_module.add_module("shortcut" + str(i), shortcut)

        elif x == "yolo":
            masks = network['mask'].split(",")
            masks = [int(mask) for mask in masks]
            anchors = network['anchors'].split(",")
            anchors = [int(anchor) for anchor in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            DetectionLayer(anchors)

        in_channels = out_channels

        module_list.append(sequential_module)

    return network_info, module_list



if __name__ == "__main__":
    file_directory = os.getcwd() + '\cfg\yolov3.cfg.txt'
    block_list = read_cfg(file_directory)
    network_info, module_list = create_network(block_list)
