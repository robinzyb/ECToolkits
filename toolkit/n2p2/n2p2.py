from random import shuffle
import numpy as np

# User input
in_file = "./input.data"
frac_select = 0.1

def read_n2p2_data(file_name):
    """
    function to read n2p2 data file
    every frame will be stored as string in list
    return: list
    """
    fr = open(file_name, 'r')
    struct_list = []
    for line in fr:
        if line == "begin\n":
            tmp_struct = line
        elif line == "end\n":
            tmp_struct += line
            struct_list.append(tmp_struct)
        else:
            tmp_struct += line
    fr.close()
    return struct_list


def split_n2p2_data(n2p2_file, file_1, file_2, percentage):
    """
    to splite the n2p2 data file in to two file, the percentage is for file_1
    the rest of data is for file_2
    percentage: number smaller than 1
    return: None, just save new file to file_1, file_2
    """
    struct_list = read_n2p2_data(n2p2_file)
    #determine the number
    struct_numbs = len(struct_list)
    # k the number for splittingp point
    k = int(struct_numbs * percentage)
    shuffle(struct_list)
    part_1 = struct_list[:k]
    part_2 = struct_list[k:]
    print("You have {0} structures total".format(len(struct_list)))
    print("length of first file is {0}".format(len(part_1)))
    print("length of second file is {0}".format(len(part_2)))

    with open(file_1, "w") as fw:
        for line in part_1:
            fw.write(line)
    with open(file_2, "w") as fw:
        for line in part_2:
            fw.write(line)
    print("the data have been split into {0} and {1}".format(file_1, file_2))

