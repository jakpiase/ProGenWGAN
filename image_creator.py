import numpy as np
from PIL import Image

from board_operations import *

DATA_IMAGE_SIZE = (1, 7, 7, 1)

ROWS = 7
COLS = 7
ROOM_SIZE = ROWS * COLS


EMPTY = 0
ENEMY = 1
TREASURE = 2
WATER = 3
TABLE = 4
DRESSER = 5
MONUMENT = 6
    

def load_plain_data_from_file(PATH):
    line_number = 0

    list_of_arrays = []
    with open(PATH, "r") as file:
        lines = file.readlines()
        i = 0
        arr = np.zeros((COLS, ROWS), dtype=np.uint8)

        for line in lines:
            if line != "\n":
                arr[i % ROWS] = np.fromstring(line.strip(), dtype=np.uint8, sep=' ')
                    
                i += 1

                if i % ROWS == 0:
                    list_of_arrays.append(np.reshape(arr.copy(), DATA_IMAGE_SIZE))

    return list_of_arrays

dataset = load_plain_data_from_file("image_data.txt")
validate_dataset(dataset)

augmented_dataset = augment_data(dataset)

batched_aug_dataset = batched_dataset_from_list(augmented_dataset)
np.save("progen_images/numpy_images.npy", batched_aug_dataset)