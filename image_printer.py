import numpy as np
import os

from board_operations import *

GENERATED_FILES_PATH = "generated_images"
DATA_IMAGE_SIZE = (1, 7, 7, 1)

files = os.listdir(GENERATED_FILES_PATH)
NUM_FILES = len(files)

all_arrays = []
for file in files:
    PATH = os.path.join(GENERATED_FILES_PATH, file)
    all_arrays.append(np.load(PATH))
       
unique_arrays = []
for i in range(NUM_FILES):
    is_unique = True    

    for arr in unique_arrays:
        if np.array_equal(all_arrays[i], arr) or (all_arrays[i] == arr).sum() > 44:
            is_unique = False

    if is_unique and valid_room(all_arrays[i]):
        unique_arrays.append(all_arrays[i])


for arr in unique_arrays:
    print(np.reshape(arr, (7, 7)))

print("Found", len(unique_arrays), "unique boards in generated images")
