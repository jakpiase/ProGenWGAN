import numpy as np
import os

from board_operations import *

GENERATED_FILES_PATH = "generated_images"
BASE_DATASET_PATH = "all_datasets/numpy_images.npy"
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


new_unique_arrays = []
base_dataset = np.load(BASE_DATASET_PATH)

for arr in unique_arrays:
    is_unique = True
    for i in range(base_dataset.shape[0]):
        if arrays_similar(arr, base_dataset[i]):
            is_unique = False

    if is_unique:
        new_unique_arrays.append(arr)

save_arr = []

occurances = [0] * 7
for arr in new_unique_arrays:
    if (arr == 2).sum() > 0 or (arr == 6).sum() > 0:
        save_arr.append(arr)


for arr in new_unique_arrays:
    for i in range(7):
        if (arr == i).sum() > 0:
            occurances[i] += 1

print(len(save_arr))




#for arr in new_unique_arrays:
#    print(np.reshape(arr, (7, 7)))

print("Found", len(unique_arrays), "unique boards in generated images")
print("Found", len(new_unique_arrays), "unique boards in generated images")


for i in range(7):
    print("i =", i ,"occ[i]=", occurances[i])