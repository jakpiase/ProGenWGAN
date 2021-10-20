import numpy as np
import os

from board_operations import *

NUMBER_OF_TILE_TYPES = 4
IMAGE_SIZE = (7, 7)
DATA_IMAGE_SIZE = (1, 7, 7, 1)
BASE_DATASET_PATH = "all_datasets/numpy_images.npy"
OUTPUT_PATH = "all_datasets/numpy_images.npy"
GENERATED_FILES_PATH = "generated_images"

base_dataset = np.load(BASE_DATASET_PATH)
generated_dataset_list = load_list_of_arrays_from_folder(GENERATED_FILES_PATH)


unique_arrays = []
for i in range(len(generated_dataset_list)):
    is_unique = True    

    for arr in unique_arrays:
        if np.array_equal(generated_dataset_list[i], arr):
            is_unique = False

    if is_unique:
        unique_arrays.append(generated_dataset_list[i])

print("Found", len(unique_arrays), "unique boards in newly generated images")

unique_arrays = augment_data(unique_arrays)

print("Augmented data, now it contains", len(unique_arrays), "generated images")

batched_unique_arrays = batched_dataset_from_list(unique_arrays)

combined_dataset = np.concatenate((base_dataset, batched_unique_arrays))
print("Combining old dataset with newly generated images, now dataset has", combined_dataset.shape[0]  ,"arrays")
np.save(OUTPUT_PATH, combined_dataset)

files = os.listdir(GENERATED_FILES_PATH)
for file in files:
    os.remove(os.path.join(GENERATED_FILES_PATH, file))

#datasets_combined = np.reshape(unique_arrays[0], DATA_IMAGE_SIZE)

#for i in range(1, len(unique_arrays)):
#    datasets_combined = np.concatenate((datasets_combined, unique_arrays[i]))
#
#print("Combined dataset shape:", datasets_combined.shape)
#
#print("Deleting already generated files")
#for file in files:
#    #print("Deleting file", file)
#    os.remove(os.path.join(GENERATED_FILES_PATH, file))

#print("Saving files to " + OUTPUT_PATH)
#np.save(OUTPUT_PATH, datasets_combined)