import numpy as np

DATASET_PATH = "all_datasets/numpy_images.npy"

dataset = np.load(DATASET_PATH)

print(dataset.shape)

occurances = 7 * [0]

for i in range(dataset.shape[0]):
    for j in range(7):
        if (dataset[i] == j).sum() > 0:
            occurances[j] += 1

for i in range(7):
    print("Occurances[" + str(i) + "]=" + str(occurances[i]))

a.e() # some old shit here, delete that
import os
from board_operations import *


GEN_DIR = "generated_images"

new_dataset = load_list_of_arrays_from_folder(GEN_DIR)
new_dataset = get_unique(new_dataset)
new_dataset = batched_dataset_from_list(new_dataset)


print(new_dataset.shape)
shrinked_dataset = np.reshape(new_dataset[0], (1, 7, 7, 1))

for i in range(1, new_dataset.shape[0]):
    if (new_dataset[i] == 2).sum() != 0 or (new_dataset[i] == 4).sum() != 0 or (new_dataset[i] == 6).sum() > 0:
        shrinked_dataset = np.concatenate((shrinked_dataset, np.reshape(new_dataset[i], (1, 7 ,7 ,1))))

print(shrinked_dataset.shape)  

occurances = 7 * [0]

dataset = np.concatenate((dataset, shrinked_dataset))

for i in range(dataset.shape[0]):
    for j in range(7):
        if (dataset[i] == j).sum() > 0:
            occurances[j] += 1

for i in range(7):
    print("Occurances[" + str(i) + "]=" + str(occurances[i]))

threshold = 600

reduce_dataset = np.reshape(dataset[0], (1, 7, 7, 1))
j = 0
for i in range(1, dataset.shape[0]):
    if j < threshold:
        if (dataset[i] == 3).sum() > 0 and (dataset[i] == 4).sum() == 0 and (dataset[i] == 6).sum() == 0 and (dataset[i] == 2).sum() == 0:
            j += 1
        else:
            reduce_dataset = np.concatenate((reduce_dataset, np.reshape(dataset[i], (1, 7, 7, 1))))
    else:
        reduce_dataset = np.concatenate((reduce_dataset, np.reshape(dataset[i], (1, 7, 7, 1))))


print(reduce_dataset.shape)

occurances = 7 * [0]

for i in range(reduce_dataset.shape[0]):
    for j in range(7):
        if (reduce_dataset[i] == j).sum() > 0:
            occurances[j] += 1

for i in range(7):
    print("Occurances[" + str(i) + "]=" + str(occurances[i]))


np.save("all_datasets/numpy_images_2776.npy", reduce_dataset)