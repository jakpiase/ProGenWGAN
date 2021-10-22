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