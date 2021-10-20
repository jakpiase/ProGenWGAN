import numpy as np

from board_operations import *

a = np.load("all_datasets/numpy_images.npy")

a = convert_to_4D_list(a)

normal_images = 120

for i in range(144):
    print(np.array_equal(a[120 + i], a[normal_images + 144 + i]))


new_list = []

for i in range(120 + 144):
    new_list.append(a[i])

#print(len(get_unique(new_list)))

b = batched_dataset_from_list(new_list)


#np.save("all_datasets/numpy_images.npy", b)