import numpy as np

PATH = "generated_images/validroom0.npy"
PATH2 = "generated_images/validroom1.npy"
array = np.load(PATH)

print(np.reshape(array, (7, 7)))

#equal = 1
#different = 0
#
#for i in range(1, 1000):
#    PATH2 = "generated_images/validroom" + str(1) + ".npy"
#    array2 = np.load(PATH2)
#
#    if np.array_equal(array, array2):
#        equal += 1
#    else:
#        different += 1
#
#
#print("equal: ", equal)
#print("different: ", different)
