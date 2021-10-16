import numpy as np

def one_hot_expand(numpy_image, tile_types, filepath):
    #assuming H W C?
    #one_hot_image = np.zeros(shape=(numpy_image.shape) + (tile_types,), dtype=np.uint8)
    #one_hot_image
    #print(one_hot_image.shape)

    res = (np.arange(numpy_image.max()+1) == numpy_image[...,None]).astype(np.uint8)
    print(res)


arr = np.zeros((4, 5)).astype(np.uint8)

one_hot_expand(arr, 6, "aa")