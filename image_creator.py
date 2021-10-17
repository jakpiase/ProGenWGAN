import numpy as np
from PIL import Image

ROOM_SIZE = 7 * 7

EMPTY = 0
ENEMY = 1
TREASURE = 2
WATER = 3
FURNITURE = 4
MONUMENT = 5

def has_same_neighboor(room, x, y, TYPE):
    if x > 0:
        if room[x - 1][y] == TYPE:
            return True
    if y > 0:
        if room[x][y - 1] == TYPE:
            return True
    if x < room.shape[0] - 1:
        if room[x + 1][y] == TYPE:
            return True
    if y < room.shape[1] - 1:
        if room[x][y + 1] == TYPE:
            return True

    return False

def check_monument_correctness(room):
    if (room == MONUMENT).sum() == 0:
        return True

    if (room == MONUMENT).sum() != 4:
        return False

    indexes = np.where(room==MONUMENT)
    
    if indexes[0][0] != indexes[0][1] or indexes[0][2] != indexes[0][3] or indexes[1][0] != indexes[1][2] or indexes[1][1] != indexes[1][3]:
        return False

    return True

def check_water_correctness(room):
    if (room == WATER).sum() == 0:
        return True

    if (room == WATER).sum() > 16:
        return False

    indexes = np.where(room==WATER)

    for i in range(len(indexes[0])):
        if has_same_neighboor(room, indexes[0][i], indexes[1][i], WATER) == False:
            return False

    return True


def check_furniture_correctness(room):
    if (room == FURNITURE).sum() == 0:
        return True

    if (room == FURNITURE).sum() > 8:
        return False

    indexes = np.where(room==FURNITURE)

    for i in range(len(indexes[0])):
        if has_same_neighboor(room, indexes[0][i], indexes[1][i], FURNITURE) == False:
            return False

    return True

def valid_room(room):
    if (room == EMPTY).sum() == ROOM_SIZE:
        return False
    if (room == EMPTY).sum() < ROOM_SIZE / 2:
        return False
    if (room == ENEMY).sum() > 5:
        return False
    if (room == TREASURE).sum() > 3:
        return False
    if (room == WATER).sum() > 10:
        return False
    if (room == FURNITURE).sum() != 0:
        print("\n\nFURNITURE SHOULD NOT BE PRESENT NOW\n\n")
    if (room == MONUMENT).sum() != 0:
        print("\n\nMONUMENT SHOULD NOT BE PRESENT NOW\n\n")
#    if check_water_correctness(room) == False:
#        return False
#    if check_furniture_correctness(room) == False:
#        return False
#    if check_monument_correctness(room) == False:
#        return False

    return True
    

def initialize_arrays():
    list = [0] * 16

    list[0] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 3, 3, 3],
                 [0, 0, 0, 0, 3, 3, 3],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 1, 0, 0, 0],
                 [0, 0, 2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    list[1] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 2, 0, 0],
                 [0, 0, 0, 0, 0, 0, 3],
                 [3, 3, 0, 0, 0, 0, 3],
                 [3, 3, 0, 0, 0, 0, 3],
                 [3, 3, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    list[2] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 3, 3, 3],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 3],
                 [0, 0, 1, 0, 0, 0, 3],
                 [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[3] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 3, 3, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 3, 0, 0, 0],
                 [3, 3, 0, 3, 0, 0, 0],
                 [3, 3, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[4] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[5] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 3, 0],
                 [0, 0, 0, 0, 1, 3, 0],
                 [0, 0, 0, 0, 0, 3, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[6] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [3, 0, 3, 3, 3, 0, 0],
                 [3, 0, 0, 0, 0, 0, 0],
                 [3, 0, 0, 0, 0, 0, 0],
                 [3, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[7] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 3, 0, 0],
                 [0, 0, 3, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 3, 3],
                 [0, 0, 0, 0, 0, 3, 3],
                 [3, 3, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[8] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 2, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[9] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[10] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 2, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0]]).astype(np.uint8)
    list[11] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 3, 3, 0, 0, 0, 0],
                 [0, 0, 3, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[12] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 3, 3, 3, 0, 0],
                 [0, 0, 3, 3, 3, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[13] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 3, 0, 0, 1, 0],
                 [0, 0, 3, 0, 0, 0, 0],
                 [0, 0, 3, 0, 3, 3, 0],
                 [0, 0, 3, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[14] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 2, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    list[15] = np.array([[0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 3, 3, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 3, 3, 0, 0]]).astype(np.uint8)                

    for i in range(len(list)):
        list[i] = np.reshape(list[i], (1, ) + list[i].shape + (1,))
        valid_room(list[i])

    all_images = list[0]
    for i in range(1, len(list)):
        all_images = np.concatenate((all_images, list[i]))

    np.save(file="progen_images/numpy_images.npy", arr=all_images)
    

initialize_arrays()




