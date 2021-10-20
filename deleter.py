import os

GENERATED_FILES_PATH = "generated_images"

files = os.listdir(GENERATED_FILES_PATH)
print("Listdir finished")

i = 0
for file in files:
    #print("Deleting file", file)
    os.remove(os.path.join(GENERATED_FILES_PATH, file))
    i += 1
    if i % 1000 == 0:
        print("Deleted", i, "files already")