import numpy as np

MAX_NUM_OBJECTS = 1968
LIST_LENGTH = 12880
directory = "E:\Datasets\WIDER\wider_face_split/wider_face_train_bbx_gt.txt"



filenames = []
num_objects = []
gt_array = np.zeros((LIST_LENGTH, MAX_NUM_OBJECTS, 4))

image_count = -1
i = 0
read_num_obj = False
next_num_objects = 0
with open(directory, "r") as f:
    for line in f:
        if ".jpg" in line:
            filenames.append(line.rstrip())
            read_num_obj = True
        elif read_num_obj == True:
            next_num_objects = int(line.rstrip())
            num_objects.append(next_num_objects)
            image_count += 1
            i = 0
            read_num_obj = False
        else:
            if i < MAX_NUM_OBJECTS:
                #parse line
                line = line.split()
                gt_array[image_count, i, 0] = float(line[0]) #xmin
                gt_array[image_count, i, 1] = float(line[1]) #ymin
                gt_array[image_count, i, 2] = float(line[0]) + float(line[2]) #xmax = xmin + width
                gt_array[image_count, i, 3] = float(line[1]) + float(line[3]) #ymax = ymin + height
                i += 1

print(filenames)
