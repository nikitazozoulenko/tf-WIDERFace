from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import threading
import os
from PIL import Image
import xml.etree.ElementTree as ET

batch_size = 2

MAX_NUM_OBJECTS = 1968
LIST_LENGTH = 12880
directory = "/hdd/Data/"

images_filenames = []
im_num_objects = []
gt_unprocessed = np.zeros((LIST_LENGTH, MAX_NUM_OBJECTS, 4))

image_count = -1
i = 0
read_num_obj = False
next_num_objects = 0
with open(directory + "wider_face_split/wider_face_train_bbx_gt.txt", "r") as f:
    for line in f:
        if ".jpg" in line:
            images_filenames.append(line.rstrip())
            read_num_obj = True
        elif read_num_obj == True:
            next_num_objects = int(line.rstrip())
            im_num_objects.append(next_num_objects)
            image_count += 1
            i = 0
            read_num_obj = False
        else:
            if i < MAX_NUM_OBJECTS:
                #parse line
                line = line.split()
                gt_unprocessed[image_count, i, 0] = float(line[0]) #xmin
                gt_unprocessed[image_count, i, 1] = float(line[1]) #ymin
                gt_unprocessed[image_count, i, 2] = float(line[0]) + float(line[2]) #xmax = xmin + width
                gt_unprocessed[image_count, i, 3] = float(line[1]) + float(line[3]) #ymax = ymin + height
                i += 1

def python_function():
    #randomize which file to read
    index = np.random.randint(0, LIST_LENGTH)
    #index = 1
    #read corresponding jpeg
    num_objects = im_num_objects[index]
    image = Image.open(directory + "WIDER_train/images/" + images_filenames[index])
    im_width, im_height = image.size
    gt_array = np.zeros((MAX_NUM_OBJECTS, 4))
    gt_array[0:num_objects, 0:1] = gt_unprocessed[index, 0:num_objects, 0:1] / im_width
    gt_array[0:num_objects, 1:2] = gt_unprocessed[index, 0:num_objects, 1:2] / im_height
    gt_array[0:num_objects, 2:3] = gt_unprocessed[index, 0:num_objects, 2:3] / im_width
    gt_array[0:num_objects, 3:4] = gt_unprocessed[index, 0:num_objects, 3:4] / im_height

    image = image.resize((259,259))
    image_array = (np.asarray(image) / 255)

    image_array = image_array.astype(np.float32)
    gt_array = gt_array.astype(np.float32)


    np_arr = np.array(num_objects, dtype = np.int32)
    print(np_arr.dtype)

python_function()
