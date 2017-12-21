from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import threading
import os
from PIL import Image
import xml.etree.ElementTree as ET

batch_size = 128

MAX_NUM_OBJECTS = 1968
LIST_LENGTH = 3226
directory = "/hdd/Data/"

images_filenames = []
im_num_objects = []
gt_unprocessed = np.zeros((LIST_LENGTH, MAX_NUM_OBJECTS, 4))

image_count = -1
i = 0
read_num_obj = False
next_num_objects = 0
with open(directory + "wider_face_split/wider_face_val_bbx_gt.txt", "r") as f:
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
    #read corresponding jpeg
    num_objects = im_num_objects[index]
    image = Image.open(directory + "WIDER_val/images/" + images_filenames[index])
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

    num_objects = np.array(num_objects, dtype = np.int32)
    
    return [image_array, gt_array, num_objects]

def create_enqueue_op(queue):
    #read gt and jpeg with PIL with a python function wrapper
    resized_image_array, gt_array, gt_num_objects = tf.py_func(python_function,
                [],
                [tf.float32, tf.float32, tf.int32])
    return queue.enqueue([resized_image_array, gt_array, [gt_num_objects]])

def create_queue():
    num_threads = 5
    # create the queue
    queue = tf.FIFOQueue(capacity=1000, shapes = [[259, 259, 3],[MAX_NUM_OBJECTS, 4], [1]], dtypes=[tf.float32, tf.float32, tf.int32])

    # create our enqueue_op for this queue
    enqueue_op = create_enqueue_op(queue)

    # create a QueueRunner and add to queue runner list, probably only need 1 thread
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, [enqueue_op] * num_threads))
    return queue

def get_next_val_batch():
    # create a queue and dequeue batch_size
    queue = create_queue()
    return queue.dequeue_many(batch_size)
