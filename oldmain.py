from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import threading
import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

from yolo_net import *
from WIDER_queue import *

is_training = tf.placeholder(tf.bool)
image_array_op, gt_array_op, num_objects_op = get_next_batch()

box_tensor, confidence_tensor = inference(image_array_op, tf.constant(True))

loss_op = loss(box_tensor, confidence_tensor, gt_array_op, num_objects_op)
train_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss_op)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())

    for _ in range(300):
        loss, _ = sess.run([loss_op, train_op])
        print(loss)

    coord.request_stop()
    coord.join(threads)

# for _ in range(3):
    #     image, gt, num_objects = sess.run([image_array_op, gt_array_op, num_objects_op])
    #     show_image = (image[0]*255).astype(np.uint8)
    #     im = Image.fromarray(show_image)
    #     im.show()
    #     dr = ImageDraw.Draw(im)
    #     for i in range(num_objects[0,0]):
    #         coords = (gt[0, i, 0:4] * 259).astype(int)
    #         border_size = 5
    #         for j in range(border_size):
    #             coords_iter = (coords[0]+j, coords[1]+j, coords[2]-j, coords[3]-j)
    #             dr.rectangle(coords_iter, outline = "red")
    #     im.show()
