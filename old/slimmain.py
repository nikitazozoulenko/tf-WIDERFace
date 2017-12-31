from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import threading
import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

from yolo_net import *
from WIDER_queue_train import get_next_train_batch
from WIDER_queue_val import get_next_val_batch

save_dir = "/hdd/Coding/tf-WIDERFace/savedir/"
learning_rate = 0.0001

image_array_op, gt_array_op, num_objects_op = get_next_train_batch()

#Create the model inference
with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(
    images,
    num_classes = dataset.num_classes,
    is_training = True)

#Define the scopes that you want to exclude for restoration
exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
variables_to_restore = slim.get_variables_to_restore(exclude = exclude)



#loss_op = loss(box_tensor, confidence_tensor, gt_array_op, num_objects_op)
#train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_op)


#selected_indices_op = non_max_suppression(box_tensor, confidence_tensor)

#summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())
    
    #Create the model inference
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(
            images,
            num_classes = dataset.num_classes,
            is_training = True)

    #Define the scopes that you want to exclude for restoration
    exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
    variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
    
    coord.request_stop()
    coord.join(threads)
