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

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
decay_steps = 100
decay_rate = 0.8

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           decay_steps, decay_rate, staircase=True)
tf.summary.scalar("learning_rate", learning_rate)
# Passing global_step to minimize() will increment it at each step.

image_array_op, gt_array_op, num_objects_op = get_next_train_batch()
box_tensor, confidence_tensor = inference(image_array_op, tf.constant(True), False)
loss_op = loss(box_tensor, confidence_tensor, gt_array_op, num_objects_op)
train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_op, global_step = global_step)

val_image_array_op, val_gt_array_op, val_num_objects_op = get_next_val_batch()
val_box_tensor, val_confidence_tensor = inference(val_image_array_op, tf.constant(False), True)
val_loss_op = loss(val_box_tensor, val_confidence_tensor, val_gt_array_op, val_num_objects_op)

selected_indices_op = non_max_suppression(box_tensor, confidence_tensor)

summary_op = tf.summary.merge_all()
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(save_dir+"logs/", sess.graph)
    
    num_epochs = 40000
    for epoch in range(num_epochs+1):
        summary, _ = sess.run([summary_op, train_op])
        train_writer.add_summary(summary, epoch)   
        if epoch % decay_steps == 0:
            saver.save(sess, save_dir+"checkpoints/model_it"+str(epoch)+".ckpt")

    image, box_results = sess.run([image_array_op, selected_indices_op])
    boxes, scores, indices = box_results

    print(boxes, scores)

    processed_boxes = process_boxes(boxes, scores, indices, 0.3)
    print(processed_boxes)
    draw_and_show_boxes(image, processed_boxes, 2, "blue")
    
    coord.request_stop()
    coord.join(threads)
