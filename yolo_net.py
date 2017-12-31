from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from resnet import *

from PIL import Image, ImageDraw

<<<<<<< HEAD
batch_size = 4
S = 15

def inference(x, is_training = tf.constant(True), reuse = tf.constant(False)):
    # Nonex259x259x3 input image
    
    # Model used is ResNet-50, modified with less channels for face recognition
    x = resnet50_conv1_x(x, is_training, reuse)
    x = resnet50_conv2_x(x, is_training, reuse)
    x = resnet50_conv3_x(x, is_training, reuse)
    x = resnet50_conv4_x(x, is_training, reuse)
    x = resnet50_conv5_x(x, is_training, reuse)
    
    # x is now of shape [batch_size * 7 * 7 * 1024]
    with tf.variable_scope("yolo_layers", reuse = reuse):
        B = 2
        x = conv_wrapper(x, shape = [3,3,1024,128], strides = [1, 1, 1, 1], padding = "VALID")
            
        flat = tf.reshape(x, [-1, 6*6*128])
        dense = tf.layers.dense(inputs=flat, units=S*S*5*B)
        yolo_tensor = tf.reshape(dense, [-1, S, S, B*5])

        box_tensor = yolo_tensor[:,:,:,0:8]
        # confidence_tensor = tf.nn.sigmoid(yolo_tensor[:,:,:,8:10])
        confidence_tensor = yolo_tensor[:,:,:,8:10]
            
=======
batch_size = 128
S = 7

def inference(x, is_training = tf.constant(True), reuse = tf.constant(False)):
    #Nonex259x259x3 input image
    #model used is ResNet-18, modified to fit the tiny imagenet dataset
    with tf.variable_scope("yolo_resnet", reuse = reuse):
        with tf.variable_scope("conv1", reuse = reuse):
            with tf.variable_scope("h1_conv_bn", reuse = reuse):
                x = conv_wrapper(x, shape = [7,7,3,64], strides = [1, 2, 2, 1], padding = "VALID")
                x = tf.nn.max_pool(x, ksize = [1,3,3,1], strides = [1,2,2,1], padding = "VALID")
                x = bn_wrapper(x, is_training)
                x = tf.nn.relu(x)

        with tf.variable_scope("conv2_x", reuse = reuse):
            # 2 residual blocks, 64
            channels = 64
            with tf.variable_scope("residual_block_1", reuse = reuse):
                x = residual_block(x, channels, is_training)
            with tf.variable_scope("residual_block_2", reuse = reuse):
                x = residual_block(x, channels, is_training)

        with tf.variable_scope("conv3_x", reuse = reuse):
            # 2 residual blocks, 128
            channels = 128
            with tf.variable_scope("residual_block_1", reuse = reuse):
                x = residual_block_reduce_size(x, channels, is_training)
            with tf.variable_scope("residual_block_2", reuse = reuse):
                x = residual_block(x, channels, is_training)

        with tf.variable_scope("conv4_x", reuse = reuse):
            # 2 residual blocks, 192
            channels = 192
            with tf.variable_scope("residual_block_1", reuse = reuse):
                x = residual_block_reduce_size(x, channels, is_training)
            with tf.variable_scope("residual_block_2", reuse = reuse):
                x = residual_block(x, channels, is_training)

        with tf.variable_scope("conv5_x", reuse = reuse):
            # 2 residual blocks, 256
            channels = 256
            with tf.variable_scope("residual_block_1", reuse = reuse):
                x = residual_block_reduce_size(x, channels, is_training)
            with tf.variable_scope("residual_block_2", reuse = reuse):
                x = residual_block(x, channels, is_training)
        #x is now Nonex7x7x256
        with tf.variable_scope("yolo_layers", reuse = reuse):
            B = 2
            yolo_tensor = conv_wrapper(x, shape = [1,1,256,B*5], strides = [1, 1, 1, 1], padding = "VALID")

            box_tensor = yolo_tensor[:,:,:,0:8]
            confidence_tensor = tf.nn.sigmoid(yolo_tensor[:,:,:,8:10])
        
>>>>>>> origin
    return box_tensor, confidence_tensor

def iou(box1, box2):
    #input
    #box1: (xmin1, ymin1, xmax1, ymax1)
    #box2: (xmin2, ymin2, xmax2, ymax2)

    #output: float
    xmin = box1[0]
    ymin = box1[1]
    xmax = box1[2]
    ymax = box1[3]

    predxmin = box2[0]
    predymin = box2[1]
    predxmax = box2[2]
    predymax = box2[3]

    x0 = tf.maximum(xmin, predxmin)
    x1 = tf.minimum(xmax, predxmax)
    y0 = tf.maximum(ymin, predymin)
    y1 = tf.minimum(ymax, predymax)

    intersection_area = (x1-x0) * (y1-y0)
    pred_area = (predxmax - predxmin) * (predymax - predymin)
    gt_area = (xmax - xmin) * (ymax - ymin)
    iou = intersection_area / (gt_area + pred_area - intersection_area)

    return iou

def condition(batch_count, obj_idx, num_objects, loss, box_tensor, confidence_tensor, gt):
    return obj_idx < num_objects

<<<<<<< HEAD
def bound(tensor, minimum, maximum):
    return tf.maximum(tf.minimum(tensor, maximum-1), 0)

def body(batch_count, obj_idx, num_objects, batch_loss, box_tensor, confidence_tensor, gt):     
    #do shit
    batch_coord_loss = batch_loss[0]
    gt_conf = batch_loss[1]
=======
def body(batch_count, obj_idx, num_objects, batch_loss, box_tensor, confidence_tensor, gt):     
    #do shit
    batch_coord_loss = batch_loss[0]
    batch_confidence_loss = batch_loss[1]
>>>>>>> origin

    gt_box = gt[batch_count, obj_idx, 0:4]
    xmin = gt_box[0]
    ymin = gt_box[1]
    xmax = gt_box[2]
    ymax = gt_box[3]

    cell_y = tf.cast(tf.floor((ymin + ymax)/2 * S), tf.int32)
<<<<<<< HEAD
    cell_y = bound(cell_y, 0, S)
    cell_x = tf.cast(tf.floor((xmin + xmax)/2 * S), tf.int32)
    cell_x = bound(cell_x, 0, S)
    
=======
    cell_x = tf.cast(tf.floor((xmin + xmax)/2 * S), tf.int32)

>>>>>>> origin
    box0 = box_tensor[batch_count, cell_y, cell_x, 0:4] # [x, y, width, height]
    box1 = box_tensor[batch_count, cell_y, cell_x, 4:8]

    bndbox1_iou = iou(box1 = (box0[0], box0[1], box0[2]+box0[0], box0[3]+box0[1]),
                      box2 = gt_box)

    bndbox2_iou = iou(box1 = (box1[0], box1[1], box1[2]+box1[0], box1[3]+box1[1]),
                      box2 = gt_box)

    box_index = tf.cond(bndbox1_iou > bndbox2_iou, lambda: tf.constant(0), lambda: tf.constant(1))
    box_index = tf.cast(box_index, tf.int32)

    #extract data from gt and box_tensor
    gt_x = xmin
    gt_y = ymin
    gt_width = xmax-xmin
    gt_height = ymax-ymin
    x = box_tensor[batch_count, cell_y, cell_x, box_index*4]
    y = box_tensor[batch_count, cell_y, cell_x, box_index*4 + 1]
    width = box_tensor[batch_count, cell_y, cell_x, box_index*4 + 2]
    height = box_tensor[batch_count, cell_y, cell_x, box_index*4 + 3]

    #coord losses
<<<<<<< HEAD
    x_loss = tf.pow(x - gt_x, 2)
    y_loss = tf.pow(y - gt_y, 2)
    w_loss = tf.pow(width - gt_width, 2)
    h_loss = tf.pow(height - gt_height, 2)
    # x_loss = tf.losses.absolute_difference(gt_x, x)
    # y_loss = tf.losses.absolute_difference(gt_y, y)
    # w_loss = tf.losses.absolute_difference(gt_width, width)
    # h_loss = tf.losses.absolute_difference(gt_height, height)
    batch_coord_loss += (x_loss + y_loss + w_loss + h_loss)

    #if something is wrong then cell_x and cell_y is flipped
    one_hot0 = tf.one_hot(indices = cell_y,
                       depth = S,
                           on_value = cell_x,
                           off_value = -1,
                            axis = 0)
    one_hot1 = tf.one_hot(indices = one_hot0,
                           depth = S,
                            on_value = box_index,
                            off_value = -1,
                            axis = -1)
    obj_gt_conf = tf.one_hot(indices = one_hot1,
                                       depth = 2,
                                       on_value = 1.0,
                                       off_value = 0.0,
                                       axis = -1)

    gt_conf += obj_gt_conf
    
    #iterate
    obj_idx += 1
    return batch_count, obj_idx, num_objects, [batch_coord_loss, gt_conf], box_tensor, confidence_tensor, gt
=======
    x_loss = tf.reduce_sum(tf.pow(x - gt_x, 2))
    y_loss = tf.reduce_sum(tf.pow(y - gt_y, 2))
    w_loss = tf.reduce_sum(tf.pow(width - gt_width, 2))
    h_loss = tf.reduce_sum(tf.pow(height - gt_height, 2))
    batch_coord_loss += (x_loss + y_loss + w_loss + h_loss)

    #if something is wrong then cell_x and cell_y is flipped
    gt_conf = tf.constant(1.0)
    pred_conf = confidence_tensor[batch_count, cell_y, cell_x, box_index]
    batch_confidence_loss += -gt_conf * tf.log(pred_conf)
    
    #iterate
    obj_idx += 1
    return batch_count, obj_idx, num_objects, [batch_coord_loss, batch_confidence_loss], box_tensor, confidence_tensor, gt
>>>>>>> origin

def loss(box_tensor, confidence_tensor, gt, num_objects):
    #box_tensor is batch_size x S x S x 8       ### X, Y, WIDTH, HEIGHT, X, Y, WIDTH, HEIGHT
    #confidence_tensor is batch_size x S x S x 2
    #gt is (batch_size, num_objects, 5)     ### xmin, ymin, xmax, ymax, class prediction index
    #num_objects is [batch_size]
<<<<<<< HEAD
    alpha_coord = 4.0
    alpha_conf = 6.0

    coord_loss = tf.constant(0.0)
    confidence_loss = tf.constant(0.0)
    for batch_count in range(batch_size):
        #while loop
        batch_coord_loss = tf.constant(0.0)
        batch_gt_conf = tf.zeros([S,S,2], tf.float32)

        obj_idx = tf.constant(0)
        result = tf.while_loop(condition, body,
                               [batch_count, obj_idx, num_objects[batch_count, 0],
                               [batch_coord_loss, batch_gt_conf],
=======

    coord_loss = tf.constant(0.0)
    confidence_loss = tf.constant(0.0)

    for batch_count in range(batch_size):
        #while loop
        batch_coord_loss = tf.constant(0.0)
        batch_confidence_loss = tf.constant(0.0)

        obj_idx = tf.constant(0)
        result = tf.while_loop(condition, body,
                               [batch_count, obj_idx, 1,
                               [batch_coord_loss, batch_confidence_loss],
>>>>>>> origin
                               box_tensor, confidence_tensor, gt])
        batch_loss = result[3]

        batch_coord_loss += batch_loss[0]
<<<<<<< HEAD
        batch_gt_conf = batch_loss[1]
        batch_gt_conf = tf.minimum(batch_gt_conf, 1.0)
        
        batch_confidence_loss = tf.pow(batch_gt_conf - confidence_tensor[batch_count], 2)
        batch_confidence_loss += alpha_conf * batch_gt_conf * batch_confidence_loss
        batch_confidence_loss = tf.reduce_sum(batch_confidence_loss)

        coord_loss += batch_coord_loss
        confidence_loss += batch_confidence_loss

    coord_loss = coord_loss * alpha_coord / batch_size
    confidence_loss = confidence_loss / batch_size
    
=======
        batch_confidence_loss += batch_loss[1]

        coord_loss = batch_coord_loss
        confidence_loss = batch_confidence_loss

    alpha_coord = 3.0
    coord_loss = coord_loss * alpha_coord / batch_size
    confidence_loss = confidence_loss / batch_size
>>>>>>> origin
    tf.summary.scalar("coord_loss", coord_loss)
    tf.summary.scalar("confidence_loss", confidence_loss)

    total_loss = coord_loss + confidence_loss
    tf.summary.scalar("total_loss", total_loss)
    
    return total_loss

def non_max_suppression(boxes, confidences):
    #boxs is batch_size x S x S x 8       ### X, Y, WIDTH, HEIGHT, X, Y, WIDTH, HEIGHT
    #confidences is batch_size x S x S x 2

    ##ONLY WORKS FOR BATCH SIZE = 1, STILL 4D TENSOR

    #RESHAPE into the needed format
    #need shape  [y1, x1, y2, x2], where (y1, x1) and (y2, x2)
    #boxes: A 2-D float Tensor of shape [num_boxes, 4].
    b0y = tf.reshape(boxes[0,:,:,1], [-1])
    b1y = tf.reshape(boxes[0,:,:,5], [-1])
    y1 = tf.concat([b0y, b1y], 0)

    b0x = tf.reshape(boxes[0,:,:,0], [-1])
    b1x = tf.reshape(boxes[0,:,:,4], [-1])
    x1 = tf.concat([b0x, b1x], 0)

    b0h = tf.reshape(boxes[0,:,:,3], [-1])
    b1h = tf.reshape(boxes[0,:,:,7], [-1])
    y2 = tf.concat([b0y+b0h, b1y+b1h], 0)

    b0w = tf.reshape(boxes[0,:,:,2], [-1])
    b1w = tf.reshape(boxes[0,:,:,6], [-1])
    x2 = tf.concat([b0x+b0w, b1x+b1w], 0)

    boxes = tf.stack([y1, x1, y2, x2], axis = 1)

    #confidences
    b0conf = tf.reshape(confidences[0,:,:,0], [-1])
    b1conf = tf.reshape(confidences[0,:,:,1], [-1])
    scores = tf.concat([b0conf, b1conf], 0)

    #non-maximum suppression with tensorflows own implementation
    selected_indices = tf.image.non_max_suppression(boxes, scores, S*S*2)

    return boxes, scores, selected_indices

def process_boxes(boxes, scores, indices, threshold):
    boxes = boxes[indices]
    scores = scores[indices]
    scores = scores.reshape(boxes.shape[0], 1)
<<<<<<< HEAD
=======
    print("SHAAAAAAAAAAAAAAAAAPEEEEEEEEEEES")
    print(boxes.shape)
    print(scores.shape)
>>>>>>> origin
    combined_boxes = np.concatenate((boxes, scores), axis=1)

    processed_boxes = np.array([box for box in combined_boxes if box[4] > threshold])

    return processed_boxes

def draw_and_show_boxes(image_array, boxes, border_size, color):
    im = Image.fromarray((image_array[0,:,:,:]*255).astype(np.uint8))
    im.show()
    dr = ImageDraw.Draw(im)
<<<<<<< HEAD

    for box in boxes:
        coords = (box[0:4] * 259).astype(int)
=======
    
    for i in range(boxes.shape[1]):
        coords = (boxes[i, 0:4] * 259).astype(int)
>>>>>>> origin
        y0 = coords[0]
        x0 = coords[1]
        y1 = coords[2]
        x1 = coords[3]

        for j in range(border_size):
            final_coords = [x0+j, y0+j, x1-j, y1-j]
            dr.rectangle(final_coords, outline = color)
            
    im.show()
