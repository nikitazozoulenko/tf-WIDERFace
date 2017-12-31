from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# def testop():
#     box_index = 0
#     cell_x = 2
#     cell_y = 5
#     one_hot0 = tf.one_hot(indices = box_index,
#                        depth = 2,
#                            on_value = cell_x,
#                            off_value = -1,
#                             axis = 0)
#     one_hot1 = tf.one_hot(indices = one_hot0,
#                            depth = 7,
#                             on_value = cell_y,
#                             off_value = -1,
#                             axis = -1)
#     one_hot2 = tf.one_hot(indices = one_hot1,
#                                        depth = 7,
#                                        on_value = 1.0,
#                                        off_value = 0.0,
#                                        axis = -1)
#     return one_hot0, one_hot1, one_hot2

def testop():
    box_index = 0
    cell_x = 2
    cell_y = 5
    one_hot0 = tf.one_hot(indices = cell_y,
                       depth = 7,
                           on_value = cell_x,
                           off_value = -1,
                            axis = 0)
    one_hot1 = tf.one_hot(indices = one_hot0,
                           depth = 7,
                            on_value = box_index,
                            off_value = -1,
                            axis = -1)
    one_hot2 = tf.one_hot(indices = one_hot1,
                                       depth = 2,
                                       on_value = 1.0,
                                       off_value = 0.0,
                                       axis = -1)
    return one_hot0, one_hot1, one_hot2

testop = testop()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    one_hot0, one_hot1, one_hot2 = sess.run(testop)
    print((one_hot0, one_hot1, one_hot2))
    print(one_hot2.shape)
    print(one_hot2[5,2,0])
