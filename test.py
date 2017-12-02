from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def testop():
    box_index = 0
    cell_x = 0
    cell_y = 3

    one_hot1 = tf.one_hot(indices = cell_x,
                          depth = 7,
                          on_value = cell_y,
                          off_value = -1,
                          axis = -1)

    indices = [ 3, -1, -1, -1, -1, -1, -1]
    one_hot2 = tf.one_hot(indices = indices, depth = 7)

    return one_hot1, one_hot2
testop = testop()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    result = sess.run(testop)
    print(result)
