import tensorflow as tf

EPSILON = 0.00001

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def conv_wrapper(x, shape, strides, padding):
    weights = tf.get_variable("weights",
                              shape,
                              initializer = tf.contrib.layers.xavier_initializer_conv2d())
    biases = tf.get_variable("biases",
                             [shape[3]],
                             initializer = tf.constant_initializer(0.1))

    #variable_summaries(weights)
    #variable_summaries(biases)

    conv = tf.nn.conv2d(x,
                        weights,
                        strides = strides,
                        padding = padding)
    return conv + biases

def bn_wrapper(x, is_training):
    gamma = tf.get_variable("gamma",
                            [x.get_shape()[-1]],
                            initializer = tf.constant_initializer(1.0))
    beta = tf.get_variable("beta",
                            [x.get_shape()[-1]],
                            initializer = tf.constant_initializer(1.0))
    moving_mean = tf.get_variable("moving_mean",
                                  [x.get_shape()[-1]],
                                  initializer = tf.constant_initializer(0.0),
                                  trainable = False)
    moving_variance = tf.get_variable("moving_variance",
                                      [x.get_shape()[-1]],
                                      initializer = tf.constant_initializer(1.0),
                                      trainable = False)
    #variable_summaries(gamma)
    #variable_summaries(beta)
    #variable_summaries(moving_mean)
    #variable_summaries(moving_variance)
    
    return tf.cond(is_training,
                   lambda: bn_train_time(x, beta, gamma, moving_mean, moving_variance),
                   lambda: bn_test_time(x, beta, gamma, moving_mean, moving_variance))

def bn_train_time(x, beta, gamma, moving_mean, moving_variance):
    mean, variance = tf.nn.moments(x, axes = [0,1,2])
    ALPHA = 0.90
    op_moving_mean = tf.assign(moving_mean,
                               moving_mean * ALPHA + mean * (1-ALPHA))
    op_moving_variance = tf.assign(moving_variance,
                                   moving_variance * ALPHA + variance * (1-ALPHA))
    with tf.control_dependencies([op_moving_mean, op_moving_variance]):
        return tf.nn.batch_normalization(x,
                                         mean,
                                         variance,
                                         offset = beta,
                                         scale = gamma,
                                         variance_epsilon = EPSILON)

def bn_test_time(x, beta, gamma, moving_mean, moving_variance):
    return tf.nn.batch_normalization(x,
                                     moving_mean,
                                     moving_variance,
                                     offset = beta,
                                     scale = gamma,
                                     variance_epsilon = EPSILON)

<<<<<<< HEAD
def residual_block(x, C, is_training, reuse):
    res_block = residual_block_without_skip(x, C, is_training, reuse)
    res = x + res_block
    return tf.nn.relu(res)

def residual_block_without_skip(x, C, is_training, reuse):
    last_C = x.get_shape().as_list()[-1]
    
    with tf.variable_scope("h1_conv_bn", reuse = reuse):
        conv1 = conv_wrapper(x, shape = [1,1,last_C,C], strides = [1, 1, 1, 1], padding = "VALID")
        bn1 = bn_wrapper(conv1, is_training)
        relu1 = tf.nn.relu(bn1)
        
    with tf.variable_scope("h2_conv_bn", reuse = reuse):
        conv2 = conv_wrapper(relu1, shape = [3,3,C,C], strides = [1, 1, 1, 1], padding = "SAME")
        bn2 = bn_wrapper(conv2, is_training)
        relu2 = tf.nn.relu(bn2)
        
    with tf.variable_scope("h3_conv_bn", reuse = reuse):
        conv3 = conv_wrapper(relu2, shape = [1,1,C,C*4], strides = [1, 1, 1, 1], padding = "SAME")
        bn3 = bn_wrapper(conv3, is_training)
        
    return tf.nn.relu(bn3)

def residual_block_reduce_size(x, C, is_training, reuse):
    last_C = x.get_shape().as_list()[-1]
    
    with tf.variable_scope("h1_conv_bn", reuse = reuse):
        conv1 = conv_wrapper(x, shape = [1,1,last_C,C], strides = [1, 2, 2, 1], padding = "VALID")
        bn1 = bn_wrapper(conv1, is_training)
        relu1 = tf.nn.relu(bn1)
        
    with tf.variable_scope("h2_conv_bn", reuse = reuse):
        conv2 = conv_wrapper(relu1, shape = [3,3,C,C], strides = [1, 1, 1, 1], padding = "SAME")
        bn2 = bn_wrapper(conv2, is_training)
        relu2 = tf.nn.relu(bn2)
        
    with tf.variable_scope("h3_conv_bn", reuse = reuse):
        conv3 = conv_wrapper(relu2, shape = [1,1,C,C*4], strides = [1, 1, 1, 1], padding = "SAME")
        bn3 = bn_wrapper(conv3, is_training)

    return tf.nn.relu(bn3)
  
def resnet50_conv1_x(x, is_training, reuse):
    with tf.variable_scope("conv1", reuse = reuse):
        with tf.variable_scope("conv_bn_relu", reuse = reuse):
            x = bn_wrapper(x, is_training)
            x = conv_wrapper(x, shape = [7,7,3,64], strides = [1, 2, 2, 1], padding = "VALID")
        with tf.variable_scope("maxpool_bn_relu", reuse = reuse):
            x = tf.nn.max_pool(x, ksize = [1,3,3,1], strides = [1, 2, 2, 1], padding = "VALID")
            x = bn_wrapper(x, is_training)
            x = tf.nn.relu(x)
    return x
              
def resnet50_conv2_x(x, is_training, reuse):
    with tf.variable_scope("conv2_x", reuse = reuse):
        # 3 residual blocks, 64
        channels = 64
        with tf.variable_scope("residual_block_1", reuse = reuse):
            x = residual_block_without_skip(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_2", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_3", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
    return x
        
                
def resnet50_conv3_x(x, is_training, reuse):
    with tf.variable_scope("conv3_x", reuse = reuse):
        # 4 residual blocks, 128
        channels = 128
        with tf.variable_scope("residual_block_1", reuse = reuse):
            x = residual_block_reduce_size(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_2", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_3", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_4", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
    return x
                
def resnet50_conv4_x(x, is_training, reuse):
    with tf.variable_scope("conv4_x", reuse = reuse):
        # 6 residual blocks, 192
        channels = 192
        with tf.variable_scope("residual_block_1", reuse = reuse):
            x = residual_block_reduce_size(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_2", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_3", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_4", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_5", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_6", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
    return x
        
def resnet50_conv5_x(x, is_training, reuse):
    with tf.variable_scope("conv5_x", reuse = reuse):
        # 3 residual blocks, 256
        channels = 256
        with tf.variable_scope("residual_block_1", reuse = reuse):
            x = residual_block_reduce_size(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_2", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
        with tf.variable_scope("residual_block_3", reuse = reuse):
            x = residual_block(x, channels, is_training, reuse)
    return x

def residual_block(x, C, is_training, reuse = False):
    with tf.variable_scope("h1_conv_bn", reuse = reuse):
        conv1 = conv_wrapper(x, shape = [3,3,C,C], strides = [1, 1, 1, 1], padding = "SAME")
        bn1 = bn_wrapper(conv1, is_training)
    relu1 = tf.nn.relu(bn1)
    with tf.variable_scope("h2_conv_bn", reuse = reuse):
        conv2 = conv_wrapper(relu1, shape = [3,3,C,C], strides = [1, 1, 1, 1], padding = "SAME")
        bn2 = bn_wrapper(conv2, is_training)

    res = x + bn2
    return tf.nn.relu(res)

def residual_block_reduce_size(x, C, is_training, reuse = False):
    last_C = x.get_shape().as_list()[-1]
    with tf.variable_scope("h1_conv_bn", reuse = reuse):
        conv1 = conv_wrapper(x, shape = [3,3,last_C,C], strides = [1, 2, 2, 1], padding = "VALID")
        bn1 = bn_wrapper(conv1, is_training)
    relu1 = tf.nn.relu(bn1)
    with tf.variable_scope("h2_conv_bn", reuse = reuse):
        conv2 = conv_wrapper(relu1, shape = [3,3,C,C], strides = [1, 1, 1, 1], padding = "SAME")
        bn2 = bn_wrapper(conv2, is_training)

    return tf.nn.relu(bn2)
