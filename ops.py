import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import warnings

def log_sum_exp(x, axis=1):
    m = tf.reduce_max(x, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m), axis=axis))

#the implements of leakyRelu
def lrelu(x, alpha= 0.2, name="LeakyReLU"):
    return tf.maximum(x , alpha*x)

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h= 2, d_w=2, stddev=0.02, spectural_normed=False,
           name="conv2d", padding='SAME'):
    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        if spectural_normed:
            conv = tf.nn.conv2d(input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def instance_norm(input, scope="instance_norm"):

    with tf.variable_scope(scope):

        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)

        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv

        return scale * normalized + offset

def weight_normalization(weight, scope='weight_norm'):

  """based upon openai's https://github.com/openai/generating-reviews-discovering-sentiment/blob/master/encoder.py"""
  weight_shape_list = weight.get_shape().as_list()
  if len(weight.get_shape()) == 2: #I think you want to sum on axis [0,1,2]
    g_shape = [weight_shape_list[1]]
  else:
    raise ValueError('dimensions unacceptable for weight normalization')

  with tf.variable_scope(scope):

    g = tf.get_variable('g_scalar', shape=g_shape, initializer = tf.ones_initializer())
    weight = g * tf.nn.l2_normalize(weight, dim=0)

    return weight

def de_conv(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], tf.float32, initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases

        else:
            return deconv

def avgpool2d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k ,1], strides=[1, k, k, 1], padding='SAME')

def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h * scale, w * scale))

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def fully_connect(input_, output_size, scope=None, stddev=0.02, spectural_normed=True,
                  bias_start=0.0, with_w=False):

  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size], tf.float32,
      initializer=tf.constant_initializer(bias_start))

    if spectural_normed:
        mul = tf.matmul(input_, spectral_norm(matrix))
    else:
        mul = tf.matmul(input_, matrix)
    if with_w:
      return mul + bias, matrix, bias
    else:
      return mul + bias

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3 , [x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])])

def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse=reuse, fused=True, updates_collections=None)

def Residual(x, output_dims=256, kernel=3, strides=1, residual_name='resi'):

    with tf.variable_scope('residual_{}'.format(residual_name)) as scope:

        conv1 = instance_norm(conv2d(x, output_dims, k_h=kernel, k_w=kernel, d_h=strides, d_w=strides, name="conv1"), scope='in1')
        conv2 = instance_norm(conv2d(tf.nn.relu(conv1), output_dims, k_h=kernel, k_w=kernel,
                                     d_h=strides, d_w=strides, name="conv2"), scope='in2')
        resi = x + conv2
        return tf.nn.relu(resi)

NO_OPS = 'NO_OPS'

def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, iteration= 1):

    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    # w = tf.reshape(w, [1, w.shape.as_list()[0] * w.shape.as_list()[1]])

    u = tf.get_variable("u", [1, w.shape.as_list()[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None

    for i in range(iteration):

        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = _l2normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = _l2normalize(u_)

    #real_sn = tf.svd(w, compute_uv=False)[...,0]
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    #Get the real spectral norm
    #real_sn_after = tf.svd(w_norm, compute_uv=False)[..., 0]

    #frobenius norm
    #f_norm = tf.norm(w, ord='fro', axis=[0, 1])

    #tf.summary.scalar("real_sn", real_sn)
    tf.summary.scalar("powder_sigma", tf.reduce_mean(sigma))
    #tf.summary.scalar("real_sn_afterln", real_sn_after)
    #tf.summary.scalar("f_norm", f_norm)

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
