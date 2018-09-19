from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size,is_training, scope='block'):
	#batch_norm和dropout需要指定是不是训练集，不确定在densenet函数中的参数设定能不能传递到子函数，所以又做了一次传递
	current = slim.batch_norm(current,is_training, scope=scope + '_bn')
	current = tf.nn.relu(current)
	current = slim.conv2d(current, num_outputs,kernel_size, padding='SAME',scope=scope + '_conv')
	current = slim.dropout(current, is_training=is_training, keep_prob=0.8 ,scope=scope + '_dropout')
	return current


def transition(current, num_outputs,is_training, scope='transition'):
	
	with slim.arg_scope([slim.batch_norm, slim.dropout],
							is_training=is_training):
		current = slim.batch_norm(current,is_training, scope=scope + '_bn')
		current = tf.nn.relu(current)
		current = slim.conv2d(current, num_outputs,  [1,1], padding='SAME',scope=scope + '_conv')
		current = slim.dropout(current, is_training=is_training, keep_prob=0.8 ,scope=scope + '_dropout')
		current = slim.avg_pool2d(current, [2,2], stride=2, scope=scope + '_pooling')
	return current
	
def block(net, layers, growth,is_training, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],is_training,
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],is_training,
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=10, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
	"""Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
	"""
	growth = 3
	compression_rate = 0.5
	layer_num = 3
	def reduce_dim(input_feature):
		return int(int(input_feature.shape[-1]) * compression_rate)

	end_points = {}

	with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
		with slim.arg_scope(bn_drp_scope(is_training=is_training,keep_prob=dropout_keep_prob)) as ssc:
			with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
				# 在进入block前进行2k 7×7的卷积
				end_point = 'conv0'
				#net = slim.conv2d(images, growth*2,  [3, 3], padding='SAME',scope='_conv0')
				net = slim.conv2d(images, growth*2,  [7, 7], padding='SAME',stride=2 ,scope='_conv0')
                net = slim.max_pool2d(net, [3,3], stride=2, scope='_pooling')
				end_points[end_point] = net
				# dense block 1
				end_point = 'block1'
				with tf.variable_scope('block1'):
					net=block(net,6,growth,is_training)
					net=transition(net,reduce_dim(net),is_training)
				end_points[end_point] = net
				# dense block 2
				end_point = 'block2'
				with tf.variable_scope('block2'):
					net=block(net,12,growth,is_training)
					net=transition(net,reduce_dim(net),is_training)
				end_points[end_point] = net
				# dense block 3
				end_point = 'block3'
				with tf.variable_scope('block3'):
					net=block(net,24,growth,is_training)
					net=transition(net,reduce_dim(net),is_training)
				end_points[end_point] = net
				# dense block 4
				end_point = 'block4'
				with tf.variable_scope('block4'):
					net=block(net,16,growth,is_training)
				end_points[end_point] = net
				net=slim.batch_norm(net,is_training, scope=scope + '_bn1')
				net = tf.nn.relu(net)
				# global avgpool
				end_point = 'global_pool'
				net=tf.reduce_mean(net,[7,7],keep_dims=True,)
				end_points[end_point] = net
		with slim.arg_scope(densenet_arg_scope(weight_decay=0.004)) as slg:
			logits = slim.conv2d(net, num_classes,  [1, 1], normalizer_fn=None,padding='SAME',scope='logits')
		logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
	end_points['Logits'] = logits
	end_points['Predictions'] = slim.softmax(logits, scope='Predictions')
	return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
