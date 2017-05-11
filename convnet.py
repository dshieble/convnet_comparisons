import tensorflow as tf
import numpy as np
import convnet_helpers as ch

class Convnet:

	"""
		This class represents a standard convolutional neural network with sucessive convolutional, pooling, and relu layers, followed by a few 
		fully connected layers. The size of the receptive field (filters), and the number of layers are configurable based on the input.
	"""


	def __init__(self, params, x, training):
		"""
			Build the network
		"""
		self.params = params
		self.var_dict = {}
		# Set the size of the convolutional filters and number of layers based on the parameters specfied
		conv_shapes = [[params["rf_size"], params["rf_size"], 3, params["rf_size"]*params["rf_size"]]]
		for i in range(params["depth"] - 1):
			conv_shapes.append([params["rf_size"], params["rf_size"], params["rf_size"]*params["rf_size"], params["rf_size"]*params["rf_size"]])

		prev = x
		# Stack the convolutional layers
		for i, cshape in enumerate(conv_shapes):
			cname, pname = "conv_{}".format(i),"pool_{}".format(i)
			self.var_dict[cname] = self.conv_layer(prev, cshape, cname)
			self.var_dict[pname] = self.max_pool(self.var_dict[cname], pname)
			prev = self.var_dict[pname]
		# Flatten the convoltional output to feed it into the fully connected layers
		flat_shape = np.prod(prev.get_shape().as_list()[1:])
		flattened_pool = tf.reshape(prev, (-1, flat_shape))

		# Build the fully connected layers, and include dropout layers if this is a training network
		self.var_dict["fc1"]  =  self.fc_layer(flattened_pool, [flat_shape, 2048], "fc1")
		if training:
			self.var_dict["fc1"]= tf.nn.dropout(self.var_dict["fc1"], 0.5)

		self.var_dict["fc2"]  =  self.fc_layer(self.var_dict["fc1"], [2048, 2048], "fc2")
		if training:
			self.var_dict["fc2"]= tf.nn.dropout(self.var_dict["fc2"], 0.5)
		#Generate the prediction of the network (pre-softmax, for numerical reasons)
		self.pred  =  self.fc_layer(self.var_dict["fc2"], [2048, 2], "prediction", relu=False)


	def max_pool(self, bottom, name):
		"""
			Take the max pool 
		"""
		return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


	def conv_layer(self, inputLayer, shape, name):
		"""
			Connect the inputLayer to the output with a 2D convolution
		"""
		with tf.variable_scope(name) as scope:
			#Get the weights for the convolutional filter
			filt = self.get_conv_filter(shape)
			conv = tf.nn.conv2d(inputLayer, filt, [1, 1, 1, 1], padding='SAME')

			#Get the bias for the convolutional filter
			conv_biases = self.get_bias(shape)
			bias = tf.nn.bias_add(conv, conv_biases)
			output = tf.nn.relu(bias)
		return output

	def fc_layer(self, inputLayer, shape, name, relu=True):
		"""
			Connect the inputLayer to the output with an affine transform
		"""
		with tf.variable_scope(name) as scope:
			# Weight the variance of each layer's initialization based on the number of input neurons
			init = tf.contrib.layers.variance_scaling_initializer()
			W = tf.get_variable(name="weights", initializer=init, shape=shape)
			ch.add_decay(self.params, W)

			#Perform the affine transformation
			b = self.get_bias(shape)
			product = tf.matmul(inputLayer, W)
			output = tf.nn.bias_add(product, b) 
			#Relu is optional
			if relu:
				output = tf.nn.relu(output)

		return output

	def get_conv_filter(self, shape):
		"""
		Get the weights for the convolutional filter
		"""
		#Weight the variance of each layer's initialization based on the number of input neurons
		init = tf.contrib.layers.variance_scaling_initializer() 
		filt = tf.get_variable(name="filter", initializer=init, shape=shape)
		ch.add_decay(self.params, filt)
		return filt

	def get_bias(self, shape):
		"""
			Get the bias for a convolutional or fully connected layer
		"""
		init = tf.constant_initializer(value=np.zeros(shape[-1]), dtype=tf.float32)       
		bias = tf.get_variable(name="biases", initializer=init, shape=shape[-1])
		return bias
