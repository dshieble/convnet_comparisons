import numpy as np
import pandas as pd
from os import listdir
import tensorflow as tf



def initialize(sess, saved_weights_path=None):
	"""
		Initialize the variables of the model and load saved weights
	"""
	#Initialize the variables of the model and the saver object
	init = tf.global_variables_initializer()
	sess.run(init)
	tvars = tf.trainable_variables()
	saver = tf.train.Saver(tvars)
	if saved_weights_path is not None:
		#Load the saved weights
		print "restoring {} variables from {}".format(len(tvars), saved_weights_path)
		saver.restore(sess, saved_weights_path)
	else:
		print "not loading weights"
	return saver

def epoch_cleanup(params, sess, saver, epoch):
	"""
		Print the progress and save weights for the epoch
	"""
	print("Epoch: {} Train Costs: {} Val Costs: {}".format(epoch, params["train_costs"][-1], params["val_costs"][-1]))
	print("Train Accs: {} Val Accs: {}".format(params["train_accuracies"][-1], params["val_accuracies"][-1]))
	print("Train AUCs: {} Val AUCs: {}".format(params["train_aucs"][-1], params["val_aucs"][-1]))

	np.save("{}/PARAMETERS_epoch_{}.npy".format(params["ckpt_dir"], epoch), params)
	saved_weights_path = "{}/epoch_{}_{}.ckpt".format(params["ckpt_dir"], epoch, params["signature"])
	saver.save(sess, saved_weights_path)
	print("Saved for epoch {}!".format(epoch))
	return saved_weights_path

def get_cost_updt(params, output, y):
	"""
		Given the output of the network and the true labels of the data, get the value of the cost and the tensorflow update variable
	"""
	# Compute the cross entropy loss
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
	# p, t = tf.argmax(output, axis=1), tf.argmax(y, axis=1)

	# Add the weight decay term to the loss
	if params["regularize"]:
		regularization_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		cost += regularization_loss

	# Computes the gradients according to the adam optimizer, and then perform gradient clipping as appropriate
	opt_func = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
	gvs = opt_func.compute_gradients(cost)
	grads = [[tf.clip_by_value(grad, -params["clip"], params["clip"]), var] for grad, var in gvs if not grad is None]

	if params["print_progress"]:
		#If we set this flag, print the mean values of the cost, layer weights, and gradients. Helps spot zero activations and crazy gradient values.
		vars = tf.trainable_variables()
		#Print the cost of each epoch
		cost = tf.Print(cost, [cost], "COST")
		#Print the mean values of the gradients
		for i, g in enumerate(grads):
			cost = tf.Print(cost, [tf.reduce_mean(g[0])], "MEAN GRADIENT FOR {}".format(g[1].name))
		#Print the mean values of the weights
		for i, v in enumerate(vars):
			cost = tf.Print(cost, [tf.reduce_mean(v)], "MEAN VALUE OF {}".format(v.name))

	updt = opt_func.apply_gradients(grads)
	return cost, updt


def add_decay(params, var):
	"""
		Add weight decay to a variable
	"""
	weight_decay = tf.multiply(tf.nn.l2_loss(var), params["weight_decay"], name='weight_loss')
	tf.add_to_collection('losses', weight_decay)

