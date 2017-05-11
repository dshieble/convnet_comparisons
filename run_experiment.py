import tensorflow as tf
import sys
import numpy as np
import time
from tqdm import tqdm
import convnet_helpers as ch
import utility_functions as uf
from batcher import Batcher
from convnet import Convnet
import os
import argparse
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score



def build_graph(params, training):
	"""
		Build the convnet and the placeholder variables
	"""
	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, (None, 128, 128, 3), name="x") #the input variable
	y = tf.placeholder(tf.float32, (None, 2), name="y") #the output variable

	#Build the network and get the updatw
	network = Convnet(params, x, training=training)
	cost, updt = ch.get_cost_updt(params, network.pred, y)
	return x, y, network, cost, updt


def run_training(task, depth, rf_size, saved_weights_path, params):
	"""
		Train the neural network on the images in a directory
	"""

	#Set the parameters of the training to 
	if not "signature" in params:
		start_epoch = 0
		params["signature"] = time.time()
		params["weight_decay"] = 5e-4
		params["bsize"] = 50
		params["rf_size"] = rf_size
		params["depth"] = depth
		params["learning_rate"] = 1e-6
		params["clip"] = 10
		params["num_epochs"] = 100
		params["train_accuracies"] = []
		params["val_accuracies"] = []
		params["train_aucs"] = []
		params["val_aucs"] = []
		params["train_costs"] = []
		params["val_costs"] = []
		params["train_predictions"] = []
		params["val_predictions"] = []
		params["train_truth"] = []
		params["val_truth"] = []
		params["print_progress"] = True
	else:
		start_epoch = len(params["train_accuracies"])

	os.system("mkdir {}".format(params["ckpt_dir"]))


	B = Batcher("{}/{}.npy".format(params["base_data_split_directory"], task), params["bsize"])
	B.paths["train"] = B.paths["train"]
	B.paths["val"] = B.paths["val"]


	print "STARTING AT EPOCH {}".format(start_epoch)
	for epoch in range(start_epoch, params["num_epochs"]+start_epoch):
		B.reset()
		for kind in ["train", "val"]:
			print "DROPOUT", params["regularize"]
			x, y, network, cost, updt = build_graph(params, True if params["regularize"] else False)
			# Initialize the session
			with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
				#Initialize the variables
				print "Initializing for {}...".format(kind)
				saver = ch.initialize(sess, saved_weights_path)
				print "Initialized!"

				cost_list = []
				truth_list = []
				predictions_list = []
				num_batches = len(B.paths[kind])/B.bsize
				print "running {}!".format(kind)
				for X, Y  in tqdm(B.batch(kind), total=num_batches):
					#Run the session to get the network predictions
					if kind == "train":
						truth, predictions, costs, _ = sess.run([y, network.pred, cost, updt], feed_dict={x:X, y:Y})
					elif kind == "val":
						truth, predictions, costs = sess.run([y, network.pred, cost], feed_dict={x:X, y:Y})

					# Run softmax on the predictions (done separately from the cost computation for numerical reasons)
					# Compute the metrics of interest
					cost_list.append(np.mean(costs))
					truth_list += list(truth)
					predictions_list += list(predictions)


				# Aggregate the epoch-wise metrics

				predictions_list_vals = uf.softmax(predictions_list)
				params["{}_accuracies".format(kind)].append(np.mean(accuracy_score(np.argmax(truth_list, axis=1), np.argmax(predictions_list_vals, axis=1))))
				params["{}_costs".format(kind)].append(np.mean(cost_list))
				params["{}_aucs".format(kind)].append(np.mean(roc_auc_score(np.vstack(truth_list)[:,1], predictions_list_vals[:,1])))
				params["{}_predictions".format(kind)] += predictions_list
				params["{}_truth".format(kind)] += truth_list
				if kind == "train":
					#Save the weights so they can be loaded when the validation starts up
					saved_weights_path = "{}/temporary_weights.ckpt".format(params["ckpt_dir"])
					saver.save(sess, saved_weights_path) 
				elif kind == "val":
					#Perform epoch cleanup (weight saving, performance printing, etc)
					saved_weights_path = ch.epoch_cleanup(params, sess, saver, epoch)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='run an experiment.')
	parser.add_argument('task', default=None)
	parser.add_argument('depth', default=None)
	parser.add_argument('rf_size', default=None)
	parser.add_argument('--regularize', default=True)
	parser.add_argument('--Continue', default=False)
	parser.add_argument('--base_ckpt_dir', default="checkpoints_directory")
	parser.add_argument('--base_data_split_directory', default="train_test_splits_directory")

	args = parser.parse_args()
	task = sys.argv[1]
	depth = int(sys.argv[2])
	rf_size = int(sys.argv[3])

	#Make sure that specified paths are valid
	assert os.path.isdir(args.base_ckpt_dir)
	assert os.path.isdir(args.base_data_split_directory)

	# Whether to continue training from a previous training session
	ckpt_dir = "{}/depth_{}_rf_size_{}_task_{}".format(args.base_ckpt_dir, depth, rf_size, task)
	if not str(args.Continue).lower() == "false":
		files = uf.ls_function(ckpt_dir)
		epoch_files = [f for f in files if "epoch_" in f and ".ckpt.meta" in f]
		param_files = [f for f in files if "PARAMETERS_" in f and ".npy" in f]
		assert len(epoch_files) > 0
		assert len(param_files) > 0
		saved_weights_path = "{}/{}".format(ckpt_dir, max(epoch_files, key=lambda f:int(f.split("_")[1])).split(".meta")[0])
		params = np.load("{}/{}".format(ckpt_dir, max(param_files, key=lambda f:int(f.split("_")[2].split(".npy")[0])))).item()
	else:
		# Locations of data files and data splits
		saved_weights_path = None
		params = {}
		params["ckpt_dir"] = ckpt_dir
		params["base_data_split_directory"] = args.base_data_split_directory
		params["regularize"] = not str(args.Continue).lower() == "false"
	run_training(task, depth, rf_size, saved_weights_path, params)




