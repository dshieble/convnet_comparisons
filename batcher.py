import utility_functions as uf


import numpy as np


def get_labels(fnames):
	"""
		The 0 and 1 labels are coded onto the data files
	"""
	one_zero = [int(f.split("/")[-1].split("_")[1]) for f in fnames]
	output = np.zeros((len(fnames), 2))
	for i, oz in enumerate(one_zero):
		output[i, oz] = 1
	return output
		
class Batcher:

	"""
		This class manages loading the data (saved as files, not tfrecords) and organizing into training/validation batches
	"""

	def __init__(self, data_splits_filename, bsize):
		"""
			Initialize the batcher object
		"""
		self.bsize = bsize
		self.paths = np.load(data_splits_filename).item()
		# self.paths = np.load("/gpfs/data/tserre/data/dan_data/train_test_splits_{}/{}.npy".format(split_type, data_dir)).item()
		# Make sure that there aren't any files overlapping between the training and testing sets
		assert len(set(self.paths["train"]).intersection(set(self.paths["val"]))) == 0
		self.pointers = {"train":0, "val":0}
		self.is_empty = {"train":False, "val":False}

	def get_batch(self, kind):
		# Load a training or validation batch from disk
		filepaths = self.paths[kind][self.pointers[kind]:self.pointers[kind]+self.bsize]
		self.pointers[kind] += self.bsize
		return uf.files_to_images(filepaths), get_labels(filepaths)

	def reset(self):
		# Reset the training and validation counters
		self.pointers = {"train":0, "val":0}
		self.is_empty= {"train":False, "val":False}

	def batch(self, kind):
		# Forms a generator for training or validation batches
		while self.pointers[kind] < len(self.paths[kind]):
			yield self.get_batch(kind)





