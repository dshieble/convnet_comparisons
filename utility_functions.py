import os
from os import listdir
from scipy.misc import imread
import numpy as np

def ls_function(mypath):
	"""
		List all of the files in a directory
	"""
	assert os.path.isdir(mypath)
	return [f for f in listdir(mypath) if not f == '.DS_Store']


def softmax(x):
	"""
		Compute softmax values for each sets of scores in x. Useful for analyzing results. 
	"""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


def files_to_images(files):
	"""
		Leads the files in a list of filenames into numpy images
	"""
	images = []
	for f in files:
		if "npy" in f:
			I = np.load(f)
		else:
			I = imread("{}".format(f))
		images.append(I)#process_image(I, size)) 
	return images





