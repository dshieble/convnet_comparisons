import sys
sys.path.append("../.") # Making the package pip installable is overkill, so this works for tests. just make sure to run py.test from within the tests directory
import mock
from mock import patch
import unittest
import numpy as np

import batcher



class TestBatcher(unittest.TestCase):
    

    def test_get_labels(self):
        labels = batcher.get_labels(["sample_0_90714", "sample_1_90714", "sample_0_90714"]) 
        assert np.all(labels[0] == np.array([1,0]))   
        assert np.all(labels[1] == np.array([0,1]))   
        assert np.all(labels[2] == np.array([1,0]))   





