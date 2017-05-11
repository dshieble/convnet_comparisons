import sys
sys.path.append("../.") # Making the package pip installable is overkill, so this works for tests. just make sure to run py.test from within the tests directory
import mock
from mock import patch
import unittest
import numpy as np
import tensorflow

from convnet import Convnet
import convnet_helpers as ch


class TestConvnet(unittest.TestCase):
    

    @classmethod
    def setUpClass(self):
        self.params = {}
        self.params["rf_size"] = 3
        self.params["depth"] = 6
        self.params["weight_decay"] = 5e-4
        self.params["learning_rate"] = 0.001
        self.params["regularize"] = False
        self.params["print_progress"] = False
        self.params["clip"] = 10


    @mock.patch('tensorflow.train.Saver', return_value=mock.Mock())
    def test_initialize(self, mock_saver):
        # Test that the initialize method only loads weights when you want it to
        tensorflow.reset_default_graph()
        saver = ch.initialize(mock.Mock())
        assert not saver.restore.called
        saver = ch.initialize(mock.Mock(), "path")
        assert saver.restore.called        
 
    def test_convnet_naming(self):
        #Test that layers are name scoped properly
        tensorflow.reset_default_graph()
        network = Convnet(self.params, tensorflow.placeholder(tensorflow.float32, (None, 128, 128, 3), name="x"), training=True)
        for k, v in  network.var_dict.items():
            if "conv" in k:
                assert k in v.name
        tvars = tensorflow.trainable_variables()
        for t in tvars:
            assert "conv" in t.name or "fc" in t.name or "prediction" in t.name

    def test_convnet_dropout(self):
        #Test that dropout is performed correctly
        tensorflow.reset_default_graph()
        network = Convnet(self.params, tensorflow.placeholder(tensorflow.float32, (None, 128, 128, 3), name="x"), training=True)
        assert sum(["dropout" in v.name for v in network.var_dict.values()]) == 2
        tensorflow.reset_default_graph()
        network = Convnet(self.params, tensorflow.placeholder(tensorflow.float32, (None, 128, 128, 3), name="x"), training=False)
        assert sum(["dropout" in v.name for v in network.var_dict.values()]) == 0


    def test_gradient_connectivity(self):
        # Make sure that all trainable variables have gradients computed when we backprop from error
        tensorflow.reset_default_graph()
        network = Convnet(self.params, tensorflow.placeholder(tensorflow.float32, (None, 128, 128, 3), name="x"), training=True)
        y = tensorflow.placeholder(tensorflow.float32, (None, 2), name="y") #the output variable

        cost, updt = ch.get_cost_updt(self.params, network.pred, y)
        gvs = tensorflow.train.AdamOptimizer(learning_rate=self.params["learning_rate"]).compute_gradients(cost)
        gvnames = set([g[1] for g in gvs])
        for v in tensorflow.trainable_variables():
            assert v in gvnames






