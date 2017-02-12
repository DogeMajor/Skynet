#!/usr/bin/env python

import sys
sys.path.append('../') #enables importing Neuron -module

import unittest
import numpy as np
from Skynet.bin import Layer

class LayerTests(unittest.TestCase):

    def setUp(self):
        self.layer=Layer.Layer(np.array([0.1,0.2,0.3]),0)
        self.layer._set_neurons()
        self.layer.set_weights(np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]]))


    def test_activation(self):
        self.assertEqual(0, self.neuron._activation(0))

    def test_sum(self):
        self.assertEqual([0.1,0,0], self.layer._sum(np.array([1,0,0])))

    def test_output(self):
        self.assertAlmostEqual(0.86172315, self.layer.output(np.array([1,1,0])))

    def test_derivation(self):
        self.assertAlmostEqual(0.257433197, self.layer._derivative(np.array([1,1,0])))

    def tearDown(self):
        del self.layer

if __name__=="__main__":
    unittest.main()
