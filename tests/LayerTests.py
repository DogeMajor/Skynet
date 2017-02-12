#!/usr/bin/env python

import sys
sys.path.append('../') #enables importing Layer -module

import unittest
import numpy as np
from Skynet.bin import Layer

class LayerTests(unittest.TestCase):

    def setUp(self):
        self.layer=Layer.Layer(3, 0)
        self.layer._set_neurons()
        self.layer.set_weights(np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]]))

    def test_activation(self):
        self.assertEqual([0.2913126124515909, 0.5370495669980353, 0.71629787019902447],
         self.layer._activation([0.3,0.6,0.9]))

    def test_sum(self):
        self.assertEqual([0.1,0.4,0.7], self.layer._sum(np.array([1,0,0])))

    def test_output(self):
        self.assertAlmostEqual([0.2913126124515909, 0.5370495669980353, 0.71629787019902447],
         self.layer.output(np.array([0,0,1])))

    def test_derivation(self):
        self.assertAlmostEqual([0.91513696182662918, 0.71157776258722283, 0.4869173611483415],
         self.layer.derivative(np.array([0,0,1])))

    def tearDown(self):
        del self.layer

if __name__=="__main__":
    unittest.main()
