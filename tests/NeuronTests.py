#!/usr/bin/env python

import sys
sys.path.append('../') #enables importing Neuron -module

import unittest
import numpy as np
from Skynet.bin import Neuron

class NeuronTests(unittest.TestCase):

    def setUp(self):
        self.neuron=Neuron.Neuron(np.array([0.1,0.2,0.3]),1)

    def test_activation(self):
        self.assertEqual(0, self.neuron._activation(0))

    def test_sum(self):
        self.assertEqual(1.3, self.neuron._sum(np.array([1,1,0])))

    def test_output(self):
        self.assertAlmostEqual(0.86172315, self.neuron.output(np.array([1,1,0])))

    def test_derivation(self):
        self.assertAlmostEqual(0.257433197, self.neuron._derivative(np.array([1,1,0])))

    def tearDown(self):
        del self.neuron

if __name__=="__main__":
    unittest.main()
