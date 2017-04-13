#!/usr/bin/env python

import sys
sys.path.append('../') #enables importing Layer -module

import unittest
import numpy as np
from Skynet.bin import Layer
#Achtung!!! Assert(Almost)Equal does not know how to compare np.arrays,
#you have to convert them into lists first
## Every time almosEqual fails the error messages will claim they don't support the operation for lists
## However, this is false

class LayerTests(unittest.TestCase):

    def setUp(self):
        self.hlayer=Layer.HiddenLayer(3, np.array([0]), None)
        self.hlayer.weights = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        self.ilayer=Layer.InputLayer(3)
        self.olayer=Layer.OutputLayer(3, np.array([0]), None)
        self.olayer.weights = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])

    def test_activation(self):
         excpected_result = [0.33864333077063419, 0.65189569581749662, 0.92145008987432331]
         h_result = self.hlayer._activation(np.array([0.3,0.6,0.9]))
         i_result = self.ilayer._activation(np.array([0.3,0.6,0.9]))
         self.assertAlmostEqual(excpected_result, list(h_result))
         self.assertAlmostEqual([0.3,0.6,0.9], list(i_result))

    def test_sum(self):
        h_result = list(self.hlayer._sum(np.array([1,0,0])))
        o_result = list(self.olayer._sum(np.array([1,0,0])))
        self.assertAlmostEqual([0.1,0.4,0.7], h_result)
        self.assertAlmostEqual([0.1,0.4,0.7], o_result)

    def test_output(self):
        h_result = list(self.hlayer.output(np.array([0,0,1])))
        excpected_result = [0.33864333077063419, 0.65189569581749662, 0.92145008987432331]
        o_result = list(self.olayer.output(np.array([0,0,1])))
        self.assertAlmostEqual(h_result, excpected_result)
        self.assertAlmostEqual(h_result, o_result)

    def test_derivation(self):
        h_result = list(self.hlayer._derivative(self.hlayer._sum(np.array([0,0,1]))))
        o_result = list(self.olayer._derivative(self.olayer._sum(np.array([0,0,1]))))
        i_result = list(self.ilayer._derivative(self.ilayer._sum(np.array([0,0,1]))))
        excpected_result = [1.0982784043437055, 0.97784465565703038, 0.81323593034898334]
        self.assertAlmostEqual(h_result, excpected_result)
        self.assertAlmostEqual(h_result, o_result)
        self.assertAlmostEqual(i_result, [1,1,1])

    def tearDown(self):
        del self.hlayer
        del self.ilayer
        del self.olayer

if __name__=="__main__":
    unittest.main()
