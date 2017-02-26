#!/usr/bin/env python

import sys
sys.path.append('../') #enables importing the Net -module

import unittest
import numpy as np
from Skynet.bin import Net

#test_data
net_form=[3,2]
net_weights=np.array([[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],
[[-.1,-.2,-.3],[-.4,-.5,-.6]]])
net_biases=np.array([[1.1,1.2,1.3],[-1.4,-1.5]])

class NetTests(unittest.TestCase):

    def setUp(self):
        self.net=Net.Net(net_form)
        self.net.set_weights(net_weights)
        self.net.set_biases(0*net_biases)

    def test_output(self):
        self.assertAlmostEqual([-0.48525653219380138, -0.85040078770263161],
         list(self.net.output([1,1,1])))

    def test_derivation(self):
        self.assertAlmostEqual([[0.71157776258722261, 0.18070663892364858, 0.032383774341317895],
         [0.76452609796324622, 0.27681850027474364]],
         list(self.net.derivative(np.array([1,1,1]))))

    def test_randomize_weights(self):
        self.net._randomize()
        self.assertNotEqual([-0.48525653219380138, -0.85040078770263161],
         list(self.net.output([1,1,1])))

    def tearDown(self):
        del self.net

if __name__=="__main__":
    unittest.main()
