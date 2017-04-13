#!/usr/bin/env python

import sys
sys.path.append('../') #enables importing the Net -module

import unittest
import numpy as np
from Skynet.bin import Net
from Skynet.bin.DAO import DAO as xorDAO
from Skynet.bin.xor_dao import XorDAO

#test_data
net_form=[2,3,1]
net_weights=[np.array([[0.1,0.2],[0.3,0.4]]),
np.array([[-.1,-.2,],[-.3,-.4,],[-.5,-.6,]]),np.array([[.7,.8,.9]])]
net_biases=[np.array([1.1,1.2]),np.array([-1.3,-1.4,-1.5]),np.array([1.6])]

class NetTests(unittest.TestCase):

    def setUp(self):
        data = XorDAO()
        self.net=Net.Net(net_form, data)
        self.net._set_weights(net_weights)
        self.net._set_biases(net_biases)

    def test_kth_output(self):
        print('kth_output',list(self.net._kth_output(np.array([1,1]),1)))
        result = self.net._kth_output(np.array([1,1]),1)
        print('kth_output',list(result))
        self.assertEqual([-1.35240662, -1.51912296, -1.61196495],
         list(result))

    def test_output(self):
        self.assertEqual([-0.48525653219380138],
         list(self.net.output([1,1])))

    def test_derivation(self):
        self.assertEqual([[0.71157776258722261, 0.18070663892364858, 0.032383774341317895],
         [0.76452609796324622, 0.27681850027474364]],
         list(self.net._kth_derivative(0,np.array([1,1]))))
         #_kth_derivative(self,k, input_):

    def test_randomize_weights(self):
        self.net._randomize()
        self.assertNotEqual([-0.48525653219380138, -0.85040078770263161],
         list(self.net.output([1,1])))

    def tearDown(self):
        del self.net

if __name__=="__main__":
    unittest.main()
