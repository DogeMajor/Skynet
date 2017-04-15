#!/usr/bin/env python

import sys
sys.path.append('../') #enables importing the Net -module

import unittest
import numpy as np
import os
from os import walk
from Skynet.bin import net
from Skynet.bin.DAO import DAO as xorDAO
from Skynet.bin.xor_dao import XorDAO

#test_data
net_form=[2,3,1]
net_weights=[[],np.array([[-.1,-.2,],[-.3,-.4,],[-.5,-.6,]]),np.array([[.7,.8,.9]])]
net_biases=[np.array([0,0]),np.array([-1.3,-1.4,-1.5]),np.array([1.6])]

class NetTests(unittest.TestCase):

    def setUp(self):
        data = XorDAO()
        self.net=net.Net(net_form, data)
        self.net._set_weights(net_weights)
        self.net._set_biases(net_biases)

    def test_kth_sum(self):
        result = self.net._kth_sum(2,np.array([0,1]))
        self.assertEqual([-3.4262579077340112],list(result))

    def test_kth_output(self):
        result = self.net._kth_output(2,np.array([0,1]))
        self.assertEqual([-1.6806443156853279],list(result))

    def test_output(self):
        self.assertEqual([-1.6806443156853279],
         list(self.net.output([0,1])))

    def test_delta(self):
        result = self.net._delta(1,np.array([0,1]),np.array([1]))
        self.assertEqual([-0.041864029980198454, -0.034750357290093105, -0.027705400624613907],list(result))

    def test_kth_derivative(self):
        self.assertEqual([0.48001533365918669, 0.34864336550043484, 0.24707798057168281],
         list(self.net._kth_derivative(1,np.array([0,1]))))

    def test_weights_derivative(self):
        result = list(self.net._weights_derivative(1,np.array([1,1]),np.array([0])))
        self.assertEqual([-0.012151220391902841, -0.012151220391902841],list(result[1]))

    def test_randomize_weights(self):
        self.net._randomize()
        self.assertNotEqual([-1.6806443156853279],
         list(self.net.output([0,1])))

    def test_save_net(self):
        self.net.save_net('horse')
        files = []
        for (dirpath, dirnames, filenames) in walk('/'):
            files.extend(filenames)
        result =[item for item in files if item.startswith('horse')]
        length = len(result)
        print(length)
        with open('horse', 'r') as net_file:
            beginning = net_file.readline()
        expected_result = "{'weights': [[], [[-0.1, -0.2],"
        self.assertNotEqual(length,0)
        self.assertEqual(beginning[0:31], expected_result)

    def test_set_coefficients_from_file(self):
        self.net.save_net('horse')
        self.net._randomize()
        self.net.set_coefficients_from_file('horse')
        self.assertEqual(list(self.net.weights[1][0]), list(net_weights[1][0]))

    def tearDown(self):
        del self.net

if __name__=="__main__":
    unittest.main()
