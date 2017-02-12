#!/usr/bin/env python

import scipy
import numpy as np

class Neuron(object):

    def __init__(self, weights, bias):
        self.weights=weights
        self.bias=bias

    def set_weights(self, weights):
        self.weights=weights

    def get_weights(self):
        return weights

    def _activation(self, input):
        return np.tanh(input)

    def _sum(self, input):
        temp=0.0
        for i in range(0,len(self.weights)):
            temp+=self.weights[i]*input[i]
        return temp +self.bias

    def output(self, input):
        return self._activation(self._sum(input))

    def _derivative(self, input): #Only works for tanh obv.
        return 1-self._activation(self._sum(input))**2

'''
neuron=Neuron(np.array([0.1,0.2,0.3,0.4]), 1)
print(neuron._derivative([1,0,1,0]))
print(neuron._sum([1,0,1,0]))
'''
