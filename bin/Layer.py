#!/usr/bin/env python

import time
import random
import scipy
import numpy as np
import Neuron

class Layer(object):

    def __init__(self, length, bias):
        self.length=length
        self.bias=bias
        self.neurons=[]

    def _set_neurons(self):
        for i in range(0,self.length):
            self.neurons.append(Neuron.Neuron([],self.bias))

    def set_weights(self, weights):
        for i in range(0,len(weights)):
            self.neurons[i].set_weights(weights[:][i])

    def _get_weights(self):
        result=[]
        for item in self.neurons:
            result.append()
        return self.weigths

    def _sum(self, input):
        sum=[]
        for item in self.neurons:
            sum.append(item._sum(input))
        return sum

    def _activation(self, input):
        result=[]
        for item in self.neurons:
            result.append(item._activation(input))
        return result

    def derivative(self, input):
        result=[]
        for item in self.neurons:
            result.append(item._derivative(input))
        return result

    def output(self, input):
        return self._activation(self._sum(input))


layer=Layer(3,0)#Bias is zero in order to make checking matrix ops. easier
layer._set_neurons()
#print(layer.__dict__)
layer.set_weights(np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]]))
print(layer._sum(np.array([1,0,0])))
print(layer._sum(np.array([0,1,0])))
print(layer._sum(np.array([0,0,1])))

'''
print(layer.output(np.array([1,0,0])))
print(layer.output(np.array([0,1,0])))
print(layer.output(np.array([0,0,1])))
'''
print(layer._activation(np.array([0,0,1])))
print(layer.derivative(np.array([0,0,1])))
print(layer._activation([0,0,1]))
print(layer.derivative([0,0,1]))
