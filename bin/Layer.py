#!/usr/bin/env python
import time
import scipy
import numpy as np
import math
#Every ROW of the matrix corresponds to the weights of the specific neuron!

def timer(function):

    def timer_wrapper(*args, **kwargs):
        t0 = time.time()
        result = function(*args,**kwargs)
        delta_t = time.time()-t0
        print('Function {} took {} seconds to run.'.format(function.__name__, delta_t))
        return result
    return timer_wrapper
    #Defines a timer decorator

class Layer(object):

    def __init__(self, length, biases, weights):
        self._length = length
        self._biases = biases
        self._weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, values):
        self._weights = values

    @property
    def biases(self):
        return self._biases

    @biases.setter
    def biases(self, values):
        self._biases = values

    def _sum(self, input_):
        sum_ = np.dot(self._weights, input_) #Soooo much easier than dealing with neurons
        sum_ = np.add(sum_,self._biases)
        return sum_

    def _activation(self, sum_):
        return 1.7159*np.tanh(0.666*sum_)
        #return result
    #result = (1.0/(1.0+np.exp(-sum_)))

    def _derivative(self, sum_):
        return 1.1427894*(1-(np.tanh(0.666*sum_))**2)
        #return result
    #result. = np.exp(sum_)/(1.0+np.exp(sum_))**2)

    def output(self, input_):
        temp = self._sum(input_)
        return self._activation(temp)

if __name__=='__main__':

    layer = Layer(3,[0,0,0],[])#Biases are all zero in order to make checking matrix ops. easier


    layer.weights = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
    #Uses property.setter!!!


    print(layer._sum(np.array([0,0,1])))

    print(np.array(layer.output(np.array([0,0,1]))))

    print(layer._activation(layer._sum(np.array([0,0,1]))))
    print(layer.derivative(layer._sum(np.array([0,0,1]))))
    print(layer._activation(layer._sum(np.array([0,0,1]))))
    print(layer.derivative(layer._sum([0,0,1])))

    print(layer.weights)
    #Uses property!

    print(layer.output(np.array([0,0,1])))

    print(layer._sum(np.array([1,1,1])))

    print(np.array(layer.output(np.array([1,1,1]))))

    print(layer._activation(layer._sum(np.array([1,1,1]))))
    print(layer.derivative(layer._sum(np.array([1,1,1]))))

    print(layer.derivative(layer._sum([1,1,1])))
