#!/usr/bin/env python

import scipy
import numpy as np

#Every ROW of the matrix corresponds to the weights of the specific neuron!

class Layer(object):

    def __init__(self, length, bias, weights):
        self.length=length
        self.bias=bias
        self.weights=weights

    def set_weights(self, weights):
        self.weights=np.array(weights)
        #Every ROW of the matrix corresponds to the weights of the specific neuron!

    def _get_weights(self):
        return self.weights

    def _sum(self, input):
        sum=np.dot(self.weights, input) #Soooo much easier than dealing with neurons
        sum+=self.bias*np.ones(self.length)
        return sum

    def _activation(self, sum):
        result=[]
        for i in range(0, self.length):
            result.append(1.0/(1.0+np.exp(-sum[i])))
        return np.array(result)

    def derivative(self, sum):
        result=[]
        #print('#neurons: ', len(self.weights))
        for i in range(0, self.length):
            result.append(np.exp(sum[i])/(1.0+np.exp(sum[i]))**2)
        return np.array(result)

    def output(self, input):
        return self._activation(self._sum(input))


#layer=Layer(3,0,[])#Bias is zero in order to make checking matrix ops. easier


#layer.set_weights(np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]]))

'''
print(layer._sum(np.array([0,0,1])))

print(np.array(layer.output(np.array([0,0,1]))))

print(layer._activation(layer._sum(np.array([0,0,1]))))
print(layer.derivative(layer._sum(np.array([0,0,1]))))
print(layer._activation(layer._sum(np.array([0,0,1]))))
print(layer.derivative(layer._sum([0,0,1])))
'''

'''
A=np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
B=np.array([1,0,0])
print(np.dot(A,B))
print(A)

print(layer.get_weights())

print(layer.output(np.array([0,0,1])))

print(layer._sum(np.array([1,1,1])))

print(np.array(layer.output(np.array([1,1,1]))))

print(layer._activation(layer._sum(np.array([1,1,1]))))
print(layer.derivative(layer._sum(np.array([1,1,1]))))

print(layer.derivative(layer._sum([1,1,1])))
'''
