#!/usr/bin/env python

import time
import random
import scipy
import numpy as np
import Layer

class Net(object):

    def __init__(self, form, bias):
        #self.dao=DAO() #To be added
        self.form=form #A vector giving the dimensions of the layers
        self.layers=[]
        self.bias=bias
        self._set_layers()

    def _set_layers(self):
        for i in range(0, len(self.form)):
            self.layers.append(Layer.Layer(self.form[i],self.bias))

    def set_weights(self, weights):
        for i in range(0,len(weights)):
            self.layers[i].set_weights(weights[i])
        #Every Matrix of this tensor corresponds to the weights of a specific layer!

    def _randomize_weights(self):
        return "Not OK!!!!"

    def _get_weights(self):
        result=[]
        for item in self.layers:
            result.append(item._get_weights())
        return np.array(result)

    def derivative(self, input):
        result=[]
        for item in self.layers:
            result.append(item._derivative(input))
        return result

    def output(self, input):
        result=input
        for item in self.layers:
            result=item.output(result)
            #print(result)
        return result


'''
    def _sum(self, input):
        sum=[]
        for item in self.neurons:
            sum.append(item._sum(input))
        return sum

    def _activation(self, input):
        result=[]
        for i in range(0,self.length):
            result.append(self.neurons[i]._activation(input[i]))
        return result
'''



net=Net([3,3],0)

A=np.array([[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],
[[-.1,-.2,-.3],[-.4,-.5,-.6],[-.7,-.8,-.9]]])
net.set_weights(A)
#net.layers[0].set_weights(A[0])
#net.layers[1].set_weights(A[1])
print(net.output(np.array([1,1,1])))
#print(A[0][:][:])

netb=Net([3,2],0)
print(([3,2])[1])
B=np.array([[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],
[[-.1,-.2,-.3],[-.4,-.5,-.6]]])
netb.set_weights(B)
#net.layers[0].set_weights(A[0])
#net.layers[1].set_weights(A[1])
print(netb.output(np.array([1,1,1])))
