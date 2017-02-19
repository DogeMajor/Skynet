#!/usr/bin/env python

import time
import random
import scipy
import numpy as np
from numpy import linalg as la
import Layer
import DAO
random.seed(time)

class Net(object):

    def __init__(self, form, bias):
        self.dao=DAO.DAO() #To be added
        self.form=form #A vector giving the dimensions of the layers
        self.layers=[]
        self.bias=bias
        self.__set_layers()
        self._randomize_weights()

    def __set_layers(self):
        for i in range(0, len(self.form)):
            self.layers.append(Layer.Layer(self.form[i],self.bias, []))

    def set_weights(self, weights):
        for i in range(0,len(weights)):
            self.layers[i].set_weights(weights[i])
        #Every Matrix of the tensor weights corresponds to the weights of a specific layer!

    def _randomize_weights(self):
        prevlayer=self.form[0]
        temp_matrix=[]
        temp_list=[]

        for layer_index in range(0,len(self.form)):
            for row in range(0,self.form[layer_index]):
                for column in range(0, prevlayer):
                    temp_list.append(random.uniform(-1,1))
                temp_matrix.append(temp_list)
                temp_list=[]
            self.layers[layer_index].set_weights(np.array(temp_matrix))
            temp_matrix=[]
            prevlayer=self.form[layer_index]
        #OK... I don't know how to make this any shorter, sorry guys
        #Randomizes weights uniformally in [-1,1]

    def _get_weights(self):
        result=[]
        for item in self.layers:
            result.append(item._get_weights())
        return np.array(result)
        #OK

    def derivative(self, input):
        result=[]
        for k in range(0, len(self.form)):
            result.append(list(self.layers[k].derivative(self.kth_sum(input,k))))
        return np.array(result)
        #OK

    def output(self, input):
        result=input
        for item in self.layers:
            result=item.output(result)
        return result
        #OK

    def kth_output(self, input, k):
        result=input
        for i in range(0, k+1):
            result=self.layers[i].output(result)
        return result
        #OK

    def kth_sum(self, input, k):
        if k==-1:
            return input
        else:
            temp=self.kth_output(input, k-1)
            return self.layers[k]._sum(temp)
        #OK

    def error(self, input, dao_output):
        return 0.5*la.norm(self.output(input)-dao_output)

    def _delta(self, k, row, input, dao_output):
        result=0.0
        derivative=self.derivative(input)
        output=self.output(input)
        if k==len(self.form)-1:
            result = (output[row]-dao_output[row])*derivative[k][row]
            return result
        else:
            for l in range(0, self.form[k+1]):
                result+=self.layers[k+1].weights[l][row]*derivative[k][row]*self._delta(k+1,l, input, dao_output)
            return result
        #OK, allthough some intermediate outputs are calculated many times I think

    def weights_derivative(self, k, row, column, input, dao_output):
        return self._delta(k, row, input, dao_output)*self.kth_sum(input, k-1)[column]

    def weights_derivatives(self, input, dao_output):
        prev_layer=self.form[0]
        result=[]
        matrix=[]
        temp_list=[]
        for k in range(0,len(self.form)):
            for row in range(0,self.form[k]):
                for column in range(0, prev_layer):
                    temp_list.append(self.weights_derivative(k,row,column,input, dao_output))
                matrix.append(np.array(temp_list))
                temp_list=[]
            result.append(np.array(matrix))
            matrix=[]
            prev_layer=self.form[k]
        return np.array(result)
        #OK


    def _stochastic_derivative(self):
        derivatives=self.weights_derivatives(self.dao.data[0][0],self.dao.data[0][1])*0
        for item in self.dao.data:
            derivatives=np.add(derivatives,self.weights_derivatives(item[0],item[1]))
        return derivatives/self.dao.size
        #OK

    def _stochastic_error(self):
        error=0.0
        for item in self.dao.data:
            error+=self.error(item[0],item[1])
        return error/self.dao.size
        #OK

    def back_propagation(self,learning):
        derivatives=self._stochastic_derivative()
        error=self._stochastic_error()
        for k in range(0,len(self.form)):
            matrix=self.layers[k]._get_weights()
            matrix=np.add(matrix, -learning*derivatives[k])
            self.layers[k].set_weights(matrix)
            derivatives=self._stochastic_derivative()
            error=self._stochastic_error()
        #OK

    def learn_by_back_propagation(self, cycles,learning):
        for i in range(cycles):
            self.back_propagation(learning)

    def learn_by_randomisation(self, cycles):
        weights=[]
        error=0
        for i in range(cycles):
            weights=self._get_weights()
            error=self._stochastic_error()
            self._randomize_weights()
            if error<self._stochastic_error():
                self.set_weights(weights)
        #OK

    def learn_by_exponential_back_propagation(self, cycles,learning, exponent):
        for i in range(cycles):
            self.back_propagation(learning*np.exp(-exponent*i))

    def learn_by_stochastic_back_propagation(self, cycles,learning):
        for i in range(cycles):
            self.back_propagation(learning)
            self.learn_by_randomisation(1)
        #OK




netA=Net([2,1],0)

#netA.set_weights(np.array([[[0.1,0.2],[0.3,0.4]],[[0.5,0.6]]]))
#print(netA._get_weights())
#print(netA.weights_derivatives([1,1], [0]))
#print(netA.kth_output(np.array([1,1]),0))
#print(netA.kth_output(np.array([1,1]),1))

#print(type(netA._get_weights()))
'''
print(netA._stochastic_error())
netA.learn_by_back_propagation(100,0.5)
print(netA._stochastic_error())
netA.learn_by_randomisation(100)
print(netA._stochastic_error())
netA.learn_by_back_propagation(100,.7)
print(netA._stochastic_error())
'''
'''
print(netA._stochastic_error())
for i in range(10):
    learning=2-(i+1)*0.2
    print(learning)
    #netA.set_weights(np.array([[[0.1,0.2],[0.3,0.4]],[[0.5,0.6]]]))
    netA.learn_by_back_propagation(100,learning)
    print(netA._stochastic_error())
'''
print(netA.output(np.array([0,0])))
print(netA.output(np.array([0,1])))
print(netA.output(np.array([1,0])))
print(netA.output(np.array([1,1])))
print(netA._stochastic_error())
netA.learn_by_back_propagation(1000,.3)

print(netA._stochastic_error())
print(netA.output(np.array([0,0])))
print(netA.output(np.array([0,1])))
print(netA.output(np.array([1,0])))
print(netA.output(np.array([1,1])))
'''
print(netA._stochastic_error())

netA.learn_by_exponential_back_propagation(200,2,-0.001)
print(netA._stochastic_error())
'''
'''
netA.set_weights(np.array([[[-0.79119655, -0.79119641],[-0.79119638, -0.79119658]],[[ 1.06612512,  0.77319679]]]))

for i in range(10):
    print(netA._get_weights())
    print(netA._stochastic_error())
    netA.learn_by_stochastic_back_propagation(1000,0.7)
    print(netA._stochastic_error())
    print(netA._get_weights())
'''
