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

    def __init__(self, form):
        self.dao=DAO.DAO() #To be added
        self.form=form #A vector giving the dimensions of the layers
        self.layers=[]
        self.__set_layers()
        self._randomize()

    def __set_layers(self):
        last_layer=len(self.form)-1
        for i in range(0, last_layer):
            self.layers.append(Layer.Layer(self.form[i],[], []))
        self.layers.append(Layer.Layer(self.form[last_layer],[], []))

    def set_weights(self, weights):
        for i in range(0,len(weights)):
            self.layers[i].set_weights(weights[i])
        #Every Matrix of the tensor weights corresponds to the weights of a specific layer!

    def set_biases(self, biases):
        for i in range(0,len(biases)):
            self.layers[i].set_weights(biases[i])

    def _randomize(self):
        prevlayer=self.form[0]
        temp_matrix=[]
        temp_list=[]
        biases=[]

        for layer_index in range(0,len(self.form)):
            for row in range(0,self.form[layer_index]):
                for column in range(0, prevlayer):
                    temp_list.append(random.uniform(-2,2))
                temp_matrix.append(temp_list)
                if(layer_index!=len(self.form)-1):
                    biases.append(random.uniform(-2,2))
                temp_list=[]
            self.layers[layer_index].set_weights(np.array(temp_matrix))
            if(layer_index!=len(self.form)-1):
                self.layers[layer_index].set_biases(np.array(biases))
            else:
                self.layers[layer_index].set_biases(np.zeros(self.form[layer_index]))
            temp_matrix=[]
            biases=[]
            prevlayer=self.form[layer_index]
        #OK... I don't know how to make this any shorter, sorry guys
        #Randomizes weights uniformally in [-2,2]

    def _get_weights(self):
        result=[]
        for item in self.layers:
            result.append(item._get_weights())
        return np.array(result)
        #OK

    def _get_biases(self):
        result=[]
        for item in self.layers:
            result.append(item._get_biases())
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
        #OK

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

    def _stochastic_delta(self, k, row):
        delta=0
        for item in self.dao.data:
            delta+=self._delta(k,row,item[0],item[1])
        return delta/self.dao.size
        #OK

    def weights_derivative(self, k, row, column, input, dao_output):
        return self._delta(k, row, input, dao_output)*self.kth_sum(input, k-1)[column]

    def _stochastic_derivative(self, k, row, column):
        derivative=0
        for item in self.dao.data:
            derivative+=self.weights_derivative(k,row,column,item[0],item[1])
        return derivative/self.dao.size
        #OK

    def _stochastic_error(self):
        error=0.0
        for item in self.dao.data:
            error+=self.error(item[0],item[1])
        return error/self.dao.size
        #OK

    def back_propagation(self,learning,old_weights,old_biases):
        momentum=learning/5.0   #Tunable
        weight=change=0.0
        prev_layer=self.form[len(self.form)-2]
        for k in range(len(self.form)-1,-1,-1):
            if k!=0:
                prev_layer=self.form[k-1]
            for row in range(0, self.form[k]):
                for column in range(0, prev_layer):
                    weight=self.layers[k].weights[row][column]
                    change=learning*self._stochastic_derivative(k, row, column)
                    if abs(weight-change)<10:
                        self.layers[k].weights[row][column]-=change*(1-momentum)+(weight-old_weights[k][row][column])*momentum
                bias=self.layers[k].biases[row]
                change=learning*self._stochastic_delta(k, row)
                if abs(bias-change)<10 and k!=len(self.form)-1:
                    self.layers[k].biases[row]-=change*(1-momentum)+(bias-old_biases[k][row])*momentum
        #OK #Momentum added!!!

    def learn_by_back_propagation(self, cycles, learning):
        #old_weights=self._get_weights()
        error=self._stochastic_error()
        error_progress=[]
        force_rand=0
        for i in range(cycles):
            old_weights=self._get_weights()
            old_biases=self._get_biases()
            self.back_propagation(learning, old_weights, old_biases)
            if i %30==0 and i!=0:
                error_change=self._stochastic_error()-error
                print(error)
                print(error_change)
                if error_change>0.05:
                    self._randomize()
                #elif abs(error_change)<0.001:
                #    learning*=0.75
                error_progress.append(error_change)
                print(learning)
                error=self._stochastic_error()
        #OK  Randomizes weights whenever error is not converging to 0

    def learn_by_randomisation(self, cycles):
        weights=[]
        error=0
        for i in range(cycles):
            weights=self._get_weights()
            error=self._stochastic_error()
            self._randomize()
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

netA=Net([2,5,1])
input=[1,0]
dao_output=[1]

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

B=np.array([[[-8.69899859, -9.88685292],[ 8.37943205, -8.57870461]],[[-9.56749478,2.12896463],[4.61960524,5.14759242]],[[ 7.83274341, -8.00756242]]])

netA.set_weights(B)
'''
#C=np.array([[[0.94070597,0.93761745],[-1.02608829, -1.86169371]],[[4.57657845, 0.49799085],[4.44945028,-3.57209561]],[[-1.02338944,1.02579393]]])
#netA.set_weights(C)

print(netA._get_weights())
#netA.learn_by_randomisation(50)
for i in range(1):
    #print(netA._get_weights())

    #netA._randomize()
    print(netA._stochastic_error())

    #print(netA.output(np.array([0,0])))
    #print(netA.output(np.array([0,1])))
    #print(netA.output(np.array([1,0])))
    #print(netA.output(np.array([1,1])))
    #print(netA._stochastic_error())
    W=netA._get_weights()
    netA.learn_by_back_propagation(500,0.4)
    #netA.learn_by_back_propagation(100,0.9*np.exp(-i*0.05))
    #print(netA._stochastic_error())


    #print(netA._stochastic_error())
    #print(netA._get_weights())
print(netA.output(np.array([0,0])))
print(netA.output(np.array([0,1])))
print(netA.output(np.array([1,0])))
print(netA.output(np.array([1,1])))
#print(netA._stochastic_derivatives())
#print(netA._stochastic_derivative(1,0,0))
print(netA._get_weights())
print(netA._get_biases())

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
