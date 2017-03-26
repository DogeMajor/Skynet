#!/usr/bin/env python

import time
import random
import scipy
import numpy as np
from numpy import linalg as la
import Layer
from Layer import timer
from DAO import DAO
from FinanceDAO import FinanceDAO
from FinanceService import FinanceService
from yahoo_dao import SingleClosingSeries
np.random.seed(int(time.time()))
from memory_profiler import profile
import datetime

class Net(object):

    def __init__(self, form, dao):
        self.dao = dao
        self.form = form #A vector giving the dimensions of the layers
        self.layers = []
        self._set_layers()
        self._randomize()

    @property
    def depth(self):
        return len(self.form)

    @property
    def weights(self):
        result = [item.weights for item in self.layers]
        return result

    @property
    def biases(self):
        result = [item.biases for item in self.layers]
        return result

    def _set_layers(self):
        last_layer=self.depth-1
        for i in range(last_layer+1):
            self.layers.append(Layer.Layer(self.form[i],[], []))

    def _set_weights(self, weights):
        for layer, weight_matrix in zip(self.layers, weights):
            layer.weights = weight_matrix
        #For testing

    def _set_biases(self, biases):
        for layer, bias_vector in zip(self.layers, biases):
            layer.biases = bias_vector
        #For testing

    def _randomize(self):
        # fancy iterating
        for i, (layer, size) in enumerate(zip(self.layers, self.form)):
            prevsize = self.form[i-1] if i>0 else self.dao.input_size
            layer.weights = np.random.uniform(-2, 2, (size, prevsize)) if i!=0 else np.identity(size)
            not_output_layer = i != self.depth-1
            bias = np.random.uniform(-2, 2, size) if (not_output_layer and i!=0) else np.zeros(size)
            layer.biases = bias

    def _kth_derivative(self,k, input_):
        return self.layers[k]._derivative(self._kth_sum(input_,k))

    def output(self, input_):
        result = input_
        for item in self.layers:
            result = item.output(result)
        return result

    def _kth_output(self, input_, k):
        result = input_
        for i in range(k+1):
            result = self.layers[i].output(result)
        return result

    def _kth_sum(self, input_, k):
        if k==-1:
            return input_
        else:
            temp = self._kth_output(input_, k-1)
            return self.layers[k]._sum(temp)

    def error(self, input_, dao_output):
        return 0.5*la.norm(self.output(input_)-dao_output)

    #@timer
    def _avg_error(self):
        temp = (self.error(input_, output_) for input_, output_ in self.dao.data)
        return np.sum(temp)/self.dao.size

    def _delta(self, k, input_, dao_output):
        derivative = self._kth_derivative(k, input_)
        output = self.output(input_)
        if k==self.depth-1:
            return (output-dao_output)*derivative
        else:
            return np.dot(self._delta(k+1, input_, dao_output), self.layers[k+1].weights)*derivative

    def _avg_delta(self, k):
        delta = np.zeros(self.form[k])
        temp = (self._delta(k,input_,output_) for input_, output_ in self.dao.data)
        return np.sum(temp)/self.dao.size

    def _weights_derivative(self, k, input_, dao_output):
        return np.outer(self._delta(k, input_, dao_output), self._kth_sum(input_, k-1))

    def _avg_weights_derivative(self, k):
        derivative = np.zeros((self.form[k], self.form[k-1]))
        temp = (self._weights_derivative(k,input_,output_) for input_,output_ in self.dao.data)
        return np.sum(temp)/self.dao.size

    def _back_propagation(self,learning_rate,old_weights,old_biases):
        momentum = learning_rate/5   #Tunable
        reg = 1.e-6
        weight = change = 0.0
        prev_layer = self.form[self.depth-2]
        for k in range(self.depth-1,0,-1):
            prev_layer = self.form[k-1]
            weight = self.layers[k].weights
            change = learning_rate*self._avg_weights_derivative(k)
            self.layers[k].weights -= change*(1-momentum) + (weight-old_weights[k])*momentum + reg*weight**3
            bias = self.layers[k].biases
            change = learning_rate*self._avg_delta(k)
            self.layers[k].biases -= change*(1-momentum) + (bias-old_biases[k])*momentum + reg*bias**3
        #OK!!! #Momentum added!!! ##regularisation added

    @timer
    #@profile
    def learn_by_back_propagation(self, max_cycles, learning_rate):
        error = self._avg_error()
        error_progress = []
        for i in range(max_cycles):
            old_weights = self.weights
            old_biases = self.biases
            self._back_propagation(learning_rate, old_weights, old_biases)
            if i %30==0 and i!=0:
                error_change = self._avg_error()-error
                print(error)
                if error<10**-3:
                    print(i)
                    break
                elif (abs(error_change)<0.001 and error>0.1) or error_change>0.05:
                    self._randomize()
                error_progress.append(error_change)
                error = self._avg_error()
        print('Error progress (step==30):', np.around(error_progress,4))
        #OK!!!

if __name__=='__main__':

    from xor_dao import XorDAO
    net_form=[2,3,1]
    net_weights=[np.array([[0.1,0.2],[0.3,0.4]]),
    np.array([[-.1,-.2,],[-.3,-.4,],[-.5,-.6,]]),np.array([[.7,.8,.9]])]
    net_biases=[np.array([1.1,1.2]),np.array([-1.3,-1.4,-1.5]),np.array([1.6])]
    netA = Net(net_form, XorDAO())
    netA._set_weights(net_weights)
    netA._set_biases(net_biases)
    a = netA._delta(1,np.array([0,1]),[1])
    netA._randomize()
    netA.learn_by_back_propagation(100,.5)

    start = '01/01/2016'
    end = '01/01/2017'
    ccj = SingleClosingSeries('CCJ',start,end,20)
    ccj._norm_data()
    financeNet = Net([ccj.input_size, ccj.input_size,1],ccj)
    financeNet.learn_by_back_propagation(100,0.7)
    input_ = ccj.input_data('01/01/2016','02/02/2016')
    output_ = list(ccj.inv_transform(financeNet.output(input_)))
    print(output_)
    print('real price change and simulated', +.30, output_)
    print(financeNet._avg_error())
