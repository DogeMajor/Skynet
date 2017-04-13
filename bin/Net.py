#!/usr/bin/env python

import time
import random
#import scipy
import numpy as np
from numpy import linalg as la
import Layer
import datetime
from datetime import date
#from Layer import BaseLayer
#from Layer import InputLayer
#from Layer import HiddenLayer
#from Layer import OutputLayer
from Layer import timer
from DAO import DAO
#from FinanceDAO import FinanceDAO
#from FinanceService import FinanceService
from yahoo_dao import SingleClosingSeries
np.random.seed(int(time.time()))
from memory_profiler import profile
import datetime
#simport re

class Net(object):

    def __init__(self, form, dao):
        self.dao = dao
        self.form = form #A vector giving the dimensions of the layers
        self.layers = []
        self._depth = len(form)
        self._data_sz = self.dao.size
        self._set_layers()
        self._randomize()

    @property
    def depth(self):
        return self._depth

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
        self.layers.append(Layer.InputLayer(self.form[0]))
        for i in range(1,last_layer):
            self.layers.append(Layer.HiddenLayer(self.form[i],[], []))
        self.layers.append(Layer.OutputLayer(self.form[last_layer],[], []))

    def _set_weights(self, weights):
        for layer, weight_matrix in zip(self.layers, weights):
            layer.weights = weight_matrix

    def _set_biases(self, biases):
        for layer, bias_vector in zip(self.layers, biases):
            layer.biases = bias_vector

    def _randomize(self):
        # fancy iterating
        for i, (layer, size) in enumerate(zip(self.layers, self.form)):
            prevsize = self.form[i-1] if i>0 else self.dao.input_size
            layer.weights = np.random.uniform(-2, 2, (size, prevsize)) if i!=0 else []
            not_output_layer = i != self.depth-1
            bias = np.random.uniform(-2, 2, size) if  i!=0 else []
            layer.biases = bias

    def output(self, input_):
        result = input_
        for item in self.layers:
            result = item.output(result)
        return result

    def _kth_output(self, k, input_):
        result = input_
        for i in range(k+1):
            result = self.layers[i].output(result)
        return result

    def _kth_sum(self, k, input_):
        if k==-1:
            return input_
        else:
            temp = self._kth_output(k-1, input_)
            return self.layers[k]._sum(temp)

    def _kth_derivative(self,k, input_):
        temp = self._kth_sum(k,input_)
        return self.layers[k]._derivative(temp)
    #@timer
    def error(self, input_, dao_output):
        vec = self.output(input_)-dao_output
        return 0.5*np.dot(vec,vec)

    #@timer
    def _avg_error(self):
        temp = (self.error(input_, output_) for input_, output_ in self.dao.data)
        return np.sum(temp)/self._data_sz

    #@timer
    def _delta(self, k, input_, dao_output):
        derivative = self._kth_derivative(k, input_)
        output = self.output(input_)
        if k==self.depth-1:
            return (output-dao_output)*derivative
        else:
            return np.dot(self._delta(k+1, input_, dao_output), self.layers[k+1].weights)*derivative

    def _avg_delta(self, k):
        temp = (self._delta(k,input_,output_) for input_, output_ in self.dao.data)
        return np.sum(temp)/self._data_sz

    #@timer
    def _weights_derivative(self, k, input_, dao_output):
        return np.outer(self._delta(k, input_, dao_output), self._kth_sum(k-1,input_))

    #@timer
    def _avg_weights_derivative(self, k):#This function takes the most time, since the function ABOVE is so slow!!!
        temp = (self._weights_derivative(k,input_,output_) for input_,output_ in self.dao.data)
        return np.sum(temp)/self._data_sz

    def _back_propagation(self, learning_rate, reg, momentum, old_weights, old_biases):
        weight = change = bias_change = 0.0
        form = self.form
        prev_layer = form[self.depth-2]
        for k in xrange(self.depth-1,0,-1):
            prev_layer = form[k-1]
            weight = self.layers[k].weights
            change = learning_rate*self._avg_weights_derivative(k)
            self.layers[k].weights = np.add(weight, -change*(1-momentum) - (weight-old_weights[k])*momentum - reg*weight**3)
            bias = self.layers[k].biases
            if bias != []:
                bias_change = learning_rate*self._avg_delta(k)
                self.layers[k].biases = np.add(bias,-bias_change*(1-momentum) - (bias-old_biases[k])*momentum - reg*bias**3)
        #OK!!! #Momentum added!!! ##regularisation added

    @timer
    #@profile
    def learn_by_back_propagation(self, max_cycles, learning_rate, reg, momentum):
        error = self._avg_error()
        error_progress = []
        for i in xrange(max_cycles):
            old_weights = self.weights
            old_biases = self.biases
            self._back_propagation(learning_rate, reg, momentum, old_weights, old_biases)
            if i %20==0 and i!=0:
                error_change = self._avg_error()-error
                print(error)
                if error<10**-3:
                    print(i)
                    break
                #elif (abs(error_change)<0.001 and error>0.1) or error_change>0.05:
                    #self._randomize()
                error_progress.append(error_change)
                error = self._avg_error()
        print('Error progress (step==20):', np.around(error_progress,4))
        #OK!!!

    def save_net(self, file_name):
        #Apparently you first have to convert np.arrays to a lists
        net_dict = {}
        net_dict['form'] = self.form
        net_dict['weights'] = [item.tolist() for item in self.weights if item != []]
        net_dict['biases'] = [item.tolist() for item in self.biases if item != []]
        print(net_dict)
        with open(file_name, 'w') as nn:
            nn.write(repr(net_dict))
        nn.close()

    @staticmethod
    def load_net(file_name):
        with open(file_name) as nn:
            content = eval(nn.read())
            content['weights'] = [np.array(item) for item in content['weights']]
            content['biases'] = [np.array(item) for item in content['biases']]
            return content
        nn.close()

    def set_coefficients_from_file(self, file_name):
        content = Net.load_net(file_name)
        if self.form == content['form']:
            self._set_weights(content['weights'])
            self._set_biases(content['biases'])
        else:
            print('Nope!!, wrong dimensions in the file')


if __name__=='__main__':
    '''
    from xor_dao import XorDAO
    net_form=[2,3,1]
    net_weights=[np.array([[1,0],[0,1]]),
    np.array([[-.1,-.2,],[-.3,-.4,],[-.5,-.6,]]),np.array([[.7,.8,.9]])]
    net_biases=[np.array([0,0]),np.array([-1.3,-1.4,-1.5]),np.array([1.6])]
    netA = Net(net_form, XorDAO())
    netA._set_weights(net_weights)
    netA._set_biases(net_biases)
    print(netA._kth_output(-1,np.array([0,1])))
    print(netA._kth_output(0,np.array([0,1])))
    print(netA._kth_output(1,np.array([0,1])))
    print(netA._kth_sum(1,np.array([0,1])))
    print(netA._kth_derivative(0,np.array([0,1])))
    print(netA.error(np.array([0,1]),netA.output(np.array([0,1]))))
    print(netA.output(np.array([0,1])))
    a = netA._delta(1,np.array([0,1]),[1])
    print(netA.weights)
    print(netA.biases)
    netA._randomize()
    print(netA.weights)
    print(netA.biases)
    print('output:',netA.output(np.array([0,1])))
    #print(netA.weights)
    #netA.set_coefficients_from_file('xor_net')
    #print(netA.weights)
    netA.learn_by_back_propagation(200,0.5,10**-4,0.5/5)
    #netA.save_net('xor_net')
    #print(Net.load_net('xor_net'))
    '''

    start = '01/01/2016'
    end = '01/01/2017'
    #t = datetime.datetime(2012, 2, 23, 0, 0)
    #t.strftime('%m/%d/%Y')
    today = date.fromtimestamp(time.time())
    print(today)
    today = today.strftime('%m/%d/%Y')
    print(today)
    print(type(today))
    ccj = SingleClosingSeries('CCJ',start,today,20)
    ccj._norm_data()
    financeNet = Net([ccj.input_size, ccj.input_size,1],ccj)

    #financeNet.learn_by_back_propagation(1000,0.6,10**-4,0.6/4)
    financeNet.learn_by_back_propagation(3,0.5,10**-4,0.6/4)

    financeNet.learn_by_back_propagation(6,0.6, 10**-4, 0.6/4)
    #financeNet.save_net('ccj_net')

    input_ = ccj.input_data('01/01/2016','02/02/2016')
    output_ = list(ccj.inv_transform(financeNet.output(input_)))
    print(output_)
    print('real price change and simulated', +.30, output_)

    #print(financeNet._avg_error())

    random_results = ccj.random_data_dyads(20)
    print(ccj._stats)
    for input_, output_ in random_results:
        calculated_output_ = list(ccj.inv_transform(financeNet.output(input_)))
        print( list(ccj.inv_transform(output_)), calculated_output_)
    print(financeNet.biases)
