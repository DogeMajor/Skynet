#!/usr/bin/env python

import time
import random
#import scipy
import numpy as np
from numpy import linalg as la
import layer
import datetime
from datetime import date
from layer import timer
from DAO import DAO
from yahoo_dao import SingleClosingSeries
np.random.seed(int(time.time()))
from memory_profiler import profile
import datetime
import cProfile
import ast #For reading a txt-file!

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
        self.layers.append(layer.InputLayer(self.form[0]))
        for i in range(1,last_layer):
            self.layers.append(layer.HiddenLayer(self.form[i],[], []))
        self.layers.append(layer.OutputLayer(self.form[last_layer],[], []))

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

    def _weights_derivative(self, k, input_, dao_output):
        return np.outer(self._delta(k, input_, dao_output), self._kth_sum(k-1,input_))

    def _avg_weights_derivative(self, k):#This function takes the most time, since the function ABOVE is so slow!!!
        temp = (self._weights_derivative(k,input_,output_) for input_,output_ in self.dao.data)
        return np.sum(temp)/self._data_sz

    def _back_propagation(self, learning_rate, reg, momentum, old_weights, old_biases):
        weight = change = bias_change = 0.0
        form = self.form
        prev_layer = form[self.depth-2]
        prev_coeff = np.copy(self.weights), np.copy(self.biases)
        for k in xrange(self.depth-1,0,-1):
            prev_layer = form[k-1]
            weight = self.layers[k].weights
            change = learning_rate*self._avg_weights_derivative(k)
            self.layers[k].weights = np.add(weight, -change*(1-momentum) - (weight-old_weights[k])*momentum - reg*weight**3)
            bias = self.layers[k].biases
            if bias != []:
                bias_change = learning_rate*self._avg_delta(k)
                self.layers[k].biases = np.add(bias,-bias_change*(1-momentum) - (bias-old_biases[k])*momentum - reg*bias**3)
        return prev_coeff
        #OK!!! #Momentum added!!! ##regularisation added

    #@timer
    #@profile
    def learn_by_back_propagation(self, max_error, max_cycles, learning_rate, reg, momentum):
        error = self._avg_error()
        print(error, max_error)
        if error > max_error:
            error_progress = []
            old_coeff = np.copy(self.weights), np.copy(self.biases)
            for i in xrange(max_cycles):
                old_coeff = self._back_propagation(learning_rate, reg, momentum, old_coeff[0], old_coeff[1])
                if i %20==0 and i!=0:
                    error_change = self._avg_error()-error
                    print(error)
                    if error<max_error:
                        print(i)
                        break
                    elif (abs(error_change)<0.01*max_error and error>8*max_error) or error_change>50*max_error:
                        self._randomize()
                    error_progress.append(error_change)
                    error = self._avg_error()
            print('Error progress (step==20):', np.around(error_progress,4))
        else:
            print('Network is already trained enough!')
        #OK!!!

    def save_net(self, file_name):
        def converter(x): #Apparently you first have to convert np.arrays to a lists
            if x != []:
                return x.tolist()
            else:
                return []

        net_dict = {}
        net_dict['form'] = self.form
        net_dict['weights'] = [converter(item) for item in self.weights]
        net_dict['biases'] = [converter(item) for item in self.biases]
        with open(file_name, 'w') as nn:
            print('file opened')
            nn.write(str(net_dict))
        print('file ' + file_name + ' saved and closed')
        #OK

    @staticmethod
    def load_net(file_name):
        def inv_converter(x):
            if x != []:
                return np.array(x)
            else:
                return []

        with open(file_name, 'r') as nn:
            content = ast.literal_eval(nn.read())
            content['weights'] = [inv_converter(item) for item in content['weights']]
            content['biases'] = [inv_converter(item) for item in content['biases']]
            print(file_name, ' loaded!')
            return content
        print('file closed')

    def set_coefficients_from_file(self, file_name):
        content = Net.load_net(file_name)
        if self.form == content['form']:
            self._set_weights(content['weights'])
            self._set_biases(content['biases'])
        else:
            print('Nope!!, wrong dimensions in the file')


if __name__=='__main__':
    pass
