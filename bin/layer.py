#!/usr/bin/env python
import time
import numpy as np
import abc
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

class BaseLayer(object):
    __metaclass__  = abc.ABCMeta

    def __init__(self, length, biases = [], weights = []):
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

    @abc.abstractmethod
    def _sum(self, input_):
        raise NotImplementedError

    @abc.abstractmethod
    def _activation(self, sum_):
        raise NotImplementedError

    @abc.abstractmethod
    def _derivative(self, sum_):
        raise NotImplementedError

    #@timer
    def output(self, input_):
        temp = self._sum(input_)
        return self._activation(temp)


class InputLayer(BaseLayer):

    def __init__(self, length):
        super(InputLayer,self).__init__(length, [], [])

    def _sum(self, input_):
        return input_

    def _activation(self, sum_):
        return sum_

    def _derivative(self, sum_):
        return np.ones(len(sum_))
    #@timer


class HiddenLayer(BaseLayer):

    def __init__(self, length, biases, weights):
        super(HiddenLayer,self).__init__(length, biases, weights)

    def _sum(self, input_):
        sum_ = np.dot(self._weights, input_)
        if self.biases != []:
            sum_ = np.add(sum_,self._biases)
        return sum_

    def _activation(self, sum_):
        return 1.7159*np.tanh(0.6666*sum_)

    def _derivative(self, sum_):
        return 1.1427894*(1-(np.tanh(0.6666*sum_))**2)


class OutputLayer(BaseLayer):

    def __init__(self, length, biases, weights):
        super(OutputLayer,self).__init__(length, biases, weights)

    def _sum(self, input_):
        sum_ = np.dot(self._weights, input_)
        if self.biases != []:
            sum_ = np.add(sum_,self._biases) 
        return sum_

    def _activation(self, sum_):
        return 1.7159*np.tanh(0.6666*sum_)

    def _derivative(self, sum_):
        return 1.1427894*(1-(np.tanh(0.6666*sum_))**2)


if __name__=='__main__':
    pass
