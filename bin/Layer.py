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
        super(InputLayer,self).__init__(length, None, None)

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
            sum_ = np.add(sum_,self._biases) ##This might be OK to include in fact!!!
        return sum_

    def _activation(self, sum_):
        return 1.7159*np.tanh(0.6666*sum_)

    def _derivative(self, sum_):
        return 1.1427894*(1-(np.tanh(0.6666*sum_))**2)


if __name__=='__main__':

    layer = HiddenLayer(3,[0,0,0],[])#Biases are all zero in order to make checking matrix ops. easier
    layer.weights = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
    layer.biases = np.zeros(3)
    #Uses property.setter!!!
    print(layer._sum(np.array([0,0,1])))
    print(np.array(layer.output(np.array([0,0,1]))))

    print(layer._activation(layer._sum(np.array([0,0,1]))))
    print(layer._derivative(layer._sum(np.array([0,0,1]))))



    #Uses property!

    print(layer._sum(np.array([1,1,1])))
    print(np.array(layer.output(np.array([1,1,1]))))

    print(layer._activation(layer._sum(np.array([1,1,1]))))
    print(layer._derivative(layer._sum(np.array([1,1,1]))))




    ilayer = InputLayer(3)#Biases are all zero in order to make checking matrix ops. easier


    #layer.weights = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
    #Uses property.setter!!!


    print(ilayer._sum(np.array([0,0,1])))

    print(np.array(ilayer.output(np.array([0,0,1]))))

    print(ilayer._activation(ilayer._sum(np.array([0,0,1]))))
    print(ilayer._derivative(ilayer._sum(np.array([0,0,1]))))
    print(ilayer._activation(ilayer._sum(np.array([0,0,1]))))
    print(ilayer._derivative(ilayer._sum([0,0,1])))

    #print(layer.weights)
    #Uses property!

    print(ilayer.output(np.array([0,0,1])))

    print(ilayer._sum(np.array([1,1,1])))

    print(np.array(ilayer.output(np.array([1,1,1]))))

    print(ilayer._activation(ilayer._sum(np.array([1,1,1]))))
    print(ilayer._derivative(ilayer._sum(np.array([1,1,1]))))

    print(ilayer._derivative(ilayer._sum([1,1,1])))

    '''
    layer = HiddenLayer(3,[0,0,0],[])#Biases are all zero in order to make checking matrix ops. easier


    layer.weights = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
    #Uses property.setter!!!


    print(layer._sum(np.array([0,0,1])))

    print(np.array(layer.output(np.array([0,0,1]))))

    print(layer._activation(layer._sum(np.array([0,0,1]))))
    print(layer._derivative(layer._sum(np.array([0,0,1]))))
    print(layer._activation(layer._sum(np.array([0,0,1]))))
    print(layer._derivative(layer._sum([0,0,1])))

    print(layer.weights)
    #Uses property!

    print(layer.output(np.array([0,0,1])))

    print(layer._sum(np.array([1,1,1])))

    print(np.array(layer.output(np.array([1,1,1]))))

    print(layer._activation(layer._sum(np.array([1,1,1]))))
    print(layer._derivative(layer._sum(np.array([1,1,1]))))

    print(layer._derivative(layer._sum([1,1,1])))
    '''
    olayer = OutputLayer(3,[0,0,0],[])#Biases are all zero in order to make checking matrix ops. easier

    olayer.weights = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
    olayer.biases = np.zeros(3)
    print(olayer._sum(np.array([0,0,1])))
    print(np.array(olayer.output(np.array([0,0,1]))))

    print(olayer._activation(olayer._sum(np.array([0,0,1]))))
    print(olayer._derivative(olayer._sum(np.array([0,0,1]))))

    print(olayer._sum(np.array([1,1,1])))
    print(np.array(olayer.output(np.array([1,1,1]))))

    print(olayer._activation(olayer._sum(np.array([1,1,1]))))
    print(olayer._derivative(olayer._sum(np.array([1,1,1]))))
    print(type(olayer.weights))
    print(type(olayer.biases))
    print(type(layer.weights))
    print(type(layer.biases))
    print(type(ilayer.weights))
    print(type(ilayer.biases))
