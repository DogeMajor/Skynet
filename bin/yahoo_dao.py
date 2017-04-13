#!/usr/bin/env python

# import time
import random
import scipy
import numpy as np
from pandas_datareader import get_data_yahoo
from dao import DAO
import datetime
#import pandas_datareader.data as web
# random.seed(time)

class SingleClosingSeries(DAO):

    def __init__(self, ticker, start, end, input_size):
        series = get_data_yahoo(ticker,start, end )['Close']
        self._data = series.diff()
        self._data = self._data.dropna()#drops the NaN's
        self.size = len(series)-input_size
        self.input_size = input_size
        self.output_size = 1
        self._stats = (self._data.mean(), self._data.var(), self._data.max(), self._data.min())

    #Inverts the norming
    def inv_transform(self, data_):
        for point in data_:
            yield 4.0*point*self._stats[1] + self._stats[0]

    def _norm_data(self):
        data = self._data
        self._data = (data-data.mean())/(4*data.var())

    def data_dyad(self):
        input_size = self.input_size
        series = self._data
        i = random.randint(input_size, series.size-1)
        input_ = series.iloc[i-input_size:i].values
        output_ = series.iloc[i:i+1].values
        return input_, output_

    def random_data_dyads(self,amount):
        input_size = self.input_size
        series = self._data
        rand_generator = (random.randint(input_size, series.size-1) for i in range(amount))
        def result(iterable):
            for i in iterable:
                input_ = series.iloc[i-input_size:i].values
                output_ = series.iloc[i:i+1].values
                yield input_, output_
        return result(rand_generator)

    def latest_data(self):
        input_size = self.input_size
        series = self._data
        i = self.size-1
        input_ = series.iloc[i-input_size:i].values
        output_ = series.iloc[i:i+1].values
        result = input_[1:]
        result = np.append(result, output_)
        return result

    def data_monad(self):
        input_size = self.input_size
        series = self._data
        i = random.randint(input_size, series.size)
        input_ = series.iloc[i-input_size:i].values
        return input_

    def input_data(self, start, end):
        input_size = self.input_size
        series = self._data[start:end]
        input_ = series.values
        return input_

    @property
    def data(self):
        input_size = self.input_size
        series = self._data
        for i in range(input_size, self.size-1):
            input_ = series.iloc[i-input_size:i].values
            output_ = series.iloc[i:i+1].values
            yield np.array(input_), np.array(output_)

if __name__=='__main__':

    ccj = SingleClosingSeries('CCJ','01/01/2016','01/01/2017',20)
    '''
    data = list(ccj.data)[0:2]
    print(data)
    print(len(data[0]))
    print(ccj.size)
    print(ccj.input_size)
    ccj._norm_data()
    data = list(ccj.data)[0:2]
    #print(data)
    #print(len(data[0]))
    print(type(list(ccj.data)[0]))
    #print(len(ccj._data))
    '''
    #print(list(ccj.inv_transform([1,-1])))
    #print(ccj.data_monad())
    dada = ccj.input_data('01/01/2016','02/02/2016')
    print(dada)
    #print(len(dada))
    #print()
    ccj._norm_data()
    #output = ccj.random_data_dyads(10)
    #print(list(output))
    print(list(ccj.data))
    last_20 = ccj.latest_data()
    print(last_20)
    print(len(last_20))
