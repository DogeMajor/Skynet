#!/usr/bin/env python

# import time
import random
import scipy
import numpy as np
from pandas_datareader import get_data_yahoo
from dao import DAO

# random.seed(time)

class SingleClosingSeries(DAO):

    def __init__(self, ticker, insize=50):
        series = get_data_yahoo(ticker)['Close']
        self._data = series.diff()
        self.size = len(series)-insize
        self.input_size = insize
        self.output_size = 1

    def data_dyad(self):
        insize = self.input_size
        series = self._data
        i = random.randint(insize, series.size-1)
        input_ = series.iloc[i-insize:i].values
        output = series.iloc[i:i+1].values
        return input_, output

    def data_monad(self):
        insize = self.input_size
        series = self._data
        i = random.randint(insize, series.size)
        input_ = series.iloc[i-insize:i].values
        return input_

    def iter(self):
        insize = self.input_size
        series = self._data
        for i in range(insize, self.size-1):
            input_ = series.iloc[i-insize:i].values
            output = series.iloc[i:i+1].values
            yield input_, output
            