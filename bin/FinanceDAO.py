 #!/usr/bin/env python

"""Does include both DAO and Service class in fact
"""
import time
from yahoo_finance import Share
import random
import scipy
import numpy as np
from numpy import linalg as la
#import pandas as pd
random.seed(time.time())

class FinanceDAO(object):

    def __init__(self, instrument):
        self.data=[]
        self._instrument = instrument
        self.service_all_data(4)

    @property
    def size(self):
        return len(self.data)

    @property
    def input_size(self):
        return len(self.data[0][0])

    @property
    def instrument(self):
        return self._instrument

    def get_raw_data(self):#takes one month of price data
        share = Share(self.instrument)
        result = share.get_historical('2017-01-01', '2017-03-01')
        print('Connecting to Yahoo finance...')
        return result
        #OK

    def service_closing_price(self):
        all_data=self.get_raw_data()
        result = temp = []
        for item in all_data:
            temp.append(float(item['Close']))
        return np.array(result)
        #OK

    def service_slice_price_into(self, days):
        price = self.service_closing_price()
        result = temp = []
        output_ = []
        input_ = []
        for i, item in enumerate(price):
            if (i)%(days) == 0 and i != 0 and i < len(price):
                #result.append([np.array(temp), np.array([price[i]])])
                input_.append(np.array(temp))
                output_.append(np.array([price[i]]))
                #print('i, price[i]', i ,price[i])
                temp = []
            temp.append(item)
        #print('input:', input_)
        #print('output:', output_)
        return map(list ,zip(input_, output_))
        #OK

    def service_normed_price(self, price_vector):
        mean = np.mean(price_vector[0])
        max_price = np.max(price_vector[0])
        min_price = np.min(price_vector[0])
        variance = np.var(price_vector[0])
        delta = max_price-min_price
        #print('Mean, variance, price_delta: ', mean, variance, delta)
        input_ = map(lambda item: (item-mean)/delta, price_vector)
        output_ = np.array(map(lambda item: (item-mean)/delta, price_vector[1])) 
        #print('Mean, variance, price_delta: ', mean, variance, delta)
        return map(list ,zip(input_, [output_]))[0]
        #OK!!

    def service_all_data(self, days):
        price_data = self.service_slice_price_into(days)
        result = map(self.service_normed_price, price_data)
        self.data = result
        return result
        #Not OK!!!!



if __name__=='__main__':
    dogefund = FinanceDAO('CCJ')
    #print(dogefund.get_raw_data())
    #print(dogefund.service_slice_price_into(4))
    #print(dogefund.service_slice_price_into(10)[0][0])
    price = dogefund.service_slice_price_into(4)
    #print(price[0])
    print(dogefund.service_normed_price(price[0]))
    #print(price)
    #dogefund.service_all_data(4)
    #print(dogefund.data)
    #print(dogefund.data[0][0])
    #print(dogefund.service_closing_price())
    #print(dogefund.size)
    print(dogefund.input_size)
    print((dogefund.data[0][1]))

'''
@property
def data(self):
    return self._data

@data.setter
def data(self, values):
    self._data = values
'''
