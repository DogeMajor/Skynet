#!/usr/bin/env python

import time
import random
import scipy
import numpy as np
from numpy import linalg as la
from FinanceDAO import FinanceDAO

class FinanceService(object):

    def __init__(self, instrument, days):
        self._start_date = '2017-01-01'
        self._end_date = '2017-03-01'
        self.data = []
        self.stats = []
        self.finance = FinanceDAO(instrument, days)
        self.all_data()

    @property
    def size(self):
        return len(self.data)

    @property
    def input_size(self):
        return len(self.data[0][0])

    def _closing_price(self, start_date, end_date):
        all_data=self.finance.get_raw_data(start_date, end_date)
        result = temp = []
        for item in all_data:
            temp.append(float(item['Close']))
        return np.array(result)
        #OK

    def _slice_price_into(self):
        price = self._closing_price(self._start_date, self._end_date)
        result = temp = []
        output_ = []
        input_ = []
        for i, item in enumerate(price):
            if (i)%(self.finance.days) == 0 and i != 0 and i < len(price):
                input_.append(np.array(temp))
                output_.append(np.array([price[i]]))
                temp = []
            temp.append(item)
        return map(list ,zip(input_, output_))
        #OK

    def _average_and_range(self, price_vector):
        mean = np.mean(price_vector[0])
        max_price = np.max(price_vector[0])
        min_price = np.min(price_vector[0])
        #variance = np.var(price_vector[0])
        delta = max_price-min_price
        return [mean, delta]

    def normed_price(self, price_vector):
        stats = self._average_and_range(price_vector)
        mean = stats[0]
        delta = stats[1]
        input_ = map(lambda item: (item-mean)/delta, price_vector)
        output_ = np.array(map(lambda item: (item-mean)/delta, price_vector[1]))
        return map(list ,zip(input_, [output_]))[0]
        #OK!!

    def real_price(self, price, stats):
        return stats[1]*price+stats[0]

    def all_data(self):
        price_data = self._slice_price_into()
        result = map(self.normed_price, price_data)
        stats = map(self._average_and_range, price_data)
        self.data = result
        self.stats = stats
        return result
        #OK!!!

if __name__=='__main__':
   dogefund = FinanceService('CCJ', 4)
   #print(dogefund.get_raw_data())
   #print(dogefund.service_slice_price_into(4))
   #print(dogefund.service_slice_price_into(10)[0][0])
   price = dogefund._slice_price_into()
   #print(price[0])
   print(dogefund.normed_price(price[0]))
   #print(price)
   dogefund.all_data()

   #print(dogefund.data[0][0])
   #print(dogefund.service_closing_price())
   #print(dogefund.size)
   print(dogefund.input_size)
   print((dogefund.data[0][1]))
   print(dogefund.data)
   print(dogefund.stats)
