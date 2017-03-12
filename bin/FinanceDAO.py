 #!/usr/bin/env python

import time
from yahoo_finance import Share
import random
import scipy
import numpy as np
from numpy import linalg as la
random.seed(time.time())

class FinanceDAO(object):

    def __init__(self, instrument, days):
        '''Takes two months of stockprice (instrument) data
        and slices it into components with length of days,
        then takes the price the next day as target output'''
        self.data=[]
        self._days = days
        self._instrument = instrument

    @property
    def days(self):
        return self._days

    @property
    def instrument(self):
        return self._instrument

    def get_raw_data(self, start_date, end_date):#takes one month of price data
        share = Share(self.instrument)
        result = share.get_historical(start_date, end_date)
        print('Connecting to Yahoo finance...')
        return result
        #OK

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
