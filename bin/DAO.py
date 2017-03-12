#!/usr/bin/env python

"""Does include both DAO and Service class in fact
"""

import time
import random
import scipy
import numpy as np
from numpy import linalg as la

random.seed(time)

data=[[np.array([0.0,0.0]),np.array([-1.0])],[np.array([0.0,1.0]),np.array([1.0])],[np.array([1.0,0.0]),np.array([1.0])],[np.array([1.0,1.0]),np.array([-1.0])]]
#data=np.array([[[1.0,1.0],[0.0]],[[0.0,1.0],[1.0]]])

class DAO(object):

    def __init__(self):
        self.data=data
        #self.size=len(self.data)
        #self.input_size = len(data[0][0])


    @property
    def size(self):
        return len(self.data)

    @property
    def input_size(self):
        return len(self.data[0][0])

    def one_data_point(self, i):
        return np.array(self.data[i])

    def random_data_point(self):
        number=random.randint(0,len(self.data)-1)
        return np.array(self.data[number])



#print(dao.get_all_data())
#print(dao.service_closing_price())
if __name__=='__main__':

    dao=DAO()
    print(dao.data[0][0])
    print(dao.data)
    print(dao.size)
    print(dao.input_size)
    print(len(dao.data[0][1]))


'''
def get_all_data(self):#takes one month of price data
        yahoo = Share('CCJ')
        result=yahoo.get_historical('2017-01-01', '2017-02-01')
        return result

def service_closing_price(self):
        all_data=self.get_all_data()
        result=[]
        for item in all_data:
            result.append(float(item['Close']))
        return result
'''
