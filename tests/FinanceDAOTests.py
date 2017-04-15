#!/usr/bin/env python

import sys
sys.path.append('../') #enables importing FinanceDAO -module

import unittest
import numpy as np
from Skynet.bin import FinanceDAO
#Achtung!!! AssertAlmostEqual does not know how to compare np.arrays,
#you have to convert them into lists first

class FinanceDAOTests(unittest.TestCase):

    def setUp(self):
        self.financedao = FinanceDAO.FinanceDAO('CCJ')

    def test_get_raw_data(self):
        self.assertEqual([[1,1],[0]],list(self.financedao.one_data_point(3)))

    def test_random_data_point(self):
        result = list(self.financedao.random_data_point())[1]
        self.assertTrue(result==[0] or result==[1])


    def tearDown(self):
        del self.financedao

if __name__=="__main__":
    unittest.main()

    #ccj = SingleClosingSeries('CCJ','01/01/2016','01/01/2017',20)
    tesla = SingleClosingSeries('TSLA','01/12/2016','01/04/2017',20)
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
    dada = tesla.input_data('01/01/2017','02/02/2017')
    print(dada)
    #print(len(dada))
    #print()
    print(list(tesla.data)[0])
    tesla._norm_data()
    one_data_point = list(tesla.data)[0]
    print(one_data_point)
    print(list(tesla.inv_transform(one_data_point)))

    #output = ccj.random_data_dyads(10)
    #print(list(output))
    #print(list(ccj.data))
    print('stats',tesla._stats)
    last_20 = tesla.latest_data()
    print(last_20)
    print(len(last_20))
