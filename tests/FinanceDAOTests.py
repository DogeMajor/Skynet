#!/usr/bin/env python

import sys
sys.path.append('../') #enables importing FinanceDAO -module

import unittest
import scipy
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
