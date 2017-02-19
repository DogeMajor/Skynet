#!/usr/bin/env python

import sys
sys.path.append('../') #enables importing DAO -module

import unittest
import scipy
import numpy as np
from Skynet.bin import DAO
#Achtung!!! AssertAlmostEqual does not know how to compare np.arrays,
#you have to convert them into lists first

class DAOTests(unittest.TestCase):

    def setUp(self):
        self.dao=DAO.DAO()


    def test_one_data_point(self):
        self.assertEqual([[1,1],[0]],list(self.dao.one_data_point(3)))

    def test_random_data_point(self):
        result=list(self.dao.random_data_point())[1]
        self.assertTrue(result==[0] or result==[1])


    def tearDown(self):
        del self.dao

if __name__=="__main__":
    unittest.main()
