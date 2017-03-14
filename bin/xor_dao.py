#!/usr/bin/env python

"""Does include both DAO and Service class in fact
"""

# import time
import random
import scipy
import numpy as np
from numpy import linalg as la
from dao_interface import DAO

# random.seed(time)

xor_data = [
    ([0.0, 0.0], [-1.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [-1.0]),
    ]
#data=np.array([[[1.0,1.0],[0.0]],[[0.0,1.0],[1.0]]])

class XorDAO(DAO):

    def __init__(self):
        self.data=xor_data
        self.size=len(self.data)
        self.input_size = len(xor_data[0][0])

    def one_data_point(self, i):
        return np.array(self.data[i])

    def data_dyad(self):
        i = random.randint(0, len(self.data)-1)
        return np.asarray(self.data[i])

    def data_monad(self):
        return self.data_dyad()[0]

    def iter(self):
        return self.data