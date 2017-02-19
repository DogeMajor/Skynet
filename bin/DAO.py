#!/usr/bin/env python

import time
import random
import scipy
import numpy as np
from numpy import linalg as la
#Does include both DAO and Service class in fact

random.seed(time)
data=np.array([[[0.0,0.0],[0.0]],[[0.0,1.0],[1.0]],[[1.0,0.0],[1.0]],[[1.0,1.0],[0.0]]])
#data=np.array([[[0.0,0.0],[0.0]],[[0.0,1.0],[1.0]]])
class DAO(object):

    def __init__(self):
        self.data=data
        self.size=len(self.data)

    def one_data_point(self, i):
        return np.array(self.data[i])

    def random_data_point(self):
        number=random.randint(0,len(self.data)-1)
        return np.array(self.data[number])
