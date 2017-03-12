#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import Layer
from DAO import DAO
from FinanceDAO import FinanceDAO
from FinanceService import FinanceService
from Net import Net
#This can be expanded to include a month of test data etc. 
class UI(object):

    def __init__(self, net, service):
        self.net = net
        self.service = service

    def learn(self, max_cycles, learning_rate):
        self.net.learn_by_back_propagation(max_cycles, learning_rate)

    def stats(self, input_):
        mean = np.mean(input_)
        max_ = np.max(input_)
        min_ = np.min(input_)
        return [mean, max_-min_]

    def price(self, input_):
        stats = self.stats(input_)
        temp = self.net.output(input_)
        return stats[1]*temp+stats[0]


if __name__=='__main__':
    ccj = FinanceService('CCJ',4)
    #ccj._start_date =
    input_ = ccj._closing_price('2017-03-01', '2017-03-11')
    net = Net([4,4,1],ccj)
    ui = UI(net,ccj)
    print(input_)
    print(input_[0:4])
    ui.net.learn_by_back_propagation(200,0.4)
    ui.learn(40, 0.4)
    print(input_)
    out = ui.price(np.array(input_[0:4]))
    print(out)
