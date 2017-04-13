#!/usr/bin/env python

import time
import datetime
from datetime import date, timedelta
import numpy as np
from numpy import linalg as la
import os
from xor_dao import XorDAO
from yahoo_dao import SingleClosingSeries as finance
from Net import Net
from os import walk

class UI(object):
    '''Creates a three-layered net based on the data. Trains the network to
    predict stock price change from 20 days prior price information. Predicts
    the price change for tomorrow.'''

    def __init__(self, ticker):
        self.ticker = ticker
        end = date.fromtimestamp(time.time())
        start = end + timedelta(days = -100)
        end = end.strftime('%m/%d/%Y')
        start = start.strftime('%m/%d/%Y')
        self.dao = finance(ticker, start, end, 20)
        self.dao._norm_data()
        form = UI._form(self.dao)
        self.net = Net(form, self.dao)

    @staticmethod
    def _form(dao):
        input_size = dao.input_size
        output_size = dao.output_size
        return [input_size, input_size, output_size]

    @staticmethod
    def find_all_files(ticker):
        walk('../')
        f = []
        for (dirpath, dirnames, filenames) in walk('../'):
            f.extend(filenames)
        return f

    def learn(self, max_cycles=100, learning_rate=0.6, reg=10**-4):
        name = str(self.ticker) + '_net__'+ str(date.fromtimestamp(time.time()))
        self.net.learn_by_back_propagation(max_cycles, learning_rate, reg, learning_rate/4)
        self.net.save_net(name)

    def price_tomorrow(self):
        latest_data = self.dao.latest_data()
        temp = self.net.output(latest_data)
        return self.dao.inv_transform(temp)

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

    ui = UI('CCJ')
    files = UI.find_all_files('CCJ')
    print(len(files))
    #print(files)
    #print(files[50])
    result =[item for item in files if item.startswith('CCJ')]
    name = result[0]
    #with open(name, r) as ccj:
    ui.net.set_coefficients_from_file(name)
    ui.learn(200,0.45)
    print(list(ui.price_tomorrow()))

    '''
    ui.learn(100, 0.6, 10**-4, 0.6/4)
    random_results = ui.dao.random_data_dyads(20)
    for input_, output_ in random_results:
        calculated_output_ = list(ui.dao.inv_transform(ui.net.output(input_)))
        print( list(ui.dao.inv_transform(output_)), calculated_output_)

    print(list(ui.price_tomorrow()))
    '''
