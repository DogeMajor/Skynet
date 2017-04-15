#!/usr/bin/env python

import time
import datetime
from datetime import date, timedelta
import numpy as np
from numpy import linalg as la
import os
from xor_dao import XorDAO
from yahoo_dao import SingleClosingSeries as finance
from net import Net
from os import walk

class Stock(object):
    '''Creates a three-layered net based on the data. Trains the network to
    predict stock price change from 20 days prior price information. Predicts
    the price change for tomorrow. This program is meant for entertainment
    and education purposes only and should not be considered as advice to make investments.
    Before making any financial decisions, always consult your local Dogecoin
    broker first.'''

    def __init__(self, ticker):
        self.ticker = ticker
        end = date.fromtimestamp(time.time())
        start = end + timedelta(days = -100)
        end = end.strftime('%m/%d/%Y')
        start = start.strftime('%m/%d/%Y')
        self.dao = finance(ticker, start, end, 20)
        self.dao._norm_data()
        form = Stock._form(self.dao)
        self.net = Net(form, self.dao)
        self._load_previous_net()

    @staticmethod
    def _form(dao):
        input_size = dao.input_size
        output_size = dao.output_size
        return [input_size, input_size, output_size]

    @staticmethod
    def find_all_files(ticker):
        files = []
        for (dirpath, dirnames, filenames) in walk('data/'):
            files.extend(filenames)
        result =[item for item in files if item.startswith(ticker)]
        return result

    @staticmethod
    def pick_latest_file(files, today):
        for delta_day in range(100):#Since we use 100 days of price data!
            today = today + timedelta(delta_day)
            results = [item for item in files if item.endswith(str(today))]
            if results != []:
                return results[0]

    def _load_previous_net(self):
        files = Stock.find_all_files(self.ticker)
        today = date.fromtimestamp(time.time())
        coeff_file = Stock.pick_latest_file(files, today)
        if coeff_file != None:
            self.net.set_coefficients_from_file('data/'+coeff_file)

    def learn(self, max_error=0.05,max_cycles=100, learning_rate=0.6, reg=10**-5):
        self._load_previous_net()
        today = date.fromtimestamp(time.time())
        name = str(self.ticker) + '_net__'+ str(today)
        param =(max_error,max_cycles, learning_rate, reg, learning_rate/4)
        self.net.learn_by_back_propagation(max_error,max_cycles, learning_rate, reg, learning_rate/4)
        self.net.save_net('data/'+name)

    def price_tomorrow(self):
        latest_data = self.dao.latest_data()
        temp = self.net.output(latest_data)
        return self.dao.inv_transform(temp)


class Portfolio(object):

    def __init__(self, tickers):
        self.tickers = tickers
        self.stocks = [Stock(ticker) for ticker in tickers]

    def learn(self,max_error=0.05, max_cycles=200, learning_rate=0.6, reg=10**-4):
        for stock in self.stocks:
            stock.learn(max_error, max_cycles, learning_rate, reg)

    def prices_tomorrow(self):
        prices = [list(stock.price_tomorrow()) for stock in self.stocks]
        tickers = self.tickers
        return dict(zip(tickers, prices))


if __name__=='__main__':

    dogefund = Portfolio(['TSLA','UTX'])
    dogefund.learn(0.05, 200, 0.5)
    prices = dogefund.prices_tomorrow()
    print(prices)
