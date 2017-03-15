#!/usr/bin/env python

import pandas as pd
import numpy as np
from dao_interface import DAO

class CurveDAO(DAO):

    def __init__(self, insize=3):
        resolution = .5
        x = np.arange(insize*resolution, 2*np.pi, resolution)
        self._data = pd.Series(np.sin(x), index=x)
        self.size = len(self._data) - insize
        self.input_size = insize
        self.output_size = 1

    @property
    def data(self):
        insize = self.input_size
        series = self._data
        for i in range(insize, self.size-1):
            input_ = series.iloc[i-insize:i].values
            output = series.iloc[i:i+1].values
            yield input_, output
