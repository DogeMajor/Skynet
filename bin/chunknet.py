
import numpy as np
from numpy import tanh, exp, empty, zeros, empty_like, dot, asarray, copy, einsum
from numpy.random import rand, seed

seed(123)

K = lambda x: 1/(1 + exp(-x))
dK = lambda x: K(x)*(1 - K(x))

obs = [([0.0, 0.0], [.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [.0]),]
obs = obs*100
N = len(obs)
y = asarray([yo for yi, yo in obs]).reshape((N, 1, 1))

lrate = .4
mom = .5
reg = 1.e-12

class ChunkNet(object):

    def __init__(self, form, dao=None):
        if not tuple(form)==(2, 3, 1):
            raise NotImplementedError()
        if dao is not None:
            raise NotImplementedError()
        self.dao = dao
        self.form = form
        self.x = [empty((N, 2, 1)), empty((N, 3, 1)), empty((N, 1, 1))]
        self.s = [empty_like(xi) for xi in self.x[1:]]
        symmetric_rand = lambda shape: 2*(2*rand(*shape) - 1)
        self.w = [symmetric_rand((N, 3, 2)), symmetric_rand((N, 1, 3))]
        self.x[0] = asarray([yi for yi, yo in obs]).reshape((N, 2, 1))
        
    def forward_pass(self):
        x0, x1, x2 = self.x
        s1, s2 = self.s
        w1, w2 = self.w
        s1[:] = einsum('dij,djk->dik', w1, x0)
        x1[:] = K(s1)
        s2[:] = einsum('dij,djk->dik', w2, x1)
        x2[:] = K(s2)

    def weight_gradient(self):
        w1, w2 = self.w
        x0, x1, x2 = self.x
        s1, s2 = self.s
        error = y - x2
        ds2 = dK(s2)*error
        dw2 = ds2*x1.swapaxes(-1, -2)
        ds1 = ds2*w2.swapaxes(-1, -2)*dK(s1)
        dw1 = ds1*x0.swapaxes(-1, -2)
        dw1 -= w1**3*reg
        dw2 -= w2**3*reg
        return dw1, dw2

class Descender(object):

    def __init__(self, net):
        self.net = net
        self.w1_old, self.w2_old = copy(net.w[0]), copy(net.w[1])

    def descend_batch(self):
        w1, w2 = self.net.w
        _, _, x2 = self.net.x

        self.net.forward_pass()
        dw1, dw2 = self.net.weight_gradient()

        w1[:] += lrate*dw1*(1 - mom) + (w1 - self.w1_old)*mom
        w2[:] += lrate*dw2*(1 - mom) + (w2 - self.w2_old)*mom
        self.w1_old[:] = w1
        self.w2_old[:] = w2
        sqe = (y - x2)[0]**2
        return sqe

    def descend(self, N):
        for i in range(N):
            chunk_errors = [self.descend_batch() for _ in range(100)]
            sqe = np.mean(chunk_errors)
            yield i, sqe

if __name__=='__main__':

    net = ChunkNet((2, 3, 1))
    desc = Descender(net)
    idx, sqes = zip(*desc.descend(100))

    from pylab import *
    ion()
    plot(idx, sqes)