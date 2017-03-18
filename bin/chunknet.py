
import numpy as np
from numpy import tanh, exp, empty, zeros, zeros_like, empty_like, dot, asarray, copy, einsum
from numpy.random import rand, seed

seed(123)

# K, dK = lambda x: 1/(1 + exp(-x)), lambda x: K(x)*(1 - K(x))
K, dK = lambda x: 1.1427894*(1-(tanh(0.666*x))**2), lambda x: 1.1427894*(1-(tanh(0.666*x))**2)

obs = [([0.0, 0.0], [.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [.0]),]
# obs = obs*100
BATCHSIZE = len(obs)
INSIZE = 2
OUTSIZE = 1
y = asarray([yo for yi, yo in obs]).reshape((BATCHSIZE, 1, 1))

lrate = .004
mom = .0
reg = 1.e-12
# reg = 0.

class ChunkNet(object):

    def __init__(self, hidden_sizes, dao=None):
        # Shape of net
        assert(len(hidden_sizes) >= 1)
        self.form = (INSIZE,) + tuple(hidden_sizes) + (OUTSIZE,)
        self.depth = len(self.form)
        # Data access
        if dao is not None:
            raise NotImplementedError()
        # Data arrays
        self.x = [empty((BATCHSIZE, size, 1)) for size in self.form]
        # Data arrays but unactivated
        self.s = [empty_like(xi) for xi in self.x[1:]]
        # Change of data arrays
        self.ds = [empty_like(si) for si in self.s]
        # Uniform random for weight initialization
        unif_rand = lambda shape: 2*(2*rand(*shape) - 1)
        # Weights
        self.w = [unif_rand((BATCHSIZE, self.form[i+1], self.form[i])) for i in range(self.depth-1)]
        # Weight gradients
        self.dw = [zeros_like(wi) for wi in self.w]
        # First data array equals input observations
        self.x[0] = asarray([yi for yi, yo in obs]).reshape((BATCHSIZE, 2, 1))
        
    def activate(self):
        s, x, w = self.s, self.x, self.w
        for i in range(1, self.depth):
            s[i-1][:] = einsum('dij,djk->dik', w[i-1], x[i-1])
            x[i][:] = K(s[i-1])

    def weight_gradient(self):
        s, x, w, ds, dw = self.s, self.x, self.w, self.ds, self.dw
        # Last layer is slightly special
        error = y - x[-1]
        ds[-1] = dK(s[-1])*error
        dw[-1][:] = ds[-1]*x[-2].swapaxes(-1, -2)
        # Backpropagate gradients
        for i in range(self.depth-2, 0, -1):
            wi = i - 1 # weight index
            ds[wi] = ds[wi+1]*w[wi+1].swapaxes(-1, -2)*dK(s[wi])
            dw[wi] = ds[wi]*x[i-1].swapaxes(-1, -2)
        return dw

class Descender(object):

    def __init__(self, net):
        self.net = net
        self.delta_prev = [copy(dwi) for dwi in net.dw]

    def descend_batch(self):
        w = self.net.w
        x = self.net.x
        # Activate net
        self.net.activate()
        # Backpropagate gradients
        dw = self.net.weight_gradient()
        # Update weights
        for wi in range(net.depth-1):
            # Weight change
            delta_wi = lrate*dw[wi]
            # Save weight change for next iteration momentum
            self.delta_prev[wi][:] = w[wi]
            # Descend step
            w[wi][:] += delta_wi + mom*self.delta_prev[wi]
        # Squared errors
        sqe = (y - x[-1])**2
        return sqe

    def descend(self, N):
        for i in range(N):
            chunk_errors = [self.descend_batch() for _ in range(30)]
            sqe = np.mean(chunk_errors)
            yield i, sqe

if __name__=='__main__':

    net = ChunkNet([3])
    desc = Descender(net)
    idx, sqes = zip(*desc.descend(300))

    from pylab import *
    ion()
    plot(idx, sqes)
    ylim([0, sqes[0]*1.05])
