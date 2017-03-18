
import numpy as np
from numpy import tanh, exp, empty, zeros, zeros_like, empty_like, dot, asarray, copy, einsum
from numpy.random import rand, seed

# seed(123)

# K, dK = lambda x: 1/(1 + exp(-x)), lambda x: exp(-x)/(1 + exp(-x))**2
K, dK = lambda x: 1.7159*np.tanh(0.666*x), lambda x: 1.1427894*(1-(tanh(0.666*x))**2)

obs = [([0.0, 0.0], [.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [.0]),]
obs = obs*1000
BATCHSIZE = len(obs)
INSIZE = 2
OUTSIZE = 1
y = asarray([yo for yi, yo in obs]).reshape((BATCHSIZE, 1, 1))

lrate = .2
mom = .4
reg = 1.e-10
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
        self.s = [None] + [empty_like(xi) for xi in self.x[1:]]
        # Change of data arrays
        self.ds = [None] + [empty_like(xi) for xi in self.x[1:]]
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
            s[i][:] = einsum('dij,djk->dik', w[i-1], x[i-1])
            x[i][:] = K(s[i])

    def weight_gradient(self):
        s, x, W, dEds, dEdW = self.s, self.x, self.w, self.ds, self.dw

        # error = y - x[-1]
        # ds[-1] = dK(s[-1])*error
        # dEdW[-1][:] = ds[-1]*x[-2].swapaxes(-1, -2)

        # Output layer is special
        error = x[-1] - y
        dEds[-1] = error*dK(s[-1])
        dEdW[-1] = dEds[-1]*x[-2].swapaxes(-1, -2)
        # Backpropagate gradients over the rest of the weight layers.
        # For 1 hidden layer, this loop is just i=0
        for i in range(self.depth-3, -1, -1):

            # ds[i+1] = ds[i+2]*W[i+1].swapaxes(-1, -2)*dK(s[i+1])
            # dEdW[i] = ds[i+1]*x[i].swapaxes(-1, -2)

            dEds[i+1] = einsum('dnm,dni->di', dEds[i+2], W[i+1])[..., None]*dK(s[i+1])
            dEdW[i] = dEds[i+1]*x[i].swapaxes(-1, -2)
            # Regularization term gradient
            dEdW[i] += W[i]**3*reg
        return dEdW

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
        dEdW = self.net.weight_gradient()
        # Update weights
        for wi in range(net.depth-1):
            # Weight change. Doesn't include momentum
            delta_wi = -lrate*dEdW[wi]
            # Descend step
            w[wi][:] += delta_wi + mom*self.delta_prev[wi]
            # Save weight change for next iteration momentum
            self.delta_prev[wi][:] = delta_wi
        # Squared errors
        sqe = (y - x[-1])**2
        return sqe

    def descend(self, N):
        for i in range(N):
            chunk_errors = [self.descend_batch() for _ in range(1)]
            sqe = np.mean(chunk_errors)
            print(i, sqe)
            yield i, sqe

if __name__=='__main__':

    net = ChunkNet([3, 50, 3])
    desc = Descender(net)
    idx, sqes = zip(*desc.descend(100))

    from pylab import *
    ion()
    plot(idx, sqes)
    ylim([0, sqes[0]*1.05])
