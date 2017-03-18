
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
reg = 1.e-4
# reg = 0.

class ChunkNet(object):

    """
    Data attributes:
    W       list of weight matrices
    x       list of unactivated data vectors (vertical)
    ax      list of activated data vectors (vertical)
    """

    def __init__(self, hidden_sizes, dao=None):
        # Shape of net
        assert(len(hidden_sizes) >= 1)
        self.form = (INSIZE,) + tuple(hidden_sizes) + (OUTSIZE,)
        self.depth = len(self.form)
        # Data access
        if dao is not None:
            raise NotImplementedError()
        # Data arrays
        self.ax = [empty((BATCHSIZE, size, 1)) for size in self.form]
        # Data arrays but unactivated
        self.x = [None] + [empty_like(xi) for xi in self.ax[1:]]
        # Gradients of energy wrt. unactivated data at each layer
        self.dEdx = [None] + [empty_like(xi) for xi in self.ax[1:]]
        # Uniform random for weight initialization
        unif_rand = lambda shape: 2*(2*rand(*shape) - 1)
        # Weights
        self.W = [unif_rand((BATCHSIZE, self.form[i+1], self.form[i])) for i in range(self.depth-1)]
        # Weight gradients
        self.dW = [zeros_like(Wi) for Wi in self.W]
        # First data array equals input observations
        self.ax[0] = asarray([yi for yi, yo in obs]).reshape((BATCHSIZE, 2, 1))

    def activate(self):
        x, ax, W = self.x, self.ax, self.W
        for i in range(1, self.depth):
            x[i][:] = einsum('dij,djk->dik', W[i-1], ax[i-1])
            ax[i][:] = K(x[i])

    def weight_gradient(self):

        x, ax, W, dEdx, dEdW = self.x, self.ax, self.W, self.dEdx, self.dW

        # Output layer first.
        # Observation error
        error = ax[-1] - y
        # E for energy = cost function, d for derivative
        dEdx[-1] = error*dK(x[-1])
        dEdW[-1] = dEdx[-1]*ax[-2].swapaxes(-1, -2)

        # The rest of the weight layers.
        # For 1 hidden layer, this loop is just i=0
        for i in range(self.depth-3, -1, -1):
            # Gradient wrt. unactivated data.
            # The sum corresponds to contributions by (unactivated) i+2 data layer.
            # To match dimensions, shape of the sum expression is spread.
            # Each activation gradient is multiplied elementwise.
            dEdx[i+1] = einsum('dnm,dni->di', dEdx[i+2], W[i+1])[..., None]*dK(x[i+1])
            # Gradient wrt. weight matrix
            dEdW[i] = dEdx[i+1]*ax[i].swapaxes(-1, -2)
            # Regularization term gradient
            dEdW[i] += W[i]**1*reg

        return dEdW

class Descender(object):

    def __init__(self, net):
        self.net = net
        self.delta_prev = [copy(dWi) for dWi in net.dW]

    def descend_batch(self):
        W = self.net.W
        ax = self.net.ax
        # Activate net
        self.net.activate()
        # Energy gradient matrices
        dEdW = self.net.weight_gradient()
        # Update weights
        for wi in range(net.depth-1):
            # Weight increment. Doesn't include momentum
            delta_wi = -lrate*dEdW[wi]
            # Descend step
            W[wi][:] += delta_wi + mom*self.delta_prev[wi]
            # Save weight increment for the next step
            self.delta_prev[wi][:] = delta_wi
        # Squared errors
        sqerrors = (y - ax[-1])**2
        return sqerrors

    def descend(self, N):
        for i in range(N):
            chunk_errors = [self.descend_batch() for _ in range(1)]
            sqerror = np.mean(chunk_errors)
            print(i, sqerror)
            yield i, sqerror

if __name__=='__main__':

    net = ChunkNet([3, 50, 3])
    desc = Descender(net)
    idx, sqes = zip(*desc.descend(100))

    from pylab import *
    ion()
    plot(idx, sqes)
    ylim([0, sqes[0]*1.05])
