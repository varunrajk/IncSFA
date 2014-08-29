
__author__ = 'Varun Kompella, varun@idsia.ch'

'''
Signal stats node

'''

import numpy as np

class signalAvgNode(object):
    def __init__(self, mode='Avg', avgN=1000.):
        self.avg = 0
        self.n = 0
        self.mode = mode
        self.N = avgN

    def update(self, x):
        input_data = np.atleast_2d(x).copy()
        if self.mode == 'Avg':
            self.avg = (self.n * self.avg + input_data.sum(axis=0)) / (self.n + input_data.shape[0])
        elif self.mode == 'movAvg':
            alpha = 2.0/(self.N+1)
            if self.n == 0:
                self.avg = (self.n * self.avg + input_data.sum(axis=0)) / (self.n + input_data.shape[0])
            else:
                for i in xrange(input_data.shape[0]):
                    self.avg = (1-alpha) * self.avg + alpha*input_data[i,:]
        self.n += input_data.shape[0]
        return self.avg

    def deMean(self, x):
        if self.n == 1:
            return x
        else:
            return (x - self.avg)

    def reset(self):
        self.avg = 0
        self.n = 0




class signalVarNode(object):
    def __init__(self):
        self.avg = 0.
        self.var = 0.
        self.n = 0
    
    def update(self, x):
        input_data = np.atleast_2d(x).copy()
        delta = input_data - self.avg
        self.avg = (self.n * self.avg + input_data.sum(axis=0)) / (self.n + input_data.shape[0])
        if (self.n + input_data.shape[0]) > 1:
            x_mean = input_data - self.avg
            x_var = delta * x_mean
            self.var = ((self.n - 1) * self.var + x_var.sum(axis=0)) / (self.n - 1 + input_data.shape[0])
        self.n += input_data.shape[0]
        return self.var
    
    def getStd(self):
        return np.sqrt(self.var)


class signalDerivNode(object):
    def __init__(self):
        self.xPrev = None
    
    def update(self, x):
        x = np.atleast_2d(x).copy()
        if x.shape[0] == 1:
            if self.xPrev is None:
                self.xPrev = x.copy()
            diff = x - self.xPrev[0, :x.shape[1]]
            self.xPrev[0, :x.shape[1]] = x[0, :].copy()
            return diff
        else:
            xdot = x[1:,:] - x[:-1, :]
            self.xPrev = x[x.shape[0] - 1:, :].copy()
            return xdot

    @staticmethod
    def deriv(x):
        x = np.atleast_2d(x).copy()
        assert(x.shape[0] > 1)
        return x[1:,:] - x[:-1, :]


