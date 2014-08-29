
__author__ = 'Varun Kompella, varun@idsia.ch'

"""Extract the minor components from the input data incrementally.
More information about Minor Component Analysis can be found in 
Peng, D. and Yi, Z, A new algorithm for sequential minor component analysis, 
International Journal of Computational Intelligence Research,
2(2):207--215, 2006.
"""

from signalstats import signalAvgNode
import numpy as np


class MCANode(object):
    """Extract the minor components from the input data incrementally.
    More information about Minor Component Analysis can be found in 
    Peng, D. and Yi, Z, A new algorithm for sequential minor component analysis, 
    International Journal of Computational Intelligence Research,
    2(2):207--215, 2006.

    **Inputs**
    
      ``input_dim``
          Input Dimension
    
      ``output_dim``
          Output Dimension

      ``eps`` (default: 0.001)
          Learning rate

      ``gamma`` (default: 1.0)
          Sequential addition coefficient
          

    **kwargs**

      ``deMean`` (default: True)
          Subtract signal average.
      
      ``avgMode`` (default: Avg) 
          Type of incremental average updates (Avg, movAvg)

      ``avgN`` (default: 1000 for movAvg) 
          Moving average coefficient (only for movAvg avgMode)

      ``normalize`` (default: True)
          If True, eigenvectors are normalized after every update.
          Useful for non-stationary input data.
    
    **Instance variables of interest**
    
      ``self.v`` (monitor_key : eigenVectors)
         Normalized eigen vectors
    
      ``self.d`` (monitor_key : eigenVals)
         Corresponding eigen values
 
    """
    
    def __init__(self, input_dim, output_dim, eps=0.001, gamma=1.0, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = eps
        self.gamma = gamma
        self.kwargs = kwargs
        
        self.deMeanInput = self.kwargs.get('deMean', True)
        self.xavg = signalAvgNode(mode=self.kwargs.get('avgMode', 'Avg'), avgN=self.kwargs.get('avgN',1000))
        self.normalize = self.kwargs.get('normalize', True)
        
        self.n = 1                                                          # n value for the mca
        self._v = 0.1*np.random.randn(self.output_dim, self.input_dim)      # Internal Eigen Vector (unNormalized)
        _d = np.sum(np.absolute(self._v)**2,axis=-1)**(1./2)                # Internal Eigen Values     
        self._v = self._v/_d.reshape(self._v.shape[0],1)               # Internal Eigen Vector (Normalized)
        self.v = self._v.copy()                                             # Eigen Vector (Normalized) (reduced if reduce is True)
        self.d = _d.copy()                                                  # Eigen Value (reduced if reduce is True)

        self._validTrainingModes = ['Incremental']
        

    def update(self, x, **kwargs):
        assert(x.shape[0] == 1)
        self.n += 1

        if self.deMeanInput:       
            self.xavg.update(x)
            x = self.xavg.deMean(x).copy()
        
        C = np.dot(x.T, x)
        for j in xrange(self.output_dim):
            if self.eps is None:
                n = 0.2/(self.gamma*((1/1.2) + j))
            else:
                n = self.eps/(1 + j*1.2)
            v = self._v[j:j+1,:x.shape[1]].T
            a = np.dot(C, v)
            if self.normalize:
                v = (1.5 - n) * v - n * a 
            else:
                v = (1.5 - n*(self.d[j]**2)) * v - n * a 
            l = np.dot(v.T, v)
            C = C + self.gamma * np.dot(v, v.T) / l 
            v=v.T
            self.d[j] = np.sqrt(l)
            if self.normalize:
                self._v[j,:x.shape[1]] = v / self.d[j]
            else:
                self._v[j,:x.shape[1]] = v.copy()

        self.v = self._v[:,:x.shape[1]].copy()

    def execute(self, x):
        if self.deMeanInput:
            xm = self.xavg.deMean(x)
        else:
            xm = x
        return np.dot(xm, self.v.T)

    def monitorVariables(self, buf):
        if 'eigenVectors' in buf.keys():
            buf['eigenVectors'].append(self.v.copy())
        if 'eigenValues' in buf.keys():
            buf['eigenValues'].append(self.d.copy())
        return buf


