
__author__ = 'Varun Kompella, varun@idsia.ch'

"""Extract the principal components from the input data incrementally.
More information about Candid-Covariance free Incremental Principal Component
Analysis can be found in Weng J., Zhang Y. and Hwang W., Candid covariance-free 
incremental principal component analysis, IEEE Trans. Pattern Analysis and Machine Intelligence,
vol. 25, 1034--1040, 2003.

"""

from signalstats import signalAvgNode
import numpy as np


class CCIPCANode(object):
    """Extract the principal components from the input data incrementally.
    More information about Candid-Covariance free Incremental Principal Component
    Analysis can be found in Weng J., Zhang Y. and Hwang W., Candid covariance-free 
    incremental principal component analysis, IEEE Trans. Pattern Analysis and Machine Intelligence,
    vol. 25, 1034--1040, 2003.

    **Inputs**
    
      ``input_dim``
          Input Dimension
    
      ``output_dim``
          Output Dimension
          

    **kwargs**

      ``reduce`` (default: False)
          Automatically reduce dimensionality. 
          
      ``var_rel`` (default: 0.001)
          Relative variance threshold to reduce dimensionality
      
      ``beta`` (default: 1.1)
          Variance ratio threshold to reduce dimensionality

      ``deMean`` (default: True)
          Subtract signal average.
      
      ``avgMode`` (default: Avg) 
          Type of incremental average updates (Avg, movAvg)

      ``avgN`` (default: 1000 for movAvg) 
          Moving average coefficient (only for movAvg avgMode)
    
    **Instance variables of interest**
    
      ``self.v`` (monitor_key : eigenVectors)
         Normalized eigen vectors
    
      ``self.d`` (monitor_key : eigenVals)
         Corresponding eigen values

    
    """
    
    def __init__(self, input_dim, output_dim, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kwargs = kwargs

        self.var_rel = self.kwargs.get('var_rel',0.001)
        self.beta = self.kwargs.get('beta',1.1)
        self.xavg = signalAvgNode(mode=self.kwargs.get('avgMode', 'Avg'), avgN=self.kwargs.get('avgN',1000))

        self.deMeanInput = self.kwargs.get('deMean', True)
        self.reduce = self.kwargs.get('reduce', False)
        
        self.n = 1                                                          # n value for the ccipca
        self._v = 0.1*np.random.randn(self.output_dim, self.input_dim)      # Internal Eigen Vector (unNormalized)
        self._d = np.sum(np.absolute(self._v)**2,axis=-1)**(1./2)           # Internal Eigen Values     
        self._vn = self._v/self._d.reshape(self._v.shape[0],1)              # Internal Eigen Vector (Normalized)
        self.explained_var_tot = self._d.sum()                              # Total Explained Variance
        self.v = self._vn.copy()                                            # Eigen Vector (Normalized) (reduced if reduce is True)
        self.d = self._d.copy()                                             # Eigen Value (reduced if reduce is True)
        
        self.reducedDim = self.output_dim

        self._validTrainingModes = ['Incremental']


    def _amnesic(self, _i):
        [n1, n2, m, C] = [20.,200.,2000.,3.]
        if _i < n1:
            l = 0
        elif (_i >= n1) and (_i < n2):
            l = C * (_i - n1) / (n2 - n1)
        else:
            l = C + (_i - n2) / m
        n = _i
        _wold = float(n - 1 - l) / n
        _wnew = float(1 + l) / n
        return [_wold, _wnew]


    def update(self, x, **kwargs):
        assert(x.shape[0] == 1)
        self.n += 1

        if self.deMeanInput:       
            self.xavg.update(x)
            x = self.xavg.deMean(x).copy()
        
        [w1, w2] = self._amnesic(self.n)
        red_j = self.output_dim
        red_j_Flag = False
        explained_var = 0.0
        
        r = x.copy()
        for j in xrange(self.output_dim):
            v = self._v[j:j+1].copy()
            v = w1 * v + w2 * np.dot(r, v.T) / self._d[j] * r
            self._d[j] = np.linalg.norm(v) 
            vn = v / self._d[j]
            r = r - np.dot(r, vn.T) * vn
            explained_var += self._d[j]
            if (self.reduce  is True) and (red_j_Flag is False):
                ratio1 = self._d[j] / self._d[0]
                ratio2 = explained_var / self.explained_var_tot
                #print j, " :  ", ratio1, " :  ", ratio2, " :  ",self._d[j]
                if (ratio1 < self.var_rel or ratio2 > self.beta):
                    red_j = j
                    red_j_Flag = True
                    #print j,  " :  ", ratio1, " :  ", ratio2, " :  ", self._d[j]
            self._v[j] = v.copy()
            self._vn[j] = vn.copy()

        if explained_var > 0.0001:
            self.explained_var_tot = explained_var
        self.v = self._vn[:red_j].copy()
        self.d = self._d[:red_j].copy()
        self.reducedDim = red_j
        

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


class WhiteningNode(CCIPCANode):
    """Incremental Whitening of the input data using CCIPCA.
    
    **Inputs**
    
      ``input_dim``
          Input Dimension
    
      ``output_dim``
          Output Dimension
          

    **kwargs**

      ``reduce`` (default: False)
          Automatically reduce dimensionality. 
          
      ``var_rel`` (default: 0.001)
          Relative variance threshold to reduce dimensionality
      
      ``beta`` (default: 1.1)
          Variance ratio threshold to reduce dimensionality

      ``deMean`` (default: True)
          Subtract signal average.
      
      ``avgMode`` (default: Avg) 
          Type of incremental average updates (Avg, movAvg)
    
    **Instance variables of interest**
    
      ``self.v`` (monitor_key : eigenVectors)
         Normalized eigen vectors
    
      ``self.d`` (monitor_key : eigenVals)
         Corresponding eigen values


    """
    
    def __init__(self, input_dim, output_dim=None, **kwargs):
        super(WhiteningNode,self).__init__(input_dim, output_dim, **kwargs)

    def update(self, x):
        super(WhiteningNode,self).update(x=x)
        if self.v is not None:
            wv = np.zeros(self.v.shape)
            for i in xrange(wv.shape[0]):
                wv[i] = self.v[i] / np.sqrt(self.d[i])
            self.v = wv


