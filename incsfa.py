
__author__ = 'Varun Kompella, varun@idsia.ch'

"""
IncSFA Python Implementation

Extract the slowly varying components from the input data incrementally.
More information about Incremental Slow Feature Analysis can be found in
Kompella V.R, Luciw M. and Schmidhuber J., Incremental Slow Feature Analysis:
Adaptive Low-Complexity Slow Feature Updating from High-Dimensional Input Streams, 
Neural Computation, 2012.

"""

from signalstats import signalAvgNode, signalDerivNode, signalVarNode
import numpy as np
from ccipca import CCIPCANode, WhiteningNode
from mca import MCANode

class IncSFANode(object):
    """Extract the slowly varying components from the input data incrementally.
    More information about Incremental Slow Feature Analysis can be found in
    Kompella V.R, Luciw M. and Schmidhuber J., Incremental Slow Feature Analysis:
    Adaptive Low-Complexity Slow Feature Updating from High-Dimensional Input Streams, 
    Neural Computation, 2012.

    **Inputs**
    
      ``input_dim``
          Input Dimension
    
      ``whitening_output_dim``
          dimensionality reduction for the ccipca step 
          
      ``output_dim``
          Output Dimension

      ``eps`` (default: 0.001)
          Learning rate


    **kwargs**

      ``reduce`` (default: False)
          Automatically reduce dimensionality (ccipca step). 
          
      ``var_rel`` (default: 0.001)
          Relative variance threshold to reduce dimensionality (ccipca step)
      
      ``beta`` (default: 1.1)
          Variance ratio threshold to reduce dimensionality (ccipca step)

      ``deMean`` (default: True)
          Subtract signal average.
      
      ``avgMode`` (default: Avg) 
          Type of incremental average updates (Avg, movAvg)

      ``avgN`` (default: 1000 for movAvg) 
          Moving average coefficient (only for movAvg avgMode)

      ``T`` (default: 1)
          Learning rate dynamics term


    **Instance variables of interest**
    
      ``self.v`` (monitor_key: slowFeatures)
         Slow feature vectors

      ``self.wv`` (monitor_key: whiteningVectors)
         Input whitening vectors

      ``self.err`` (monitor_key: error)
         Convergence error
         
      ``self.derr`` (monitor_key: derror)
         Learning progress (derivative of the error)

    """
    
    def __init__(self, input_dim, whitening_output_dim, output_dim, eps=0.001, **kwargs):
        self.input_dim = input_dim
        self.whitening_output_dim = whitening_output_dim
        self.output_dim = output_dim
        self.eps = eps
        self.kwargs = kwargs
        
        self.whiteningnode = WhiteningNode(input_dim, whitening_output_dim, **kwargs)
        self.mcanode = MCANode(whitening_output_dim, output_dim, self.eps, **kwargs)
        self.singlepcanode = CCIPCANode(whitening_output_dim, output_dim=1)
        
        self.deMeanInput = self.kwargs.get('deMean', True)
        self.whiteningnode.deMeanInput = False
        self.mcanode.deMeanInput = False
        self.singlepcanode.deMeanInput = False
        
        self.xavg = signalAvgNode(mode=self.kwargs.get('avgMode', 'Avg'), avgN=self.kwargs.get('avgN',1000))
        self.xderiv = signalDerivNode()
        self.zbvar = signalVarNode()

        self.T = self.kwargs.get('T', 1)
        self._curreps = [self.eps for _ in xrange(output_dim)]           
        self._initExp = True
        
        self.n = 1
        self.v = np.zeros([output_dim, input_dim])
        self.wv = np.zeros([whitening_output_dim, input_dim])
        self.err = 0.0
        self.derr = 0.0
        self._newerr = 0.0
        self._errcnt = 0

        self._validTrainingModes = ['Incremental']

    def _epsUpdate(self, i):
        if i < self.T:
            return 0.00001 + (self.eps - 0.00001) * i * i / (self.T * self.T)
        else:
            return self.eps


    def update(self, x, newEpisode=False, **kwargs):
        assert(x.shape[0] == 1)
        self.n += 1

        if self.deMeanInput:       
            self.xavg.update(x)
            x = self.xavg.deMean(x).copy()

        self.whiteningnode.update(x)
        self.wv = self.whiteningnode.v.copy()

        z = self.whiteningnode.execute(x)

        if newEpisode or self._initExp:
            self.xderiv.xPrev = z.copy()
            self._initExp = False
            return 1

        zdot = self.xderiv.update(z)

        if self.eps is None:        
            self.singlepcanode.update(zdot)
            self.mcanode.gamma = np.asscalar(1.2*self.singlepcanode.d/np.mean(self.zbvar.update(zdot)))         # set gamma > lamda_1
            self._curreps = [0.2/(self.mcanode.gamma[0]*((1/1.2) + eig_id)) for eig_id in xrange(self.output_dim)]
        else:
            self.mcanode.gamma = 1.2*(0.2/self.eps)
            self.mcanode.eps = self._epsUpdate(self.n)
            self._curreps = [self.eps/(1 + eig_id*1.2) for eig_id in xrange(self.output_dim)]

        self.mcanode.update(zdot)

        v_old = self.v.copy()
        self.v = np.dot(self.mcanode.v, self.wv)

        self._newerr += np.linalg.norm(self.v - v_old)
        self._errcnt += 1
        

    def execute(self, x):
        if self.deMeanInput:
            xm = self.xavg.deMean(x)
        else:
            xm = x
        return np.dot(xm, self.v.T)


    def monitorVariables(self, buf=None):
        self._newerr/=float(self._errcnt)
        self.derr = self._newerr - self.err
        self.err = self._newerr
        self._newerr = 0.0
        self._errcnt = 0
        if buf is not None:
            if 'slowFeatures' in buf.keys():
                buf['slowFeatures'].append(self.v.copy())
            if 'whiteningVectors' in buf.keys():
                buf['whiteningVectors'].append(self.wv.copy())
            if 'error' in buf.keys():
                buf['error'].append(self.err)
            if 'derror' in buf.keys():
                buf['derror'].append(self.derr)
        return buf


class IncSFA2Node(IncSFANode):
    """Extract the slowly varying components from quadratically-expanded input data incrementally.
    More information about Incremental Slow Feature Analysis can be found in
    Kompella V.R, Luciw M. and Schmidhuber J., Incremental Slow Feature Analysis:
    Adaptive Low-Complexity Slow Feature Updating from High-Dimensional Input Streams, 
    Neural Computation, 2012.

    **Inputs**
    
      ``input_dim``
          Input Dimension
    
      ``whitening_output_dim``
          dimensionality reduction for the ccipca step 
          
      ``output_dim``
          Output Dimension

      ``eps`` (default: 0.001)
          Learning rate


    **kwargs**

      ``reduce`` (default: False)
          Automatically reduce dimensionality (ccipca step). 
          
      ``var_rel`` (default: 0.001)
          Relative variance threshold to reduce dimensionality (ccipca step)
      
      ``beta`` (default: 1.1)
          Variance ratio threshold to reduce dimensionality (ccipca step)

      ``deMean`` (default: True)
          Subtract signal average.
      
      ``avgMode`` (default: Avg) 
          Type of incremental average updates (Avg, movAvg)

      ``avgN`` (default: 1000 for movAvg) 
          Moving average coefficient (only for movAvg avgMode)

      ``T`` (default: 1)
          Learning rate dynamics term


    **Instance variables of interest**
    
      ``self.v`` (monitor_key: slowFeatures)
         Slow feature vectors

      ``self.wv`` (monitor_key: whiteningVectors)
         Input whitening vectors

      ``self.err`` (monitor_key: error)
         Convergence error
         
      ``self.derr`` (monitor_key: derror)
         Learning progress (derivative of the error)

    """
    def __init__(self, input_dim, whitening_output_dim, output_dim, eps=0.001, **kwargs):
        try: import mdp.nodes; self.expnode = mdp.nodes.QuadraticExpansionNode(input_dim)
        except ImportError: raise Exception('Install MDP toolkit to use IncSFA2Node')
        super(IncSFA2Node, self).__init__(self.expnode.output_dim, whitening_output_dim, output_dim, eps, **kwargs)
        
    def update(self, x, newEpisode=False):
        return super(IncSFA2Node,self).update(self.expnode(np.atleast_2d(x)), newEpisode)

    def execute(self, x):
        return super(IncSFA2Node, self).execute(self.expnode(np.atleast_2d(x)))




