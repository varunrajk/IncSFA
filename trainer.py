
__author__ = 'Varun Kompella, varun@idsia.ch'

"""
Trainer node for UL algorithms

"""

import numpy as np
import subprocess
import sys

def updateProgressBar(val, name, barLength):
    status = ""
    if val >= 1:
        val = 1
        status = "\n%s Done...\r\n"%(name)
    block = int(round(barLength*val))
    text = "\r[{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), int(val*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()

class TrainerNode(object):
    """Trainer Node for UL algorithms
    
    **Inputs**
    
      ``ulnode``
          ulnode class instance
    
      ``mode`` (default: Incremental)
          training modes: Incremental, BlockIncremental, Batch 

      ``ticker`` (default: inf)
          Ticker frequency value to monitor variables 
          
    """
    
    def __init__(self, ulnode, mode='Incremental', **kwargs):
        self.ulnode = ulnode
        self.mode = mode
        self.kwargs = kwargs

        self.ticker = self.kwargs.get('ticker',1000)

        if not(self.mode in self.ulnode._validTrainingModes):
            raise Exception('Invalid training mode! Valid training modes: %s'%(str(self.ulnode._validTrainingModes)))

        if self.mode is 'Incremental':
            self._dataGen = self._dataIncGen
        elif self.mode is 'BlockIncremental':
            self._dataGen = self._dataBlkIncGen
            self.blkSize = int(kwargs.get('blockSize',1))
            self.ticker = self.blkSize
        elif self.mode is 'Batch':
            self._dataGen = self._dataBatGen
        else:
            raise Exception('Unrecognized mode! Training modes: Incremental, BlockIncremental, Batch ')

        tput = subprocess.Popen(['tput', 'cols'], stdout=subprocess.PIPE)
        self._barLength = int(tput.communicate()[0].strip())-10


    def _dataIncGen(self, x, iterval):
        totSamples = x.shape[0]*iterval
        for n in xrange(totSamples):
            updateProgressBar((n+1)/float(totSamples), 'Training %s'%(type(self.ulnode).__name__), self._barLength)
            i = n%x.shape[0]
            yield x[i:i+1]

    def _dataBlkIncGen(self, x, iterval):
        totBlks = x.shape[0]/int(self.blkSize) + (x.shape[0]%int(self.blkSize) > 0)
        totBlks *= iterval
        for blk in xrange(totBlks):
            updateProgressBar((blk+1)/float(totBlks), 'Training %s'%(type(self.ulnode).__name__), self._barLength)
            i = (blk*self.blkSize)%x.shape[0]
            yield x[i:i+self.blkSize]

    def _dataBatGen(self, x, iterval):
        self.ticker = x.shape[0]
        for n in xrange(iterval):
            updateProgressBar((n+1)/float(iterval), 'Training %s'%(type(self.ulnode).__name__), self._barLength)
            yield x
            

    def train(self, x, iterval=1, **kwargs):
        assert(x.ndim >= 2)
        print '\nTraining %s....'%(type(self.ulnode).__name__)
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        if x.shape[0] is 1: iterval=1          #overide iterval value when input is just a single sample

        if kwargs.has_key('monitor_keys'): 
            self.monitorVar = {key:[] for key in kwargs['monitor_keys']}
        else: 
            self.monitorVar = {}
        
        for _x in self._dataGen(x,iterval):
            self.ulnode.update(_x, **kwargs)
            if (self.ulnode.n % self.ticker == 0):
                self.monitorVar = self.ulnode.monitorVariables(self.monitorVar)
        return 1
        

    def itrain(self, x, iterval=1, **kwargs):
        '''Non-blocking interactive train method. Useful for online plotting (animations).
           yields 1 - after ticker time steps
           yields 0 - when done 
        '''
        assert(x.ndim >= 2)
        print '\nTraining %s....'%(type(self.ulnode).__name__)
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
        if x.shape[0] is 1: iterval=1          #overide iterval value when input is just a single sample
        if kwargs.has_key('monitor_keys'): 
            MONITOR = True
            self.monitorVar = {key:[] for key in kwargs['monitor_keys']}
        else: MONITOR = False
        for _x in self._dataGen(x,iterval):
            self.ulnode.update(_x, **kwargs)
            if (self.ulnode.n % self.ticker == 0):
                if MONITOR:
                    monitorVar = {key:[] for key in kwargs['monitor_keys']}
                    self.monitorVar = self.ulnode.monitorVariables(monitorVar)
                yield 1
        yield 0


