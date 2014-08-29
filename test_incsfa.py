
import numpy as np
from incsfa import IncSFANode
from trainer import TrainerNode
import time
try:
    from pyqtgraph.Qt import QtCore
    import pyqtgraph as pg
    PYQT = True
except ImportError:
    PYQT = False
    from matplotlib import pyplot as plt

#######################################################################################
#######################################################################################


def testincsfa(animate=False):
    try:        
        import mdp.nodes
    except ImportError: raise('Intall MDP toolkit for check/test')
    
    iterval = 20
    t = np.linspace(0, 4 * np.pi, 500)
    x1 = np.sin(t) + np.power(np.cos(11 * t), 2)
    x2 = np.cos(11 * t)
    x = np.zeros([t.shape[0], 2])
    x[:, 0] = x1
    x[:, 1] = x2
    expnode = mdp.nodes.PolynomialExpansionNode(2)
    input_data = expnode(x)
    
    ##Setup node/trainer    
    input_dim = input_data.shape[1]; whitening_output_dim=5; output_dim=5
    node = IncSFANode(input_dim, whitening_output_dim, output_dim, eps=0.05)
    trainer = TrainerNode(node, mode='Incremental', ticker=100)

    #For check
    bsfanode = mdp.nodes.SFANode(output_dim=output_dim)
    bsfanode(input_data)
    bv = bsfanode.sf.T

    if PYQT:
        global app
        app = pg.QtGui.QApplication([])
        
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')
        pg.setConfigOptions(antialias=True)

        ##Setup Plot

        view = pg.GraphicsView()
        l = pg.GraphicsLayout(border=(100,100,100))
        view.setCentralItem(l)
        view.setWindowTitle('IncSFA Test')
        view.resize(1200,500)

        ## Title at top
        
        text = """
        This example demonstrates the working of Incremental Slow Feature Analysis (IncSFA).
        """
        l.addLabel(text, col=0, colspan=5, size='20px')
        l.nextRow()
 
        pen = pg.mkPen(color=(255,255,200), width=1)
        labelStyle = {'color': '#FFF', 'font-size': '16px'}
       

        p1 = l.addPlot(rowspan=2, colspan=2)
 
        #Axis Settings
        p1.getAxis('left').setPen(pen)
        p1.getAxis('bottom').setPen(pen)
        p1.showGrid(1,1,.2)
        p1.setLabel('bottom', 'Ticker', units='t', **labelStyle)
        p1.setLabel('left', 'DC', **labelStyle)
        p1.setTitle('Direction Cosines', size='16px')
        p1.setXRange(0,iterval*t.shape[0]/trainer.ticker)
        p1.setYRange(0,1.1)
 
        #Legend        
        leg = pg.LegendItem(size=(100,60), offset=(-20,100))  # args are (size, offset)
        leg.setParentItem(p1.graphicsItem())   # Note we do NOT call plt.addItem in this case
          
        #init curves
        curves1 = [pg.PlotCurveItem(pen=(i,output_dim*1.3)) for i in range(output_dim)]
        for i,c in enumerate(curves1):
            p1.addItem(c)
            leg.addItem(c, 'V-%d'%i)

        l2 = l.addLayout(col=2, colspan=2, border=(50,0,0))

        curves2 = [pg.PlotCurveItem(pen=(i,output_dim*1.3)) for i in range(4)]
        
        l2.setContentsMargins(10, 10, 10, 10)
        l2.addLabel("IncSFA Feature Outputs", colspan=3, size='16px')
        l2.nextRow()
        l2.addLabel('Output Value', angle=-90, rowspan=2 , size='16px')
        p21 = l2.addPlot(); p21.addItem(curves2[0]); p21.setXRange(0,t.shape[0]); p21.setYRange(-2,2)
        p22 = l2.addPlot(); p22.addItem(curves2[1]); p22.setXRange(0,t.shape[0]); p22.setYRange(-2,2)
        l2.nextRow()
        p23 = l2.addPlot(); p23.addItem(curves2[2]); p23.setXRange(0,t.shape[0]); p23.setYRange(-2,2)
        p24 = l2.addPlot(); p24.addItem(curves2[3]); p24.setXRange(0,t.shape[0]); p24.setYRange(-2,2)
        l2.nextRow()
        l2.addLabel("Time(t)", col=1, colspan=2,  size='16px')
        
        
        if animate:
            view.show()
            trn = trainer.itrain(input_data, iterval=iterval, monitor_keys=['slowFeatures'])
            dcosines = np.zeros([iterval*t.shape[0]/trainer.ticker,output_dim])
            timer = QtCore.QTimer()
            cnt=[0]
            _tcnt = time.time()
            def animloop():
                rtnval = trn.next()
                if not rtnval:
                    timer.stop()
                    print ''
                    print 'Total Time for %d iterations: '%(iterval), time.time()-_tcnt
                    print 'Success status: ', np.allclose(dcosines[cnt[0]-1], np.ones(output_dim), atol=1e-2)
                    print 'Maximum error: ', np.max(np.abs(np.ones(output_dim)-dcosines[cnt[0]-1]))
                    return
                v = trainer.monitorVar['slowFeatures'][0]
                out = node.execute(input_data)
                for dim in xrange(output_dim):
                    dcosines[cnt[0], dim] = np.abs(np.dot(v[dim], bv[dim].T))/(np.linalg.norm(v[dim])*np.linalg.norm(bv[dim]))
                
                for i in range(output_dim):
                    curves1[i].setData(dcosines[:cnt[0],i])

                for i in range(4):
                    curves2[i].setData(out[:,i])
 
                cnt[0]+=1
            timer.timeout.connect(animloop)
            timer.start(0)
        else:
            _tcnt = time.time()
            trainer.train(input_data, iterval=iterval, monitor_keys=['slowFeatures'])
            v = trainer.monitorVar['slowFeatures']
            dcosines = np.zeros([len(v),output_dim])
            for i in xrange(len(v)):
                for dim in xrange(output_dim):
                    dcosines[i, dim] = np.abs(np.dot(v[i][dim], bv[dim].T))/(np.linalg.norm(v[i][dim])*np.linalg.norm(bv[dim]))

            out = node.execute(input_data)
          
            for i in range(output_dim):
                curves1[i].setData(dcosines[:,i])

            for i in range(4):
                curves2[i].setData(out[:,i])
  
            view.show()
            print ''
            print 'Total Time for %d iterations: '%(iterval), time.time()-_tcnt
            print 'Success status: ', np.allclose(dcosines[-1], np.ones(output_dim), atol=1e-2)
            print 'Maximum error: ', np.max(np.abs(np.ones(output_dim)-dcosines[-1]))

        app.exec_()
    else:
        _tcnt = time.time()
        trainer.train(input_data, iterval=iterval, monitor_keys=['slowFeatures'])
        v = trainer.monitorVar['slowFeatures']
        dcosines = np.zeros([len(v),output_dim])
        for i in xrange(len(v)):
            for dim in xrange(output_dim):
                dcosines[i, dim] = np.abs(np.dot(v[i][dim], bv[dim].T))/(np.linalg.norm(v[i][dim])*np.linalg.norm(bv[dim]))

        print ''
        print 'Total Time for %d iterations: '%(iterval), time.time()-_tcnt
        print 'Success status: ', np.allclose(dcosines[-1], np.ones(output_dim), atol=1e-2)
        print 'Maximum error: ', np.max(np.abs(np.ones(output_dim)-dcosines[-1]))


        plt.figure(figsize=(12,5))
        fig = plt.gcf()
        fig.canvas.set_window_title('This example demonstrates the working of Incremental Slow Feature Analysis (IncSFA).')
        plt.subplot2grid((2,4), (0, 0), colspan=2, rowspan=2)
        plt.plot(dcosines)
        plt.ylim([0,1.1])
        plt.title('Direction Cosines')
        plt.legend(['V-%d'%(i+1) for i in xrange(output_dim)])
        plt.xlabel('Ticker')

        out = node.execute(input_data)
        
        plt.subplot2grid((2,4), (0, 2))
        plt.plot(out[:,0], 'b', linewidth=1.5)
        plt.title('SF Out-1')
        plt.subplot2grid((2,4), (0, 3))
        plt.plot(out[:,1], 'g', linewidth=1.5)
        plt.title('SF Out-2')
        plt.subplot2grid((2,4), (1, 2))
        plt.plot(out[:,2], 'r', linewidth=1.5)
        plt.title('SF Out-3')
        plt.subplot2grid((2,4), (1, 3))
        plt.plot(out[:,3], 'c', linewidth=1.5)
        plt.title('SF Out-4')

        plt.tight_layout()
        plt.show()
    
#######################################################################################
#######################################################################################

if __name__ == "__main__":
    testincsfa(1)
    



    

  
  
