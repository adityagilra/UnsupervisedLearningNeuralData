import numpy as np
import scipy.io

def loadDataSet(dataFileBase,interactionFactorIdx)
    # load the model generated dataset
    retinaData = scipy.io.loadmat(dataFileBase+'.mat')
    if interactionFactorIdx != 20:
        spikeRaster = retinaData['synthset']['smp'][0,0]
        referenceRates = retinaData['synthset']['mv0'][0,0][0]
        sampleRates = retinaData['synthset']['mv'][0,0][0]
    else:
        spikeRaster = retinaData['bint']
        spikeRaster = np.reshape(np.moveaxis(spikeRaster,0,-1),(160,-1))
    return spikeRaster

def loadFit(dataFileBase,nModes):
    dataBase = shelve.open(dataFileBase+'_mixmod_modes'+str(nModes)+'.shelve')
    wModes = dataBase['mixMod.wModes']
    mProb = dataBase['mixMod.mProb']
    QModes = dataBase['mixMod.QModes']
    logL = dataBase['mixMod.logL']
    dataBase.close()
    return wModes,mProb,Qmodes,logL

class MixtureModel():
    def __init__(self,nModes,spikeRaster):
        np.random.seed(1)
        self.nModes = nModes
        self.nNeurons,self.tSteps = spikeRaster.shape
        self.wModes = np.zeros(nModes)
        self.QModes = np.zeros((nModes,self.tSteps))
        self.mProb = np.zeros((nModes,self.nNeurons))
        self.PCond = np.random.uniform(size=(self.nModes,self.tSteps))
        self.spikeRaster = spikeRaster
        self.spikeRasterZeros = (spikeRaster == 0)

    def expectation(self):
        for mode in range(self.nModes):
            # np.broadcast_to() only creates a read-only view, need to copy to modify
            mProbBroadcast = np.copy(
                            np.broadcast_to( np.transpose([self.mProb[mode,:]]),
                                            (self.nNeurons,self.tSteps) ) )
            mProbBroadcast[self.spikeRasterZeros] = \
                            1. - mProbBroadcast[self.spikeRasterZeros]
            # eq (2) in Prentice et al 2016 supplementary
            self.QModes[mode,:] = np.product( mProbBroadcast, axis=0 )
        # eq (3) in Prentice et al 2016 supplementary
        self.PCond = (self.wModes[:, np.newaxis] * self.QModes) \
                        / np.dot(self.wModes,self.QModes)
        
    def maximization(self):
        # eqs (4) & (5) in Prentice et al 2016 supplementary
        self.wModes = 1./(self.tSteps+1.) * np.sum(self.PCond,axis=1)
        self.mProb = np.dot(self.PCond,self.spikeRaster.T) \
                        / np.sum(self.PCond,axis=1)[:, np.newaxis]

    def calcLogLikelihood(self):
        # eq (1) in Prentice et al 2016 supplementary -- taking log, and sum_t
        # sum_t log( P_mix({\sigma_i(t)}) )
        self.logL = np.sum(np.log( 
                        np.sum(self.wModes[:, np.newaxis] * self.QModes, axis=0) ))
        return self.logL

    def saveFit(self,dataFileBase):
        dataBase = shelve.open(dataFileBase+'_mixmod_modes'+str(self.nModes)+'.shelve')
        dataBase['mixMod.wModes'] = self.wModes
        dataBase['mixMod.mProb'] = self.mProb
        dataBase['mixMod.QModes'] = self.QModes
        dataBase['mixMod.logL'] = self.calcLogLikelihood()
        dataBase.close()
