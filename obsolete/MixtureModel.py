import numpy as np
import scipy.io
import shelve, sys

def loadDataSet(dataFileBase,interactionFactorIdx,shuffle=True):
    # load the model generated dataset
    retinaData = scipy.io.loadmat(dataFileBase+'.mat')
    if interactionFactorIdx < 20:
        spikeRaster = retinaData['synthset']['smp'][0,0]
        referenceRates = retinaData['synthset']['mv0'][0,0][0]
        sampleRates = retinaData['synthset']['mv'][0,0][0]
    elif interactionFactorIdx == 20:
        spikeRaster = retinaData['bint']
        spikeRaster = np.reshape(np.moveaxis(spikeRaster,0,-1),(160,-1))
    nNeurons,tSteps = spikeRaster.shape
    if shuffle:
        # randomly permute the full dataset
        # careful if fitting a temporal model and/or retina has adaptation
        shuffled_idxs = np.random.permutation(np.arange(tSteps,dtype=np.int32))
        spikeRaster = spikeRaster[:,shuffled_idxs]        
    return spikeRaster

class MixtureModel():
    def __init__(self,nModes,spikeRaster,tStepsFit,tStepsTest,loadParamsFileBase=None):
        np.random.seed(1)
        self.nModes = nModes
        self.nNeurons,self.tSteps = spikeRaster.shape
        if loadParamsFileBase is None:
            # initialize wModes and mProb as per Prentice et al 2016 suppl section 2.b.1
            self.wModes = np.ones(nModes)/np.float(nModes)
            self.mProb = np.random.uniform(0.45,0.55,size=(nModes,self.nNeurons))
        else:
            dataBase = shelve.open(loadParamsFileBase+'_mixmod_modes'\
                                        +str(nModes)+'.shelve')
            self.wModes = dataBase['mixMod.wModes']
            self.mProb = dataBase['mixMod.mProb']
            dataBase.close()
        
        # if you start with maximization before expectation,
        #  then this PCond is used, serves as initialization
        # if you start with expectation before maximization (as I have),
        #  then this PCond is ignored, rather wModes and mProb are used
        self.PCond = np.random.uniform(size=(self.nModes,tStepsFit))
        
        self.spikeRaster = spikeRaster
        self.spikeRasterFit = spikeRaster[:, :tStepsFit]
        self.spikeRasterTest = spikeRaster[:, tStepsFit:tStepsFit+tStepsTest]
        self.spikeRasterFitZeros = (self.spikeRasterFit == 0)
        self.eps = np.finfo(np.float).eps

    def calcModePosterior(self,spikeRaster,spikeRasterZeros):
        ## method 1: using numpy broadcasting: (concise, but uses >24GB RAM, unclear if faster)
        ##  https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        ##  with newaxis, mProb becomes nModes x nNeurons x 1
        ##  spikeRaster is nNeurons x tSteps
        ##  under numpy broadcasting, dimensions of size 1 in both arrays get expanded
        ##  mProb and spikeRaster get expanded to nModes x nNeurons x tSteps
        ## further assume spikeRaster only contains 0s and 1s
        #QModesExtra = self.mProb[:,:,np.newaxis] * spikeRaster + \
        #                (1.-self.mProb[:,:,np.newaxis]) * (1.-spikeRaster)
        #QModes = np.product( QModesExtra, axis=1 )     # multiply along neurons
        
        # method 2: complex code, but uses < 5 GB RAM, but maybe slower?
        _,tSteps = spikeRaster.shape
        QModes = np.zeros((self.nModes,tSteps))
        for mode in range(self.nModes):
            # np.broadcast_to() only creates a read-only view, need to copy to modify
            mProbBroadcast = np.copy(
                            np.broadcast_to( np.transpose([self.mProb[mode,:]]),
                                                        (self.nNeurons,tSteps) ) )
            mProbBroadcast[spikeRasterZeros] = \
                            1. - mProbBroadcast[spikeRasterZeros]
            # eq (2) in Prentice et al 2016 supplementary
            QModes[mode,:] = np.product( mProbBroadcast, axis=0 )

        # eq (3) in Prentice et al 2016 supplementary
        # raise division by zero as FloatingPointError, default is silent
        #with np.errstate(invalid='raise'):
        # self.eps where 0, else I get nan-s and -inf-s, due to division by zero
        Z = np.dot(self.wModes,QModes)
        Z[Z<self.eps] = self.eps
        PCond = (self.wModes[:, np.newaxis] * QModes) / Z
        return PCond, Z

    def expectation(self):
        self.PCond, _ = self.calcModePosterior(self.spikeRasterFit,
                                                self.spikeRasterFitZeros)
        
    def maximization(self):
        # eqs (4) & (5) in Prentice et al 2016 supplementary
        self.wModes = 1./(self.tSteps+1.) * np.sum(self.PCond,axis=1)
        normPC = np.sum(self.PCond,axis=1)
        normPC[normPC<self.eps] = self.eps
        self.mProb = np.dot(self.PCond,self.spikeRasterFit.T) \
                        / (normPC[:, np.newaxis])

    def calcLogLikelihood(self,spikeRaster=None):
        if spikeRaster is None:
            spikeRaster = self.spikeRasterFit
            spikeRasterZeros = self.spikeRasterFitZeros
        else:
            spikeRasterZeros = (spikeRaster == 0)
        _, Z = self.calcModePosterior(spikeRaster,spikeRasterZeros)
        
        # eq (1) in Prentice et al 2016 supplementary, product over t
        #  further take log, and sum_t
        # sum_t log( P_mix({\sigma_i(t)}) )
        logL = np.sum(np.log( Z ))
        return logL

    def saveFit(self,dataFileBase):
        dataBase = shelve.open(dataFileBase+'_mixmod_modes'+str(self.nModes)+'.shelve')
        dataBase['mixMod.wModes'] = self.wModes
        dataBase['mixMod.mProb'] = self.mProb
        logL = self.calcLogLikelihood()
        logLTest = self.calcLogLikelihood(self.spikeRasterTest)
        dataBase['mixMod.logL'] = logL
        dataBase['mixMod.logLTest'] = logLTest
        dataBase.close()
        return logL, logLTest

def loadFit(dataFileBase,nModes):
    dataBase = shelve.open(dataFileBase+'_mixmod_modes'+str(nModes)+'.shelve')
    wModes = dataBase['mixMod.wModes']
    mProb = dataBase['mixMod.mProb']
    logL = dataBase['mixMod.logL']
    logLTest = dataBase['mixMod.logLTest']
    dataBase.close()
    return wModes,mProb,logL,logLTest
