import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import shelve
from sklearn.manifold import MDS
from sklearn import discriminant_analysis as DA

# whether to fit mixture models and save data
# this takes the longest time, so beware
fitMixMod = False#True
# whether to load mixture model data and do linear discriminant analysis
fitLDA = False#True
# whether to load LDA data and plot
plotData = True

class MixtureModel():
    def __init__(self,nModes,spikeRaster):
        np.random.seed(1)
        self.nModes = nModes
        self.nNeurons,self.tSteps = spikeRaster.shape
        self.wModes = np.zeros(nModes)
        self.spikeRaster = spikeRaster # nNeurons x tSteps
        self.oneMinusSpikeRaster = 1. - spikeRaster
        self.QModes = np.zeros((nModes,self.tSteps))
        self.mProb = np.zeros((nModes,nNeurons))
        self.PCond = np.random.uniform(size=(self.nModes,self.tSteps))

    def expectation(self):
        zeroesVec = self.spikeRaster == 0
        for mode in range(self.nModes):
            # np.broadcast_to() only creates a read-only view, need to copy to modify
            mProbBroadcast = np.copy(
                            np.broadcast_to( np.transpose([self.mProb[mode,:]]),
                                            (self.nNeurons,self.tSteps) ) )
            mProbBroadcast[zeroesVec] = 1. - mProbBroadcast[zeroesVec]
            self.QModes[mode,:] = np.product( mProbBroadcast, axis=0 )
        self.PCond = (self.wModes[:, np.newaxis] * self.QModes) \
                        / np.dot(self.wModes,self.QModes)
        
    def maximization(self):
        self.wModes = 1./(self.tSteps+1.) * np.sum(self.PCond,axis=1)
        self.mProb = np.dot(self.PCond,self.spikeRaster.T) \
                        / np.sum(self.PCond,axis=1)[:, np.newaxis]

dataFileBaseName = 'Learnability_data/synthset_samps'
#dataFileBase = 'IST-2017-61-v1+1_bint_fishmovie32_100'
interactionFactorList = np.arange(0.,2.,0.1)
entropies = []
ldaScoresTraining = []
ldaScoresTest = []

# loop through all the generated data files and analyse them
for fileNum, interactionFactor in enumerate(interactionFactorList):
#for fileNum in range(14,20,1):  # to fit MixMod for a few files, LDA fit needs all files
    dataFileBase = dataFileBaseName + '_' + str(fileNum+1)

    if fitMixMod:
        # load the model generated dataset
        retinaData = scipy.io.loadmat(dataFileBase+'.mat')
        spikeRaster = retinaData['synthset']['smp'][0,0]
        referenceRates = retinaData['synthset']['mv0'][0,0][0]
        sampleRates = retinaData['synthset']['mv'][0,0][0]
        nNeurons,tSteps = spikeRaster.shape
        ## find unique spike patterns and their counts
        #spikePatterns, patternCounts = np.unique(spikeRaster, return_counts=True, axis=1)

        mixMod = MixtureModel(nModes=70,spikeRaster=spikeRaster)
        nRepeat = 40
        for i in range(nRepeat):
            mixMod.maximization()
            mixMod.expectation()
            print("Mixture model fitting for file number",fileNum,"repeat",i)
        
        # Save the fitted model
        dataBase = shelve.open(dataFileBase+'_mixmod.shelve')
        dataBase['mixMod'] = mixMod
        dataBase.close()
        
        print("Finished mixture model fitting and saving for file number",fileNum)
        
    if fitLDA:
        dataBase = shelve.open(dataFileBase+'_mixmod.shelve')
        mixMod = dataBase['mixMod']
        dataBase.close()
        
        print("Loading and analyzing the fitted mixture model for file number",fileNum)

        # compute entropy of mode patterns
        entropies.append( -np.sum(mixMod.wModes*np.log(mixMod.wModes)) )
        print("Entropy of mode patterns is ",entropies[-1])

        # Linear Discriminant Analysis
        # Qmodes is nModes x tSteps, modeDataLabels is (tSteps,)
        modeDataLabels = np.argmax(mixMod.QModes,axis=0).T
        # spikeRaster is nNeurons x tSteps
        # LDA.fit_transform() requires samples x features i.e. tSteps x nNeurons
        # number of components doesn't matter in the least for accuracy!
        numComponents = 1
        LDA = DA.LinearDiscriminantAnalysis(n_components=numComponents)
        modeDataLDA = LDA.fit_transform(mixMod.spikeRaster[:,:-mixMod.tSteps//4].T,
                                            modeDataLabels[:-mixMod.tSteps//4])
        ldaScoresTraining.append( LDA.score(mixMod.spikeRaster[:,:-mixMod.tSteps//4].T,
                                modeDataLabels[:-mixMod.tSteps//4]) )
        # modeDataLDA is (tSteps,)
        print('Linear discriminant analysis on all modes using ',
                      numComponents,' components, training score is',
                      ldaScoresTraining[-1])
        ldaScoresTest.append( LDA.score(mixMod.spikeRaster[:,-mixMod.tSteps//4:].T,
                                modeDataLabels[-mixMod.tSteps//4:]) )
        print('Linear discriminant analysis on all modes using ',
                      numComponents,' components, test score is',
                      ldaScoresTest[-1])
                      
# Save the fitted LDA data
if fitLDA:
    dataBase = shelve.open(dataFileBaseName+'_LDA.shelve')
    dataBase['entropies'] = entropies
    dataBase['ldaScoresTraining'] = ldaScoresTraining
    dataBase['ldaScoresTest'] = ldaScoresTest
    dataBase.close()        

# plotting
if plotData:
    dataBase = shelve.open(dataFileBaseName+'_LDA.shelve')
    entropies = dataBase['entropies']
    ldaScoresTraining = dataBase['ldaScoresTraining']
    ldaScoresTest = dataBase['ldaScoresTest']
    dataBase.close()        

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

    axes[0].plot(interactionFactorList,entropies,'k-,')
    axes[0].set_xlabel( 'interaction factor' )
    axes[0].set_ylabel('entropy of modes')

    axes[1].plot(interactionFactorList,ldaScoresTraining,'r-,',label='training')
    axes[1].plot(interactionFactorList,ldaScoresTraining,'b-,',label='test')
    axes[1].set_xlabel( 'interaction factor' )
    axes[1].set_ylabel('LDA score')
    axes[1].legend()
    
    fig.tight_layout()
    
    plt.show()
