import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import scipy.io

import shelve, sys
from sklearn.manifold import MDS
from sklearn import discriminant_analysis as DA

np.random.seed(100)

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
interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.])
entropies = []
ldaScoresTraining = []
ldaScoresTest = []

# loop through all the generated data files and analyse them
for fileNum, interactionFactor in enumerate(interactionFactorList):
#for fileNum in range(14,20,1):  # to fit MixMod for a few files, LDA fit needs all files
    if fileNum != 20:
        dataFileBase = dataFileBaseName + '_' + str(fileNum+1)
    else:
        dataFileBase = 'IST-2017-61-v1+1_bint_fishmovie32_100'

    if fitMixMod:
        # load the model generated dataset
        retinaData = scipy.io.loadmat(dataFileBase+'.mat')
        if fileNum != 20:
            spikeRaster = retinaData['synthset']['smp'][0,0]
            referenceRates = retinaData['synthset']['mv0'][0,0][0]
            sampleRates = retinaData['synthset']['mv'][0,0][0]
        else:
            spikeRaster = retinaData['bint']
            spikeRaster = np.reshape(np.moveaxis(spikeRaster,0,-1),(160,-1))
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
        #modeDataLDA = LDA.fit_transform(mixMod.spikeRaster[:,:-mixMod.tSteps//4].T,
        #                                    modeDataLabels[:-mixMod.tSteps//4])
        #ldaScoresTraining.append( LDA.score(mixMod.spikeRaster[:,:-mixMod.tSteps//4].T,
        #                        modeDataLabels[:-mixMod.tSteps//4]) )
        ## fit a shuffled set of time points (not good if fitting some form of a temporal model)
        shuffled_idxs = np.random.permutation(np.arange(mixMod.tSteps,dtype=np.int32))
        modeDataLDA = LDA.fit_transform(mixMod.spikeRaster[:,shuffled_idxs[:-mixMod.tSteps//4]].T,
                                            modeDataLabels[shuffled_idxs[:-mixMod.tSteps//4]])
        ldaScoresTraining.append( LDA.score(mixMod.spikeRaster[:,shuffled_idxs[:-mixMod.tSteps//4]].T,
                                modeDataLabels[shuffled_idxs[:-mixMod.tSteps//4]]) )
        # modeDataLDA is (tSteps,)
        print('Linear discriminant analysis on all modes using ',
                      numComponents,' components, training score is',
                      ldaScoresTraining[-1])
        #ldaScoresTest.append( LDA.score(mixMod.spikeRaster[:,-mixMod.tSteps//4:].T,
        #                        modeDataLabels[-mixMod.tSteps//4:]) )
        ## score rest of the shuffled set of time points (not good if fitting some form of a temporal model)
        ldaScoresTest.append( LDA.score(mixMod.spikeRaster[:,shuffled_idxs[-mixMod.tSteps//4:]].T,
                                modeDataLabels[shuffled_idxs[-mixMod.tSteps//4:]]) )
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
    
    print('Interaction Factor List = ',interactionFactorList)
    print('Entropies List = ',entropies)
    print('Linear Discriminability Analysis Training Scores List = ',ldaScoresTraining)
    print('Linear Discriminability Analysis Test Scores List = ',ldaScoresTest)

    # first figure

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

    axes[0].plot(interactionFactorList[:-1],entropies[:-1],'k-,')
    axes[0].scatter([interactionFactorList[-1]],[entropies[-1]],marker='x',color='k')
    axes[0].set_xlabel( 'coupling factor' )
    axes[0].set_ylabel('entropy of modes')

    #axes[1].plot(interactionFactorList[:-1],ldaScoresTraining[:-1],'r-,',label='training')
    #axes[1].plot(interactionFactorList[:-1],ldaScoresTest[:-1],'b-,',label='test')
    #axes[1].scatter([interactionFactorList[-1]],[ldaScoresTraining[-1]],marker='x',color='r')
    #axes[1].scatter([interactionFactorList[-1]],[ldaScoresTest[-1]],marker='x',color='b')
    axes[1].plot(interactionFactorList[:-1],ldaScoresTest[:-1],'k-,',label='test')
    axes[1].scatter([interactionFactorList[-1]],[ldaScoresTest[-1]],marker='x',color='k')
    axes[1].set_xlabel( 'coupling factor' )
    axes[1].set_ylabel('linear decodability')
    #axes[1].legend()
    
    fig.tight_layout()
    fig.savefig('entropy_decodability_tradeoff.png',dpi=300)
    fig.savefig('entropy_decodability_tradeoff.pdf')

    # second figure

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
    # linear interpolation of points to enable smoother color transition
    fineInteractionFactors = np.linspace(interactionFactorList[0],interactionFactorList[-2],num=1000)
    fineLDAScoresTest = np.interp(fineInteractionFactors,interactionFactorList[:-1],ldaScoresTest[:-1])
    fineEntropies = np.interp(fineInteractionFactors,interactionFactorList[:-1],entropies[:-1])
    mainplot = axes.scatter(fineLDAScoresTest, fineEntropies,
                                c=fineInteractionFactors,
                                cmap=plt.cm.jet, marker='.')
    cax = fig.colorbar(mainplot,ax=axes)
    cax.set_label('coupling factor')
    normC = plt.Normalize(vmin=interactionFactorList[0],vmax=interactionFactorList[-2])
    axes.scatter(ldaScoresTest[-1],entropies[-1],
                    marker='x', s=100,
                    facecolor=plt.cm.jet(normC(1.)),
                    edgecolor='k', linewidth=5)
    axes.set_ylabel('entropy of modes' )
    axes.set_xlabel('linear decodability')
    axes.annotate( 'uncoupled\npopulation', xy=(ldaScoresTest[0],entropies[0]),
                    xytext=(ldaScoresTest[0],entropies[0]-0.2),
                    arrowprops=dict(facecolor='black', arrowstyle='->') )
    axes.annotate( 'strongly-coupled\npopulation', xy=(ldaScoresTest[-2],entropies[-2]),
                    xytext=(ldaScoresTest[-2]-0.3,entropies[-2]),
                    arrowprops=dict(facecolor='black', arrowstyle='->') )
    axes.annotate( 'retinal data', xy=(ldaScoresTest[-1],entropies[-1]),
                    xytext=(ldaScoresTest[-1],entropies[-1]+0.02) )

    fig.tight_layout()
    fig.savefig('entropy_decodability_tradeoff2.png',dpi=300)
    fig.savefig('entropy_decodability_tradeoff2.pdf')
    
    plt.show()
