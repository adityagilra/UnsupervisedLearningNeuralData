import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import scipy.io

import shelve, sys
from sklearn.manifold import MDS
from sklearn import discriminant_analysis as DA

from MixtureModel import loadDataSet

np.random.seed(100)

# whether to load mixture model data and do linear discriminant analysis
fitLDA = False#True

# whether to load LDA data and plot
plotLDAData = False#True

if fitLDA:
    entropies = []
    ldaScoresTraining = []
    ldaScoresTest = []

    # loop through all the generated data files and analyse them
    for fileNum, interactionFactor in enumerate(interactionFactorList):
        if fileNum != 20:
            dataFileBase = dataFileBaseName + '_' + str(fileNum+1)
        else:
            dataFileBase = 'IST-2017-61-v1+1_bint_fishmovie32_100'

        spikeRaster = loadDataSet(dataFileBase, interactionFactorIdx)
        nNeurons,tSteps = spikeRaster.shape

        # LDA fit needs the best nModes fit saved in _mixmodsummary.shelve
        dataBase = shelve.open(dataFileBase+'_mixmodsummary.shelve')
        wModes = dataBase['mixMod.wModes']
        mProb = dataBase['mixMod.mProb']
        QModes = dataBase['mixMod.QModes']
        logL = dataBase['mixMod.logL']
        dataBase.close()
        nModes,tStepsUsed = QModes.shape
        
        print("Loading and analyzing the fitted mixture model for file number",interactionFactorIdx)

        # compute entropy of mode patterns
        entropies.append( -np.sum(wModes*np.log(wModes)) )
        print("Entropy of mode patterns is ",entropies[-1])

        # Linear Discriminant Analysis
        # Qmodes is nModes x tStepsUsed, modeDataLabels is (tStepsUsed,)
        modeDataLabels = np.argmax(QModes,axis=0).T
        # spikeRaster is nNeurons x tSteps
        # LDA.fit_transform() requires samples x features i.e. tStepsUsed x nNeurons
        # number of components doesn't matter in the least for accuracy!
        numComponents = 1
        LDA = DA.LinearDiscriminantAnalysis(n_components=numComponents)
        #modeDataLDA = LDA.fit_transform(spikeRaster[:,:-tSteps//4].T,
        #                                    modeDataLabels[:-tSteps//4])
        #ldaScoresTraining.append( LDA.score(spikeRaster[:,:-tSteps//4].T,
        #                        modeDataLabels[:-tSteps//4]) )
        ## fit a shuffled set of time points (not good if fitting some form of a temporal model)
        shuffled_idxs = np.random.permutation(np.arange(tSteps,dtype=np.int32))
        modeDataLDA = LDA.fit_transform(spikeRaster[:,shuffled_idxs[:-tSteps//4]].T,
                                            modeDataLabels[shuffled_idxs[:-tSteps//4]])
        ldaScoresTraining.append( LDA.score(spikeRaster[:,shuffled_idxs[:-tSteps//4]].T,
                                modeDataLabels[shuffled_idxs[:-tSteps//4]]) )
        # modeDataLDA is (tSteps,)
        print('Linear discriminant analysis on all modes using ',
                      numComponents,' components, training score is',
                      ldaScoresTraining[-1])
        #ldaScoresTest.append( LDA.score(spikeRaster[:,-tSteps//4:].T,
        #                        modeDataLabels[-tSteps//4:]) )
        ## score rest of the shuffled set of time points (not good if fitting some form of a temporal model)
        ldaScoresTest.append( LDA.score(spikeRaster[:,shuffled_idxs[-tSteps//4:]].T,
                                modeDataLabels[shuffled_idxs[-tSteps//4:]]) )
        print('Linear discriminant analysis on all modes using ',
                      numComponents,' components, test score is',
                      ldaScoresTest[-1])

    # Save the fitted LDA data
    dataBase = shelve.open(dataFileBaseName+'_LDA.shelve')
    dataBase['entropies'] = entropies
    dataBase['ldaScoresTraining'] = ldaScoresTraining
    dataBase['ldaScoresTest'] = ldaScoresTest
    dataBase.close()        

# plotting
if plotLDAData:
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
