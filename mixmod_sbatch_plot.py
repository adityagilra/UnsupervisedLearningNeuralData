import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import shelve, sys

from MixtureModel import MixtureModel

np.random.seed(100)

dataFileBaseName = 'Learnability_data/synthset_samps'
interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.])

figMM, axesMM = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
#figMM, axesMM = plt.subplots(nrows=1, ncols=1, figsize=(8,4))

maxModes = 150
nModesList = range(1,maxModes+1,5)

findBestNModes = True           # loop over all the nModes data & find best nModes for each dataset
                                # must be done at least once before plotting or LDA
                                #  to generate _mixmodsummary.shelve

# loop through all the generated data files and analyse them
for fileNum in range(14,20,1):  # to fit MixMod for a few files, LDA fit needs all files
    if fileNum != 20:
        dataFileBase = dataFileBaseName + '_' + str(fileNum+1)
    else:
        dataFileBase = 'IST-2017-61-v1+1_bint_fishmovie32_100'

    if findBestNModes:
        logLVec = np.zeros(len(nModesList))
        for idx,nModes in enumerate(nModesList):
            _,_,_,logL = loadFit(dataFileBase,nModes)
            logLVec[idx] = logL

        # load the best nModes for this dataset
        bestNModes = nModesList[np.argmax(logLVec)]
        wModes,mProb,Qmodes,logL = loadFit(dataFileBase,bestNModes)
        # save the best nModes for this dataset    
        dataBase = shelve.open(dataFileBase+'_mixmodsummary.shelve')
        dataBase['mixMod.wModes'] = wModes
        dataBase['mixMod.mProb'] = mProb
        dataBase['mixMod.QModes'] = QModes
        dataBase['mixMod.logL'] = logL
        dataBase['nModes'] = bestNMode
        dataBase['logLVec'] = logLVec
        dataBase.close()
        print("Finished mixture model fitting and saving for file number",fileNum)
        sys.stdout.flush()

    dataBase = shelve.open(dataFileBase+'_mixmodsummary.shelve')
    bestNMode = dataBase['nModes']
    logLVec = dataBase['logLVec']
    dataBase.close()
    ax = axesMM[fileNum//5,fileNum%5]
    #ax = axesMM
    ax.plot(nModesList,logLVec,'k-,')
    ax.scatter(nModesList,logLVec,marker='x',color='k')
    ax.scatter([bestNMode],[logLVec[bestNMode]],marker='o',color='r')
    ax.set_title('best nModes = '+str(bestNMode))

figMM.tight_layout()

plt.show()
