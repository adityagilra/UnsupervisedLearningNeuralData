import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import shelve, sys

from MixtureModel import loadFit

np.random.seed(100)

dataFileBaseName = 'Learnability_data/synthset_samps'
interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.])

figMM, axesMM = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
#figMM, axesMM = plt.subplots(nrows=1, ncols=1, figsize=(8,4))

maxModes = 150
nModesList = range(1,maxModes+1,5)

findBestNModes = False      # loop over all the nModes data & find best nModes for each dataset
                            # must be done at least once before plotting or LDA
                            #  to generate _mixmodsummary.shelve

# loop through all the dataset fitting files and analyse them
for fileNum in range(21):
    if fileNum != 20:
        dataFileBase = dataFileBaseName + '_' + str(fileNum+1)
    else:
        dataFileBase = 'Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100'

    if findBestNModes:
        logLVec = np.zeros(len(nModesList))
        for idx,nModes in enumerate(nModesList):
            # not using loadFit() since I only need to look up logL
            dataBase = shelve.open(dataFileBase+'_mixmod_modes'+str(nModes)+'.shelve')
            logL = dataBase['mixMod.logL']
            dataBase.close()
            logLVec[idx] = logL

        # load the best nModes for this dataset
        bestNModesIdx = np.argmax(logLVec)
        bestNModes = nModesList[bestNModesIdx]
        wModes,mProb,QModes,logL = loadFit(dataFileBase,bestNModes)
        # save the best nModes for this dataset    
        dataBase = shelve.open(dataFileBase+'_mixmodsummary.shelve')
        dataBase['mixMod.wModes'] = wModes
        dataBase['mixMod.mProb'] = mProb
        dataBase['mixMod.QModes'] = QModes
        dataBase['mixMod.logL'] = logL
        dataBase['nModes'] = bestNModes
        dataBase['logLVec'] = logLVec
        dataBase['nModesList'] = nModesList
        dataBase['nModesIdx'] = bestNModesIdx
        dataBase.close()
        print("Finished looking up best nModes = ",bestNModes,
                    " mixture model for file number",fileNum)
        sys.stdout.flush()

    dataBase = shelve.open(dataFileBase+'_mixmodsummary.shelve')
    bestNModes = dataBase['nModes']
    bestNModesIdx = dataBase['nModesIdx']
    logLVec = dataBase['logLVec']
    dataBase.close()
    ax = axesMM[fileNum//5,fileNum%5]
    #ax = axesMM
    ax.plot(nModesList,logLVec,'k-,')
    ax.scatter(nModesList,logLVec,marker='x',color='k')
    ax.scatter([bestNModes],[logLVec[bestNModesIdx]],marker='o',color='r')
    ax.set_title('best nModes = '+str(bestNModes))

figMM.tight_layout()

plt.show()
