import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import shelve, sys

from MixtureModel import loadFit

from sklearn.manifold import MDS

np.random.seed(100)

dataFileBaseName = 'Learnability_data/synthset_samps'
interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.])

figMM, axesMM = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
#figMM, axesMM = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
figMM2, axesMM2 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
figMM3, axesMM3 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
cm = plt.cm.get_cmap('RdYlBu')

maxModes = 150
nModesList = range(1,maxModes+1,5)

findBestNModes = False      # loop over all the nModes data
                            #  & find best nModes for each dataset
                            # must be done at least once before plotting
                            #  to generate _mixmodsummary.shelve

# loop through all the dataset fitting files and analyse them
for fileNum in range(21):
    if fileNum != 20:
        dataFileBase = dataFileBaseName + '_' + str(fileNum+1)
    else:
        dataFileBase = 'Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100'

    if findBestNModes:
        logLVec = np.zeros(len(nModesList))
        logLTestVec = np.zeros(len(nModesList))
        for idx,nModes in enumerate(nModesList):
            # not using loadFit() since I only need to look up logL
            print(dataFileBase+'_mixmod_modes'+str(nModes)+'.shelve')
            dataBase = shelve.open(dataFileBase+'_mixmod_modes'+str(nModes)+'.shelve')
            logL = dataBase['mixMod.logL']
            logLTest = dataBase['mixMod.logLTest']
            dataBase.close()
            logLVec[idx] = logL
            logLTestVec[idx] = logLTest

        # load the best nModes for this dataset
        bestNModesIdx = np.argmax(logLTestVec)
        bestNModes = nModesList[bestNModesIdx]
        wModes,mProb,logL,logLTest = loadFit(dataFileBase,bestNModes)
        # save the best nModes for this dataset
        dataBase = shelve.open(dataFileBase+'_mixmodsummary.shelve')
        dataBase['mixMod.wModes'] = wModes
        dataBase['mixMod.mProb'] = mProb
        dataBase['mixMod.logL'] = logL
        dataBase['mixMod.logLTest'] = logLTest
        dataBase['nModes'] = bestNModes
        dataBase['logLVec'] = logLVec
        dataBase['logLTestVec'] = logLTestVec
        dataBase['nModesList'] = nModesList
        dataBase['nModesIdx'] = bestNModesIdx
        dataBase.close()
        print("Finished looking up best nModes = ",bestNModes,
                    " mixture model for file number",fileNum)
        sys.stdout.flush()

    dataBase = shelve.open(dataFileBase+'_mixmodsummary.shelve')
    bestNModes = dataBase['nModes']
    wModes = dataBase['mixMod.wModes']
    mProb = dataBase['mixMod.mProb']
    bestNModesIdx = dataBase['nModesIdx']
    logLVec = dataBase['logLVec']
    logLTestVec = dataBase['logLTestVec']
    dataBase.close()

    ax = axesMM[fileNum//5,fileNum%5]
    #ax = axesMM
    ax.plot(nModesList,logLVec,'k-,')
    ax.scatter(nModesList,logLVec,marker='x',color='k')
    ax.scatter(nModesList,logLTestVec,marker='*',color='b')
    ax.scatter([bestNModes],[logLTestVec[bestNModesIdx]],marker='o',color='r')
    ax.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[fileNum])+\
                        ', *nModes='+str(bestNModes))
    print('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[fileNum]),
            'logLTestVec=',logLTestVec)

    ax2 = axesMM2[fileNum//5,fileNum%5]
    #ax2 = axesMM2
    ax2.plot(np.sort(wModes)[::-1],'k-,')
    ax2.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[fileNum])+\
                        ', *nModes='+str(bestNModes))

    ax3 = axesMM3[fileNum//5,fileNum%5]
    if not np.isnan(np.sum(mProb)):
        # calculate mean spike count for each mode and sort by mode weight:
        meanSpikesMode = np.sum(mProb,axis=1)
        sortedIdxs = np.argsort(wModes)[::-1]
        print("mean spike count (sorted) for each mode",
                    meanSpikesMode[sortedIdxs])
        # non-linear dimensionality reduction from nNeurons-dim to 2-dim
        #  using multi-dimensional scaling
        lowDimMDS = MDS(n_components=2)
        # mixMod.mProb has shape nModes x nNeurons
        lowDimData = lowDimMDS.fit_transform(mProb)
        # lowDimData has shape nModes x n_components
        x,y = lowDimData.T
        s = ax3.scatter(x,y,c=np.log(wModes),s=meanSpikesMode*20,cmap=cm)
        #ax3.set_xlabel('m (MDS dim1)')
        #ax3.set_ylabel('m (MDS dim2)')
        #cbar = figMM3.colorbar(s)
        #cbar.ax.set_title(r"log w")  
        #cbar.ax.set_ylabel(r"log w",  labelpad=20, rotation=270)
        ax3.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[fileNum])+\
                            ', *nModes='+str(bestNModes))

figMM.tight_layout()
figMM.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.5, hspace=0.5)
figMM2.tight_layout()
figMM2.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.5, hspace=0.5)
figMM3.tight_layout()
figMM3.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.5, hspace=0.5)

plt.show()
