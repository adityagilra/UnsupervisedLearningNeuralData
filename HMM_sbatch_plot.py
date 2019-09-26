import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import shelve, sys, shutil

from sklearn.manifold import MDS

np.random.seed(100)

dataFileBaseName = 'Learnability_data/synthset_samps'
interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.])
interactionFactorList = np.append(interactionFactorList,[1.])

figMM, axesMM = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
#figMM, axesMM = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
figMM2, axesMM2 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
figMM3, axesMM3 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
cm = plt.cm.get_cmap('RdYlBu')

maxModes = 150
nModesList = range(1,maxModes+1,5)

findBestNModes = True       # loop over all the nModes data
                            #  & find best nModes for each dataset
                            # must be done at least once before plotting
                            #  to generate _mixmodsummary.shelve

crossvalfold = 2            # usually 1 or 2 - depends on what you set when fitting
HMMstr = '_shuffled_HMM'+(str(crossvalfold) if crossvalfold>1 else '')
#HMMstr = '_HMM'+(str(crossvalfold) if crossvalfold>1 else '')

entropies = []
# loop through all the dataset fitting files and analyse them
for fileNum in range(22):
    if fileNum < 20:
        dataFileBase = dataFileBaseName + '_' + str(fileNum+1)
    elif fileNum == 20:
        dataFileBase = 'Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100'
    elif fileNum == 21:
        dataFileBase = 'Prenticeetal2016_data/unique_natural_movie/data'

    if findBestNModes:
        logLVec = np.zeros(len(nModesList))
        logLTestVec = np.zeros(len(nModesList))
        for idx,nModes in enumerate(nModesList):
            dataFileName = dataFileBase+HMMstr+'_modes'+str(nModes)+'.shelve'
            # I only need to look up logL
            print(dataFileName)
            dataBase = shelve.open(dataFileName, flag='r')
            #logL = dataBase['logli']
            logL = dataBase['train_logli']
            logLTest = dataBase['test_logli']
            dataBase.close()
            logLVec[idx] = np.mean([logL[k,-1] for k in range(crossvalfold)])
            logLTestVec[idx] = np.mean([logLTest[k,-1] for k in range(crossvalfold)])

        # find the best nModes for this dataset
        bestNModesIdx = np.argmax(logLTestVec)
        bestNModes = nModesList[bestNModesIdx]
        shutil.copyfile(dataFileBase+HMMstr+'_modes'+str(bestNModes)+'.shelve',
                        dataFileBase+HMMstr+'summary.shelve')
        # further add the best nModes for this dataset to the summary file
        dataBase = shelve.open(dataFileBase+HMMstr+'summary.shelve')
        dataBase['nModes'] = bestNModes
        dataBase['logLVec'] = logLVec
        dataBase['logLTestVec'] = logLTestVec
        dataBase['nModesList'] = nModesList
        dataBase['nModesIdx'] = bestNModesIdx
        dataBase.close()
        print("Finished looking up best nModes = ",bestNModes,
                    " mixture model for file number",fileNum)
        sys.stdout.flush()

    dataBase = shelve.open(dataFileBase+HMMstr+'summary.shelve', flag='r')
    bestNModes = dataBase['nModes']
    params = dataBase['params']
    trans = dataBase['trans']
    wModes = dataBase['stationary_prob'].T
    bestNModesIdx = dataBase['nModesIdx']
    logLVec = dataBase['logLVec']
    logLTestVec = dataBase['logLTestVec']
    dataBase.close()

    # params is a list (size nModes) of dicts, each dict has 'm' and 'J'
    mProb = np.zeros(shape=(bestNModes,len(params[0]['m'])))
    for i,param in enumerate(params):
        mProb[i,:] = param['m'].flatten()

    ax = axesMM[fileNum//5,fileNum%5]
    #ax = axesMM
    ax.plot(nModesList,logLVec,'k-,')
    ax.scatter(nModesList,logLVec,marker='x',color='k')
    ax.scatter(nModesList,logLTestVec,marker='*',color='b')
    ax.scatter([bestNModes],[logLTestVec[bestNModesIdx]],marker='o',color='r')
    ax.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[fileNum])+\
                        ', *nModes='+str(bestNModes))
    print('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[fileNum]),
            'logLTestVec=',logLTestVec,
            'logLVec=',logLVec)

    ax2 = axesMM2[fileNum//5,fileNum%5]
    #ax2 = axesMM2
    wModes = np.sort(wModes.flatten())[::-1]
    ax2.plot(wModes,'k-,')
    ax2.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[fileNum])+\
                        ', *nModes='+str(bestNModes))
    entropies.append(-np.sum(wModes*np.log(wModes)))

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
        s = ax3.scatter(x,y,c=np.log(wModes.flatten()),s=meanSpikesMode*20,cmap=cm)
        #s = ax3.scatter(x,y,s=meanSpikesMode*20,cmap=cm)
        ax3.set_xlabel('m (MDS dim1)')
        ax3.set_ylabel('m (MDS dim2)')
        ax3.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[fileNum])+\
                            ', *nModes='+str(bestNModes))

cbar = figMM3.colorbar(s)
#cbar.ax.set_title(r"log w")  
cbar.ax.set_ylabel(r"log w",  labelpad=20, rotation=270)

figMM.tight_layout()
figMM.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.5, hspace=0.5)
figMM2.tight_layout()
figMM2.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.5, hspace=0.5)
figMM3.tight_layout()
figMM3.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.5, hspace=0.5)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
axes.scatter(interactionFactorList[:20],entropies[:20],marker='x',color='k')
axes.scatter(interactionFactorList[20],entropies[20],marker='x',color='b')
axes.scatter(interactionFactorList[21],entropies[21],marker='x',color='r')
axes.set_xlabel('interaction factor $\\alpha$')
axes.set_ylabel('entropy of modes')

plt.show()
