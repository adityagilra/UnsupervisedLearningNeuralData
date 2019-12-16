import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import shelve, sys, shutil

from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from EMBasins_sbatch import loadDataSet

np.random.seed(100)

HMMShuffled = False
EMBasinsShuffled = True
treeSpatial = True
crossvalfold = 2            # usually 1 or 2 - depends on what you set when fitting
WTATrainIter = 1            # number of training dataset repeats when running WTAcluster_sbatch.py

# fits are available for nModes in steps of 5 from 1, i.e. 1,6,11,16,21,...
nModes = 6#56

doMDS_else_PCA = False      # if True do MDS, else PCA
lowDimStr = ('MDS' if doMDS_else_PCA else 'PCA')

dataFileBaseName = 'Learnability_data/synthset_samps'
interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.,1.])

figMM, axesMM = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
cm = plt.cm.get_cmap('RdYlBu')

HMMStr = ('_shuffled' if HMMShuffled else '') + \
                '_HMM' + (str(crossvalfold) if crossvalfold>1 else '') + \
                ('' if treeSpatial else '_notree')

EMBasinsStr = ('_shuffled' if EMBasinsShuffled else '') + \
                '_EMBasins_full' + \
                ('' if treeSpatial else '_notree')

interactionsLen = len(interactionFactorList)

################## loop through all the dataset fitting files and analyse them ####################

interactionFactorIdx = 21

if interactionFactorIdx < 20:
    dataFileBase = dataFileBaseName + '_' + str(interactionFactorIdx+1)
elif interactionFactorIdx == 20:
    dataFileBase = 'Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100'
elif interactionFactorIdx == 21:
    dataFileBase = 'Prenticeetal2016_data/unique_natural_movie/data'
print('\n')
print('Dataset: ',dataFileBase)

############## MDS for EMBasins and HMM ################
for idx,fitStr in enumerate(('EMBasins','HMM')):
    if fitStr == 'EMBasins':
        ### load the fits for EMBasins
        dataBase = shelve.open(dataFileBase+EMBasinsStr+'_modes'+str(nModes)+'.shelve')
        wModes = dataBase['w'].flatten()
    else:
        ### load the fits for HMM
        print(dataFileBase+HMMStr+'_modes'+str(nModes)+'.shelve')
        dataBase = shelve.open(dataFileBase+HMMStr+'_modes'+str(nModes)+'.shelve')
        wModes = dataBase['stationary_prob'].flatten()
    params = dataBase['params']
    dataBase.close()
    print('wModes ',fitStr,' = ',wModes)

    # for the TreeBasin spatially-correlated model (set in EMBasins.cpp)
    #  params is a list (size nModes) of dicts, each dict has 'm' and 'J'
    # for the IndependentBasin spatially-uncorrelated model (set in EMBasins.cpp)
    #  params is a list (size nModes) of dicts, each dict has 'm'
    mProb = np.zeros(shape=(nModes,len(params[0]['m'])))
    for i,param in enumerate(params):
        mProb[i,:] = param['m'].flatten()

    ax = axesMM[idx]
    if not np.isnan(np.sum(mProb)):
        # calculate mean spike count for each mode and sort by mode weight:
        meanSpikesMode = np.sum(mProb,axis=1)
        sortedIdxs = np.argsort(wModes)[::-1]
        print(sortedIdxs)
        print("mean spike count (sorted) for each fitted mode",
                    meanSpikesMode[sortedIdxs])

        if doMDS_else_PCA:
            # non-linear dimensionality reduction from nNeurons-dim to 2-dim
            #  using multi-dimensional scaling
            # MDS doesn't have transform(), only fit() and fit_transform()
            lowDimMDS = MDS(n_components=2)
            lowDimData = lowDimMDS.fit_transform(mProb)
        else:
            # mProb has shape nModes x nNeurons
            #lowDimData = lowDimMDS.fit_transform(mProb)
            if fitStr == 'EMBasins':
                # PCA has transform() as well apart from fit() and fit_transform()
                #  so can transform different datasets to a common low-dim space
                lowDimPCA = PCA(n_components=2)
                lowDimData = lowDimPCA.fit(mProb)
            # find the low-dim projection onto the PCA subspace
            lowDimData = lowDimPCA.transform(mProb)
        # lowDimData has shape nModes x n_components
        x,y = lowDimData.T
        # plot in reverse order of importance, so that most important is plotted on top!
        s = ax.scatter(x[sortedIdxs][::-1],y[sortedIdxs][::-1],
                        c=np.log(wModes.flatten())[sortedIdxs][::-1],
                        s=meanSpikesMode[sortedIdxs][::-1]*10,
                        cmap=cm)
        ax.set_xlabel('m ('+lowDimStr+' dim1)')
        ax.set_ylabel('m ('+lowDimStr+' dim2)')
        ax.set_title(fitStr + ' $\\alpha=$'+"{:1.1f}".format(interactionFactorList[interactionFactorIdx])+\
                            ', *nModes='+str(nModes))
        cbar = figMM.colorbar(s,ax=ax)
        cbar.ax.set_ylabel(r"log w",  labelpad=20, rotation=270)

############ MDS for WTA #################
ax = axesMM[2]

dataBase = shelve.open(dataFileBase+HMMStr+'_WTA'+str(WTATrainIter)+'_modes'+str(nModes)+'.shelve')
# readout_test is numberofmodes x timebins
readout_test = dataBase['readout_test']
dataBase.close()
# readout_test is numberofmodes x timebins
labelsWTA = np.argmax(readout_test,axis=0).T

### load the spike rasters from the dataset
# shuffling doesn't matter here as spikeRaster is only used for WTA MDS
spikeRaster = loadDataSet(dataFileBase, interactionFactorIdx, shuffle=False)
nNeurons,tSteps = spikeRaster.shape

mProbWTA = np.zeros((nModes,nNeurons))
wModesWTA = np.zeros(nModes)
for modenum in range(nModes):
    binIdxs = np.where(labelsWTA == modenum)[0]
    wModesWTA[modenum] = float(len(binIdxs))/len(labelsWTA)
    # here mProb[modenum,i] is the prob ( neuron i spikes | mode=modenum )
    mProbWTA[modenum,:] = np.mean(spikeRaster[:,binIdxs],axis=1)
print('wModes WTA = ',wModesWTA)
# calculate mean spike count for each mode and sort by mode weight:
meanSpikesMode = np.sum(mProbWTA,axis=1)
sortedIdxs = np.argsort(wModesWTA)[::-1]
print("mean spike count (sorted) for each WTA mode",
                meanSpikesMode[sortedIdxs])
if not np.isnan(np.sum(mProbWTA)):
    if doMDS_else_PCA:
        # non-linear dimensionality reduction from nNeurons-dim to 2-dim
        #  using multi-dimensional scaling
        lowDimMDS = MDS(n_components=2)
        # mProbWTA has shape nModes x nNeurons
        lowDimData = lowDimMDS.fit_transform(mProbWTA)
    else:
        lowDimData = lowDimPCA.transform(mProbWTA)
    # lowDimData has shape nModes x n_components
    x,y = lowDimData.T
    # plot in reverse order of importance, so that most important is plotted on top!
    s = ax.scatter(x[sortedIdxs][::-1],y[sortedIdxs][::-1],
                        c=np.log(wModesWTA.flatten())[sortedIdxs][::-1],
                        s=meanSpikesMode[sortedIdxs][::-1]*10,
                        cmap=cm)
    ax.set_xlabel('m ('+lowDimStr+' dim1)')
    ax.set_xlabel('m ('+lowDimStr+' dim2)')
    ax.set_title('WTA $\\alpha=$'+"{:1.1f}".format(interactionFactorList[interactionFactorIdx])+\
                        ', *nModes='+str(nModes))
    cbar = figMM.colorbar(s,ax=ax)
    cbar.ax.set_ylabel(r"log w",  labelpad=20, rotation=270)

for fig in [figMM]:
    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0.5, hspace=0.5)

plt.show()
