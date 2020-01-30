import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import shelve, sys, shutil

from sklearn.manifold import MDS
from sklearn import discriminant_analysis as DA
from sklearn.metrics import adjusted_rand_score

from EMBasins_sbatch_plot import saveBestNMode

np.random.seed(100)

HMM = False
shuffled = True
treeSpatial = True
crossvalfold = 2            # usually 1 or 2 - depends on what you set when fitting

findBestNModes = True       # loop over all the nModes data
                            #  & find best nModes for each dataset
                            # must be done at least once before plotting
                            #  to generate _summary.shelve

interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.,1.])
interactionsLen = len(interactionFactorList)

maxModes = 150
nModesList = range(1,maxModes+1,5)
nModesLen = len(nModesList)

seedList = range(5)
nSeeds = len(seedList)

figMM, axesMM = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
#figMM, axesMM = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
figMM2, axesMM2 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
figMM5, axes = plt.subplots(nrows=1, ncols=5, figsize=(16,4))
cm = plt.cm.get_cmap('RdYlBu')

dataFileBaseName = 'Learnability_data/generated_data_1'

EMBasinsStr = ('_shuffled' if shuffled else '') + \
                ('_HMM' + (str(crossvalfold) if crossvalfold>1 else '') \
                    if HMM else '_EMBasins_full') + \
                ('' if treeSpatial else '_notree')

entropies = np.zeros((interactionsLen,nSeeds))
logLs = np.zeros((interactionsLen,nSeeds))
logLsTest = np.zeros((interactionsLen,nSeeds))
logLVecs = np.zeros((interactionsLen,nSeeds,nModesLen))
logLVecsTest = np.zeros((interactionsLen,nSeeds,nModesLen))
bestNModesList = np.zeros((interactionsLen,nSeeds))

################## loop through all the dataset fitting files and analyse them ####################

for interactionFactorIdx in range(interactionsLen):
    interactionFactor = interactionFactorList[interactionFactorIdx]
    ax = axesMM[interactionFactorIdx//5,interactionFactorIdx%5]
    ax2 = axesMM2[interactionFactorIdx//5,interactionFactorIdx%5]
    for nSeed in seedList:
        if interactionFactorIdx < 20:
            dataFileBase = dataFileBaseName + '_' + str(interactionFactorIdx+1)\
                                            + '_' + str(nSeed)
        elif interactionFactorIdx == 20:
            dataFileBase = 'Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100'
        elif interactionFactorIdx == 21:
            dataFileBase = 'Prenticeetal2016_data/unique_natural_movie/data'
        print('\n')
        print('Dataset: ',dataFileBase)

        ################ find best nModes for a given dataset and save it in 'summary' file ###################
        if findBestNModes:
			# important to use summaryType='_simple', as saveBestNMode shutil.copyfile-s over the summary file,
			# EMBasis_sbatch_plot has its own summary file with summaryType='' where more summary data is present
			saveBestNMode(nModesList, dataFileBase, EMBasinsStr, crossvalfold, summaryType='_simple')

        ################ Read in the summary data for best nModes and pre-process for later analysis ################

        dataBase = shelve.open(dataFileBase+EMBasinsStr+'_summary_simple.shelve','r')

        bestNModes = dataBase['nModes']
        params = dataBase['params']

        bestNModesIdx = dataBase['nModesIdx']
        logLVec = dataBase['logLVec']
        logLTestVec = dataBase['logLTestVec']
            
        if HMM:
            trans = dataBase['trans']
            wModes = dataBase['stationary_prob'].flatten()
        else:
            wModes = dataBase['w'].flatten()
        dataBase.close()

        ######################### Analysis across best nMode fits #########################

        bestNModesList[interactionFactorIdx,nSeed] = bestNModes
        logLs[interactionFactorIdx,nSeed] = logLVec[bestNModesIdx]
        logLsTest[interactionFactorIdx,nSeed] = logLTestVec[bestNModesIdx]
        logLVecs[interactionFactorIdx,nSeed,:] = logLVec
        logLVecsTest[interactionFactorIdx,nSeed,:] = logLTestVec
        wModes = np.sort(wModes.flatten())[::-1]
        entropies[interactionFactorIdx,nSeed] = -np.sum(wModes*np.log(wModes))

        ax.scatter([bestNModes],[logLTestVec[bestNModesIdx]],marker='.',color='r')
        ax2.plot(wModes, marker='.',color='k')

    ax.errorbar(nModesList,np.mean(logLVecs[interactionFactorIdx,:,:],axis=0),
                            yerr=np.std(logLVecs[interactionFactorIdx,:,:],axis=0),
                            marker='.',color='k')
    ax.errorbar(nModesList,np.mean(logLVecsTest[interactionFactorIdx,:,:],axis=0),
                            yerr=np.std(logLVecsTest[interactionFactorIdx,:,:],axis=0),
                            marker='.',color='b')
    ax.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactor)+\
                        ', *nModes='+str(bestNModesList[interactionFactorIdx,:]))
    print('$\\alpha=$'+"{:1.1f}".format(interactionFactor),
            'logLTestVec for all random seeds = ',np.mean(logLVecsTest[interactionFactorIdx,:,:],axis=0))

    ax2.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[interactionFactorIdx])+\
                        ', *nModes='+str(bestNModesList[interactionFactorIdx,:]))


################# plot fitting (best nModes) summary scores versus datasets #################
ax1 = axes[0]
ax1.errorbar(interactionFactorList[:20],np.mean(entropies[:20,:],axis=1),
                    yerr=np.std(entropies[:20,:],axis=1),marker='x',color='k')
ax1.scatter(interactionFactorList[20],entropies[20,0],marker='x',color='b')
ax1.scatter(interactionFactorList[21],entropies[21,0],marker='x',color='r')
ax1.set_xlabel('interaction $\\alpha$')
ax1.set_ylabel('entropy of modes')

ax2 = axes[1]
ax2.errorbar(interactionFactorList[:20],np.mean(logLs[:20,:],axis=1),
                    yerr=np.std(logLs[:20,:],axis=1),marker='x',color='k')
ax2.scatter(interactionFactorList[20],logLs[20,0],marker='x',color='b')
ax2.scatter(interactionFactorList[21],logLs[21,0],marker='x',color='r')
ax2.set_xlabel('interaction $\\alpha$')
ax2.set_ylabel('log likelihood')

ax3 = axes[2]
ax3.errorbar(interactionFactorList[:20],np.mean(logLsTest[:20,:],axis=1),
                    yerr=np.std(logLsTest[:20,:],axis=1),marker='x',color='k')
ax3.scatter(interactionFactorList[20],logLsTest[20,0],marker='x',color='b')
ax3.scatter(interactionFactorList[21],logLsTest[21,0],marker='x',color='r')
ax3.set_xlabel('interaction $\\alpha$')
ax3.set_ylabel('test log likelihood')

figList = [figMM,figMM2,figMM5]
for fig in figList:
    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0.5, hspace=0.5)

plt.show()
