import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import shelve, sys, shutil

from sklearn.manifold import MDS
from sklearn import discriminant_analysis as DA
from sklearn.metrics import adjusted_rand_score

from EMBasins_sbatch import loadDataSet,spikeRasterToSpikeTimes,spikeTimesToSpikeRaster

np.random.seed(100)

HMM = True#False
shuffled = False#True
treeSpatial = True

findBestNModes = False      # loop over all the nModes data
                            #  & find best nModes for each dataset
                            # must be done at least once before plotting
                            #  to generate _mixmodsummary.shelve
crossvalfold = 2            # usually 1 or 2 - depends on what you set when fitting

plotMeanRates = False       # plot mean rates of samples from fit and those from dataset

assignModesToData = False   # read in modes of spike patterns from fit
                            #  assign and save modes to each timebin in dataset
                            #  (need to do only once after fitting)

doMDS = True                # do MDS (multi-dimensional scaling) i.e. dim-redux on
                            #  prob ( neuralspiking | mode )

doLDA = False               # use modes for each timebin as labels
                            #  and do linear discriminant analysis

cfWTAresults = True        # use modes for each timebin as labels
                            #  and compare clustering with winner take all
                            #  (run before using WTAcluster_sbatch.py)

doMDSWTA = True             # do MDS (multi-dimensional scaling) i.e. dim-redux on
                            #  mean neural firing | mode from WTA clustering

def loadFit(dataFileBase,nModes):
    dataBase = shelve.open(dataFileBase+EMBasinsStr+'_modes'+str(nModes)+'.shelve')
    params = dataBase['params']
    w = dataBase['w']
    P = dataBase['P']
    prob = dataBase['prob']
    logli = dataBase['logli']
    P_test = dataBase['P_test']
    dataBase.close()
    return params,w,P,prob,logli,P_test

dataFileBaseName = 'Learnability_data/synthset_samps'
interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.,1.])

figMM, axesMM = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
#figMM, axesMM = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
figMM2, axesMM2 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
if doMDS:
    figMM3, axesMM3 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
if plotMeanRates:
    figMM4, axesMM4 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
if doLDA:
    figMM6, axesMM6 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
if doMDSWTA:
    figMM7, axesMM7 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
figMM5, axes = plt.subplots(nrows=1, ncols=5, figsize=(16,4))
cm = plt.cm.get_cmap('RdYlBu')

maxModes = 150
nModesList = range(1,maxModes+1,5)

EMBasinsStr = ('_shuffled' if shuffled else '') + \
                ('_HMM' + (str(crossvalfold) if crossvalfold>1 else '') \
                    if HMM else '_EMBasins_full') + \
                ('' if treeSpatial else '_notree')

interactionsLen = len(interactionFactorList)
entropies = np.zeros(interactionsLen)
logLs = np.zeros(interactionsLen)
logLsTest = np.zeros(interactionsLen)
LDAtrain = np.zeros(interactionsLen)
LDAtest = np.zeros(interactionsLen)
WTAscores = np.zeros(interactionsLen)

################## loop through all the dataset fitting files and analyse them ####################

for interactionFactorIdx in [20,21]:#range(interactionsLen):
    if interactionFactorIdx < 20:
        dataFileBase = dataFileBaseName + '_' + str(interactionFactorIdx+1)
    elif interactionFactorIdx == 20:
        dataFileBase = 'Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100'
    elif interactionFactorIdx == 21:
        dataFileBase = 'Prenticeetal2016_data/unique_natural_movie/data'
    print('\n')
    print('Dataset: ',dataFileBase)

    ################ find best nModes for a given dataset and save it in 'summary' file ###################
    if findBestNModes:
        logLVec = np.zeros(len(nModesList))
        logLTestVec = np.zeros(len(nModesList))
        for idx,nModes in enumerate(nModesList):
            # not using loadFit() since I only need to look up logL
            print(dataFileBase+EMBasinsStr+'_modes'+str(nModes)+'.shelve')
            dataBase = shelve.open(dataFileBase+EMBasinsStr+'_modes'+str(nModes)+'.shelve')
            logL = dataBase['train_logli']
            logLTest = dataBase['test_logli']
            if HMM:
                logLVec[idx] = np.mean([logL[k,-1] for k in range(crossvalfold)])
                logLTestVec[idx] = np.mean([logLTest[k,-1] for k in range(crossvalfold)])
            else:
                logLVec[idx] = logL[0,-1]
                logLTestVec[idx] = logLTest[0,-1]
            dataBase.close()

        # find the best nModes for this dataset
        bestNModesIdx = np.argmax(logLTestVec)
        bestNModes = nModesList[bestNModesIdx]
        shutil.copyfile(dataFileBase+EMBasinsStr+'_modes'+str(bestNModes)+'.shelve',
                        dataFileBase+EMBasinsStr+'summary.shelve')
        # further add the best nModes for this dataset to the summary file
        dataBase = shelve.open(dataFileBase+EMBasinsStr+'summary.shelve')
        dataBase['nModes'] = bestNModes
        dataBase['logLVec'] = logLVec
        dataBase['logLTestVec'] = logLTestVec
        dataBase['nModesList'] = nModesList
        dataBase['nModesIdx'] = bestNModesIdx
        dataBase.close()
        print("Finished looking up best nModes = ",bestNModes,
                    " mixture model for file number",interactionFactorIdx)
        sys.stdout.flush()

    ################ Read in the summary data for best nModes and pre-process for later analysis ################

    if plotMeanRates or cfWTAresults or doLDA or doMDSWTA:
        spikeRaster = loadDataSet(dataFileBase, interactionFactorIdx, shuffle=shuffled)
        nNeurons,tSteps = spikeRaster.shape
        if interactionFactorIdx <= 20:
            if HMM:
                fitCutFactor = 2
            else:
                fitCutFactor = 1
        elif interactionFactorIdx == 21:
            fitCutFactor = 1
        trainRaster = spikeRaster[:,:tSteps//(fitCutFactor*2)].T
        testRaster = spikeRaster[:,tSteps//(fitCutFactor*2):tSteps//fitCutFactor].T

    dataBase = shelve.open(dataFileBase+EMBasinsStr+'summary.shelve')

    bestNModes = dataBase['nModes']
    params = dataBase['params']

    bestNModesIdx = dataBase['nModesIdx']
    logLVec = dataBase['logLVec']
    logLTestVec = dataBase['logLTestVec']

    if plotMeanRates:
        # takes a long time to load and plot samples
        samples = dataBase['samples']
        # samples has shape time/sample index x nNeurons
        
        ## Obsolete: samples was in Fortan order,
        ##  now I modified the python wrapper to make samples row-major (C++ and python default)
        ## see its construction in EMBasins::sample() in EMBasins.cpp
        ## if samples is in Fortran order, then flatten and convert back to numpy using Fortran order
        #nNeurons,timebins = samples.shape
        #samples = np.reshape(samples,-1,order='C')
        #samples = np.reshape(samples,(nNeurons,timebins),order='F')
        
    if HMM:
        trans = dataBase['trans']
        wModes = dataBase['stationary_prob'].T
        if assignModesToData or cfWTAresults:
            # P is timebins x modes, probability of modes at each timebin
            P = dataBase['P']
            # labels has the index of the most probable mode at each timebin
            labels = np.argmax(P,axis=1)
            dataBase['modeLabels'] = labels
    else:
        wModes = dataBase['w'].flatten()
        if assignModesToData or cfWTAresults:
            statesTrain = dataBase['state_list']
            statesTest = dataBase['state_list_test']
            statesHistTrain = dataBase['state_hist']
            statesHistTest = dataBase['state_hist_test']
            Ptrain = dataBase['P']
            Ptest = dataBase['P_test']

            trainLabels = np.zeros(tSteps//2,dtype=int)
            testLabels = np.zeros(tSteps-tSteps//2,dtype=int)

            ## round trip raster to spike-times to raster seems fine, as I get same number of patterns
            #nrnSpikeTimes = spikeRasterToSpikeTimes(spikeRaster)
            #spikeRaster = spikeTimesToSpikeRaster(np.array(nrnSpikeTimes),1)
            
            ## np.unique returns patterns in sorted order which will be same as statesTrain order
            ##  since statesTrain is a C++ map which sorts on the pattern string key
            #spikePatterns, patternCounts = np.unique(trainRaster, return_counts=True, axis=0)
            
            ## get unique spikePatterns in order of appearance
            ##  but statesTrain is a C++ map which sorts on the pattern string key
            #_, idx = np.unique(trainRaster,return_index=True,axis=0)
            #spikePatterns = trainRaster[np.sort(idx),:]
             
            ## confirm patterns computed above by me and from the fit are the same
            #for i,pattern in enumerate(spikePatterns):
            #    if (pattern == statesTrain[i,:]).all():
            #        print(i)
            
            statesTrain = statesTrain.astype(int)
            statesTest = statesTest.astype(int)
            
            # numpy array is not hashable, but tuple is, so using tuples as dict keys
            # associate each pattern with its most probable mode 
            patternDictTrain = dict(zip([tuple(state) for state in statesTrain],
                                        [np.argmax(Ptrain[i,:]) for i in range(len(statesTrain))]))
            patternDictTest = dict(zip([tuple(state) for state in statesTest],
                                        [np.argmax(Ptest[i,:]) for i in range(len(statesTest))]))
            # for the pattern in each time bin in the training and test data,
            #  set its mode label using above dictionary of patterns to mode labels
            print("dicts for spike patterns made")
            for i,pattern in enumerate(trainRaster):
                patternstr = tuple(pattern)
                trainLabels[i] = patternDictTrain[patternstr]
            print("trainRaster modes / labels set")
            for i,pattern in enumerate(testRaster):
                patternstr = tuple(pattern)
                testLabels[i] = patternDictTest[patternstr]                
            print("testRaster modes / labels set")

            dataBase['modeLabelsTrain'] = trainLabels
            dataBase['modeLabelsTest'] = testLabels
            print("Saved the assigned modes/labels to spike timebins")
            print(trainRaster.shape,trainLabels)
            print(testRaster.shape,testLabels)

    if doLDA or cfWTAresults:
        if HMM:
            labelsFit = dataBase['modeLabels']
            trainLabels = labels[:len(trainRaster)]
            testLabels = labels[-len(testRaster):]
        else:
            trainLabels = dataBase['modeLabelsTrain']
            testLabels = dataBase['modeLabelsTest']
            labelsFit = np.append(trainLabels,testLabels)
    
    dataBase.close()

    ######################### Analysis across best nMode fits #########################

    # for the TreeBasin spatially-correlated model (set in EMBasins.cpp)
    #  params is a list (size nModes) of dicts, each dict has 'm' and 'J'
    # for the IndependentBasin spatially-uncorrelated model (set in EMBasins.cpp)
    #  params is a list (size nModes) of dicts, each dict has 'm'
    mProb = np.zeros(shape=(bestNModes,len(params[0]['m'])))
    for i,param in enumerate(params):
        mProb[i,:] = param['m'].flatten()

    ax = axesMM[interactionFactorIdx//5,interactionFactorIdx%5]
    #ax = axesMM
    ax.plot(nModesList,logLVec,'k-,')
    ax.scatter(nModesList,logLVec,marker='x',color='k')
    ax.scatter(nModesList,logLTestVec,marker='*',color='b')
    ax.scatter([bestNModes],[logLTestVec[bestNModesIdx]],marker='o',color='r')
    ax.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[interactionFactorIdx])+\
                        ', *nModes='+str(bestNModes))
    print('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[interactionFactorIdx]),
            'logLTestVec=',logLTestVec)
    logLs[interactionFactorIdx] = logLVec[bestNModesIdx]
    logLsTest[interactionFactorIdx] = logLTestVec[bestNModesIdx]

    ax2 = axesMM2[interactionFactorIdx//5,interactionFactorIdx%5]
    #ax2 = axesMM2
    wModes = np.sort(wModes.flatten())[::-1]
    ax2.plot(wModes,'k-,')
    ax2.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[interactionFactorIdx])+\
                        ', *nModes='+str(bestNModes))
    entropies[interactionFactorIdx] = -np.sum(wModes*np.log(wModes))

    if doMDS:
        ax3 = axesMM3[interactionFactorIdx//5,interactionFactorIdx%5]
        if not np.isnan(np.sum(mProb)):
            # calculate mean spike count for each mode and sort by mode weight:
            meanSpikesMode = np.sum(mProb,axis=1)
            sortedIdxs = np.argsort(wModes)[::-1]
            print("mean spike count (sorted) for each fitted mode",
                        meanSpikesMode[sortedIdxs])
            # non-linear dimensionality reduction from nNeurons-dim to 2-dim
            #  using multi-dimensional scaling
            lowDimMDS = MDS(n_components=2)
            # mProb has shape nModes x nNeurons
            lowDimData = lowDimMDS.fit_transform(mProb)
            # lowDimData has shape nModes x n_components
            x,y = lowDimData.T
            # plot in reverse order of importance, so that most important is plotted on top!
            s = ax3.scatter(x[sortedIdxs][::-1],y[sortedIdxs][::-1],
                            c=np.log(wModes.flatten())[sortedIdxs][::-1],
                            s=meanSpikesMode[sortedIdxs][::-1]*10,
                            cmap=cm)
            #ax3.set_xlabel('m (MDS dim1)')
            #ax3.set_ylabel('m (MDS dim2)')
            ax3.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[interactionFactorIdx])+\
                                ', *nModes='+str(bestNModes))
            cbar = figMM3.colorbar(s,ax=ax3)
            cbar.ax.set_ylabel(r"log w",  labelpad=20, rotation=270)

    if plotMeanRates:
        ax4 = axesMM4[interactionFactorIdx//5,interactionFactorIdx%5]
        # mean firing rates of samples from fitted model and from dataset (20ms bin size)
        meanFitRates = np.mean(samples,axis=1) / 20e-3
        meanDataRates = np.mean(spikeRaster,axis=1) / 20e-3
        ax4.plot(meanDataRates,'r-,',lw=3)
        ax4.plot(meanFitRates,'k-,')
        ax4.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[interactionFactorIdx])+\
                            ', *nModes='+str(bestNModes))

    if doLDA:
        # spikeRaster is nNeurons x tSteps
        # LDA.fit_transform() requires samples x features i.e. tSteps x nNeurons
        # number of components doesn't matter in the least for accuracy!
        numComponents = 50
        LDA = DA.LinearDiscriminantAnalysis(n_components=numComponents)
        modeDataLDA = LDA.fit_transform(trainRaster,trainLabels)
        # modeDataLDA is (tSteps,)
        LDAScoreTrain = LDA.score(trainRaster,trainLabels)
        LDAScoreTest = LDA.score(testRaster,testLabels)
        print('Linear discriminant analysis on all modes using ',
                      numComponents,' components, training score is',LDAScoreTrain)
        print('Linear discriminant analysis on all modes using ',
                      numComponents,' components, test score is',LDAScoreTest)
        ax6 = axesMM6[interactionFactorIdx//5,interactionFactorIdx%5]
        # if only 1 mode i.e. only 1 label, then modeDataLDA is an empty array
        if bestNModes>1:
            ax6.scatter(trainLabels,modeDataLDA[:,0])
        #ax6.set_xlabel('mode label')
        ax6.set_ylabel('LDA component 1');
        ax6.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[interactionFactorIdx])+\
                            ', M='+str(bestNModes)+\
                            ' tr,te LDA={:1.2f},{:1.2f}'.format(LDAScoreTrain,LDAScoreTest))
        LDAtrain[interactionFactorIdx] = LDAScoreTrain
        LDAtest[interactionFactorIdx] = LDAScoreTest

    if cfWTAresults or doMDSWTA:
        dataBase = shelve.open(dataFileBase+'_WTA10_modes'+str(bestNModes)+'.shelve')
        # readout_test is numberofmodes x timebins
        readout_test = dataBase['readout_test']
        Converg_avgW = dataBase['Converg_avgW'][0]
        dataBase.close()
        # readout_test is numberofmodes x timebins
        labelsWTA = np.argmax(readout_test,axis=0).T

    if cfWTAresults:
        # labelsWTA and labelsFit are vectors of length timebins
        # could be of slightly different lengths due to fitCutFactor above
        #  or last pattern is missed by WTA spikes reading in algo
        #     -- I corrected this bug in EMBasins / HMM, but not yet in WTA C++ code
        labelsLen = min(len(labelsFit),len(labelsWTA))
        WTAscore = adjusted_rand_score(labelsFit[:labelsLen],labelsWTA[:labelsLen])
        print('WTA score is ',WTAscore)
        WTAscores[interactionFactorIdx] = WTAscore

    if doMDSWTA:
        ax7 = axesMM7[interactionFactorIdx//5,interactionFactorIdx%5]
        mProbWTA = np.zeros((bestNModes,nNeurons))
        wModesWTA = np.zeros(bestNModes)
        for modenum in range(bestNModes):
            binIdxs = np.where(labelsWTA == modenum)[0]
            wModesWTA[modenum] = len(binIdxs)
            mProbWTA[modenum,:] = np.mean(spikeRaster[:,binIdxs],axis=1)
        # calculate mean spike count for each mode and sort by mode weight:
        meanSpikesMode = np.sum(mProbWTA,axis=1)
        sortedIdxs = np.argsort(wModesWTA)[::-1]
        print("mean spike count (sorted) for each WTA mode",
                        meanSpikesMode[sortedIdxs])
        if not np.isnan(np.sum(mProbWTA)):
            # non-linear dimensionality reduction from nNeurons-dim to 2-dim
            #  using multi-dimensional scaling
            lowDimMDS = MDS(n_components=2)
            # mProbWTA has shape nModes x nNeurons
            lowDimData = lowDimMDS.fit_transform(mProbWTA)
            # lowDimData has shape nModes x n_components
            x,y = lowDimData.T
            ## plot in reverse order of importance, so that most important is plotted on top!
            #s = ax7.scatter(x[sortedIdxs][::-1],y[sortedIdxs][::-1],
            #                    c=np.log(wModesWTA.flatten())[sortedIdxs][::-1],
            #                    s=meanSpikesMode[sortedIdxs][::-1]*10,
            #                    cmap=cm)
            ##ax7.set_xlabel('m (MDS dim1)')
            ##ax7.set_ylabel('m (MDS dim2)')
            #ax7.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[interactionFactorIdx])+\
            #                    ', *nModes='+str(bestNModes))
            #cbar = figMM7.colorbar(s,ax=ax7)
            #cbar.ax.set_ylabel(r"log w",  labelpad=20, rotation=270)
        ## Converg_avgW is timebins long - takes a long time to plot, so bin it -- obsolete, now saving only per iteration
        #width=1000
        #binned_Converg_avgW = Converg_avgW[:(len(Converg_avgW) // width) * width].reshape(-1, width).mean(axis=1)
        #ax7.plot(binned_Converg_avgW)
        ax7.plot(Converg_avgW)
        ax7.set_xlabel('timebin')
        ax7.set_ylabel('mean deltaW')

################# plot fitting (best nModes) summary scores versus datasets #################
ax1 = axes[0]
ax1.scatter(interactionFactorList[:20],entropies[:20],marker='x',color='k')
ax1.scatter(interactionFactorList[20],entropies[20],marker='x',color='b')
ax1.scatter(interactionFactorList[21],entropies[21],marker='x',color='r')
ax1.set_xlabel('interaction $\\alpha$')
ax1.set_ylabel('entropy of modes')

ax2 = axes[1]
ax2.scatter(interactionFactorList[:20],logLs[:20],marker='x',color='k')
ax2.scatter(interactionFactorList[20],logLs[20],marker='x',color='b')
ax2.scatter(interactionFactorList[21],logLs[21],marker='x',color='r')
ax2.set_xlabel('interaction $\\alpha$')
ax2.set_ylabel('log likelihood')

ax3 = axes[2]
ax3.scatter(interactionFactorList[:20],logLsTest[:20],marker='x',color='k')
ax3.scatter(interactionFactorList[20],logLsTest[20],marker='x',color='b')
ax3.scatter(interactionFactorList[21],logLsTest[21],marker='x',color='r')
ax3.set_xlabel('interaction $\\alpha$')
ax3.set_ylabel('test log likelihood')

if doLDA:
    ax4 = axes[3]
    ax4.scatter(interactionFactorList[:20],LDAtrain[:20],marker='x',color='k')
    ax4.scatter(interactionFactorList[:20],LDAtest[:20],marker='o',color='k')
    ax4.scatter(interactionFactorList[20],LDAtrain[20],marker='x',color='b')
    ax4.scatter(interactionFactorList[20],LDAtest[20],marker='o',color='b')
    ax4.scatter(interactionFactorList[21],LDAtrain[21],marker='x',color='r')
    ax4.scatter(interactionFactorList[21],LDAtest[21],marker='o',color='r')
    ax4.set_xlabel('interaction $\\alpha$')
    ax4.set_ylabel('LDA score x-train, o-test')

if cfWTAresults:
    ax5 = axes[4]
    ax5.scatter(interactionFactorList[:20],WTAscores[:20],marker='x',color='k')
    ax5.scatter(interactionFactorList[20],WTAscores[20],marker='x',color='b')
    ax5.scatter(interactionFactorList[21],WTAscores[21],marker='x',color='r')
    ax5.set_xlabel('interaction $\\alpha$')
    ax5.set_ylabel('WTAvsfit cluster match')

figList = [figMM,figMM2,figMM5]
if doMDS: figList.append(figMM3)
if plotMeanRates: figList.append(figMM4)
if doLDA: figList.append(figMM6)
if doMDSWTA: figList.append(figMM7)
for fig in figList:
    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0.5, hspace=0.5)

plt.show()
