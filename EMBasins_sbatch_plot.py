import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import shelve, sys, shutil

from sklearn.manifold import MDS
from sklearn import discriminant_analysis as DA

from EMBasins_sbatch import loadDataSet,spikeRasterToSpikeTimes,spikeTimesToSpikeRaster

np.random.seed(100)

HMM = False
shuffled = True
treeSpatial = True

plotMeanRates = False
doLDA = True

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
figMM3, axesMM3 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
if plotMeanRates:
    figMM4, axesMM4 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
if doLDA:
    figMM6, axesMM6 = plt.subplots(nrows=5, ncols=5, figsize=(8,4))
figMM5, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
cm = plt.cm.get_cmap('RdYlBu')

maxModes = 150
nModesList = range(1,maxModes+1,5)

findBestNModes = True       # loop over all the nModes data
                            #  & find best nModes for each dataset
                            # must be done at least once before plotting
                            #  to generate _mixmodsummary.shelve
crossvalfold = 2            # usually 1 or 2 - depends on what you set when fitting

EMBasinsStr = ('_shuffled' if shuffled else '') + \
                ('_HMM' + (str(crossvalfold) if crossvalfold>1 else '') \
                    if HMM else '_EMBasins_full') + \
                ('' if treeSpatial else '_notree')

entropies = []
logLs = []
logLsTest = []
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
                    " mixture model for file number",fileNum)
        sys.stdout.flush()

    dataBase = shelve.open(dataFileBase+EMBasinsStr+'summary.shelve')
    bestNModes = dataBase['nModes']
    params = dataBase['params']
    if plotMeanRates:
        # takes a long time to load and plot samples
        samples = dataBase['samples']
        # samples has shape time/sample index x nNeurons
        
        ## Obsolete: was in Fortan order,
        ##  but I modified the python wrapper to make it row-major (C++ and python default)
        ## see its construction in EMBasins::sample() in EMBasins.cpp
        ## so first flatten, then convert back to numpy using Fortran order
        #nNeurons,timebins = samples.shape
        #samples = np.reshape(samples,-1,order='C')
        #samples = np.reshape(samples,(nNeurons,timebins),order='F')
        
    if HMM:
        trans = dataBase['trans']
        wModes = dataBase['stationary_prob'].T
    else:
        wModes = dataBase['w'].flatten()
        statesTrain = dataBase['state_list']
        statesTest = dataBase['state_list_test']
        statesHistTrain = dataBase['state_hist']
        statesHistTest = dataBase['state_hist_test']
        Ptrain = dataBase['P']
        Ptest = dataBase['P_test']
    bestNModesIdx = dataBase['nModesIdx']
    logLVec = dataBase['logLVec']
    logLTestVec = dataBase['logLTestVec']
    dataBase.close()

    # for the TreeBasin spatially-correlated model (set in EMBasins.cpp)
    #  params is a list (size nModes) of dicts, each dict has 'm' and 'J'
    # for the IndependentBasin spatially-uncorrelated model (set in EMBasins.cpp)
    #  params is a list (size nModes) of dicts, each dict has 'm'
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
            'logLTestVec=',logLTestVec)
    logLs.append(logLVec[bestNModesIdx])
    logLsTest.append(logLTestVec[bestNModesIdx])

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
        # plot in reverse order of importance, so that most important is plotted on top!
        s = ax3.scatter(x[::-1],y[::-1],c=np.log(wModes.flatten())[::-1],s=meanSpikesMode[::-1]*10,cmap=cm)
        #ax3.set_xlabel('m (MDS dim1)')
        #ax3.set_ylabel('m (MDS dim2)')
        ax3.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[fileNum])+\
                            ', *nModes='+str(bestNModes))
        cbar = figMM3.colorbar(s,ax=ax3)
        cbar.ax.set_ylabel(r"log w",  labelpad=20, rotation=270)

    if plotMeanRates or doLDA:
        spikeRaster = loadDataSet(dataFileBase, fileNum, shuffle=shuffled)

    if plotMeanRates:
        ax4 = axesMM4[fileNum//5,fileNum%5]
        # mean firing rates of samples from fitted model and from dataset (20ms bin size)
        meanFitRates = np.mean(samples,axis=1) / 20e-3
        meanDataRates = np.mean(spikeRaster,axis=1) / 20e-3
        ax4.plot(meanDataRates,'r-,',lw=3)
        ax4.plot(meanFitRates,'k-,')
        ax4.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[fileNum])+\
                            ', *nModes='+str(bestNModes))

    if doLDA and not HMM:   # probably not yet returning the needed data via pyHMM
        nNeurons,tSteps = spikeRaster.shape
        trainRaster = spikeRaster[:,:tSteps//2].T
        testRaster = spikeRaster[:,tSteps//2:].T
        # Qmodes is nModes x tSteps, modeDataLabels is (tSteps,)
        trainLabels = np.ones(tSteps//2,dtype=int)*-1
        testLabels = np.ones(tSteps//2,dtype=int)*-1

        ## round trip raster to spike-times to raster seems fine, as I get same number of patterns
        nrnSpikeTimes = spikeRasterToSpikeTimes(spikeRaster)
        spikeRaster = spikeTimesToSpikeRaster(np.array(nrnSpikeTimes),1)

        # I'm not setting the same seed in the loadDataSet function, so not the same train and test split?
        # Thus, I get different patterns in trainRaster compared to statesTrain?
        
        # unique returns patterns in sorted order which will be same as statesTrain order
        #  since statesTrain is a C++ map which sorts on the pattern string key
        spikePatterns, patternCounts = np.unique(trainRaster, return_counts=True, axis=0)
        
        ## unique spikePatterns in order of appearance
        ##  but statesTrain is a C++ map which sorts on the pattern string key
        #_, idx = np.unique(trainRaster,return_index=True,axis=0)
        #spikePatterns = trainRaster[np.sort(idx),:]
         
        ## patterns are the same
        #for i,pattern in enumerate(spikePatterns):
        #    if (pattern == statesTrain[i,:]).all():
        #        print(i)

        #spikePatternsFit = np.unique(np.append(statesTrain,statesTest,axis=0), return_counts=False, axis=0)
        #print(spikePatterns.shape,spikePatternsFit.shape)
        #print(spikePatterns.shape,statesTrain.shape)
        #print(np.sum(patternCounts),np.sum(statesHistTrain),tSteps//2)
        #spikePatterns, patternCounts = np.unique(testRaster, return_counts=True, axis=0)
        #print(spikePatterns.shape,statesTest.shape)
        #print(np.sum(patternCounts),np.sum(statesHistTest),tSteps-tSteps//2)
        ## number of patterns in spikePatterns and those from the fitting should equal number of time bins
        ## 2 time bins missing in fitted states (199,998) out of 200,000,
        ##  but could be initial/end 0-s not getting transmitted in spike times.
        ##  though raster -> spike-times -> raster round trip above still gave back 200,000 steps
        #print(np.sum(patternCounts),np.sum(statesHistTrain)+np.sum(statesHistTest),tSteps)
        
        statesTrain = statesTrain.astype(int)
        statesTest = statesTest.astype(int)
        
        # tuple is hashable but numpy array is not, so using tuples as dict keys
        patternDictTrain = dict(zip([tuple(state) for state in statesTrain],
                                    [np.argmax(Ptrain[i,:]) for i in range(len(statesTrain))]))
        patternDictTest = dict(zip([tuple(state) for state in statesTest],
                                    [np.argmax(Ptest[i,:]) for i in range(len(statesTest))]))
        # for the pattern in each time bin in the training and test data,
        #  set its most probable mode using the above dictionary of patterns to modes
        print("dicts made")
        for i,pattern in enumerate(trainRaster):
            patternstr = tuple(pattern)
            trainLabels[i] = patternDictTrain[patternstr]
        print("trainRaster done")
        for i,pattern in enumerate(testRaster):
            patternstr = tuple(pattern)
            testLabels[i] = patternDictTest[patternstr]                
        print("testRaster done")

        # spikeRaster is nNeurons x tSteps
        # LDA.fit_transform() requires samples x features i.e. tSteps x nNeurons
        # number of components doesn't matter in the least for accuracy!
        numComponents = 1
        LDA = DA.LinearDiscriminantAnalysis(n_components=numComponents)
        modeDataLDA = LDA.fit_transform(trainRaster,trainLabels)
        # modeDataLDA is (tSteps,)
        LDAScoreTrain = LDA.score(trainRaster,trainLabels)
        LDAScoreTest = LDA.score(testRaster,testLabels)
        print('Linear discriminant analysis on all modes using ',
                      numComponents,' components, training score is',LDAScoreTrain)
        print('Linear discriminant analysis on all modes using ',
                      numComponents,' components, test score is',LDAScoreTest)
        ax6 = axesMM6[fileNum//5,fileNum%5]
        # if only 1 mode i.e. only 1 label, then modeDataLDA is an empty array
        if bestNModes>1:
            ax6.scatter(trainLabels,modeDataLDA[:,0])
        ax6.set_xlabel('mode label')
        ax6.set_ylabel('LDA component 1');
        ax6.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[fileNum])+\
                            ', *nModes='+str(bestNModes)+\
                            ' train,test LDA = {:1.1f},{:1.1f}'.format(LDAScoreTrain,LDAScoreTest))

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

figList = [figMM,figMM2,figMM3,figMM5]
if plotMeanRates: figList.append(figMM4)
if doLDA: figList.append(figMM6)
for fig in figList:
    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0.5, hspace=0.5)

plt.show()
