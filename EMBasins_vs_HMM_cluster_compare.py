import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import shelve, sys, shutil

from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score

from EMBasins_sbatch import loadDataSet

np.random.seed(100)

HMMShuffled = True
EMBasinsShuffled = True
treeSpatial = True

crossvalfold = 2            # usually 1 or 2 - depends on what you set when fitting

dataFileBaseName = 'Learnability_data/synthset_samps'
interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.,1.])

figMM, axesMM = plt.subplots(nrows=5, ncols=5, figsize=(8,4))

maxModes = 150
nModesList = range(1,maxModes+1,5)

HMMStr = ('_shuffled' if HMMShuffled else '') + \
                '_HMM' + (str(crossvalfold) if crossvalfold>1 else '') + \
                ('' if treeSpatial else '_notree')

EMBasinsStr = ('_shuffled' if EMBasinsShuffled else '') + \
                '_EMBasins_full' + \
                ('' if treeSpatial else '_notree')

interactionsLen = len(interactionFactorList)

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

    ARIscores = np.zeros(len(nModesList))
    for idx,nModes in enumerate(nModesList):
        ### load the spike rasters from the dataset
        # only the EMBasins shuffling matters as I receive prob of modes Ptrain and Ptest for spike patterns
        # so I need to get the right train / test split to ensure that 
        #  I find a pattern in Ptrain/Ptest for every pattern in trainRaster/testRaster
        spikeRaster = loadDataSet(dataFileBase, interactionFactorIdx, shuffle=EMBasinsShuffled)
        nNeurons,tSteps = spikeRaster.shape
        if interactionFactorIdx <= 20:
            fitCutFactor = 2
        elif interactionFactorIdx == 21:
            fitCutFactor = 1
        trainRaster = spikeRaster[:,:tSteps//2].T
        testRaster = spikeRaster[:,tSteps//2:tSteps].T

        ### load the fits for EMBasins
        print(dataFileBase+EMBasinsStr+'_modes'+str(nModes)+'.shelve')
        dataBase = shelve.open(dataFileBase+EMBasinsStr+'_modes'+str(nModes)+'.shelve')
        statesTrain = dataBase['state_list']
        statesTest = dataBase['state_list_test']
        Ptrain = dataBase['P']
        Ptest = dataBase['P_test']

        trainLabels = np.zeros(tSteps//2,dtype=int)
        testLabels = np.zeros(tSteps-tSteps//2,dtype=int)
        
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
        EMBasinsLabelsFit = np.append(trainLabels,testLabels)
        dataBase.close()

        ### load the fits for HMM
        print(dataFileBase+HMMStr+'_modes'+str(nModes)+'.shelve')
        dataBase = shelve.open(dataFileBase+HMMStr+'_modes'+str(nModes)+'.shelve')
        # P is timebins x modes, probability of modes at each timebin
        P = dataBase['P']
        # labels has the index of the most probable mode at each timebin
        HMMLabelsFit = np.argmax(P,axis=1)
        dataBase.close()

        # HMMLabelsFit and EMBasinsLabelsFit are vectors of length timebins
        # could be of slightly different lengths due to fitCutFactor above
        labelsLen = min(len(HMMLabelsFit),len(EMBasinsLabelsFit))
        ARIscore = adjusted_rand_score(HMMLabelsFit[:labelsLen],EMBasinsLabelsFit[:labelsLen])
        print('ARI score for nModes=',nModes,' is ',ARIscore)
        ARIscores[idx] = ARIscore

    ax = axesMM[interactionFactorIdx//5,interactionFactorIdx%5]
    ax.scatter(nModesList,ARIscores,marker='x',color='k')
    ax.set_title('$\\alpha=$'+"{:1.1f}".format(interactionFactorList[interactionFactorIdx]))
    ax.set_xlabel('nModes')
    ax.set_ylabel('HMM vs EMBasins cluster match')

for fig in [figMM]:
    fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0.5, hspace=0.5)

plt.show()
