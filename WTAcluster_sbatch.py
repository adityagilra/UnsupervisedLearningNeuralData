import numpy as np
import scipy.io
import shelve, sys, os.path

from EMBasins_sbatch import interactionFactorIdx,interactionFactor,nModes,maxModes,\
                            crossvalfold,treeSpatial,dataFileBase,loadDataSet,\
                            spikeRasterToSpikeTimes

# for this clone and compile the repo: https://github.com/adityagilra/VIWTA-SNN
# name the local repo folder VIWTA_SNN, not the default VIWTA-SNN, since hyphen can't be used in import
# be sure the parent path of the repo folder is in your PYTHONPATH
import VIWTA_SNN.VIWTA_SNN as VIWTA_SNN

Shuffle = False     # whether to shuffle time bins
np.random.seed(100)

def saveFit(dataFileBase,HMMStr,nModes,trainiter,W_star,b_star,Converg_avgW,readout_train,readout_test):
    dataBase = shelve.open(dataFileBase+HMMStr+'_WTA'+str(trainiter)+'_modes'+str(nModes)+'.shelve')
    dataBase['W_star'] = W_star
    dataBase['b_star'] = b_star
    dataBase['Converg_avgW'] = Converg_avgW
    dataBase['readout_train'] = readout_train
    dataBase['readout_test'] = readout_test
    dataBase.close()

if __name__ == "__main__":
    binsize = 200           # binsize of 20ms with sampling rate of 10kHz
    VIWTA_SNN.pyInit()

    ## currently not implemented passing earlier params to VIWTA_SNN.cpp
    #if loadEarlierFits:
    #    if os.path.isfile(dataFileBase+'_mixmod_modes'+str(nModes)+'.shelve'):
    #        print("Continuing from fitted model for file number ",interactionFactorIdx)
    #        paramsFileBase = dataFileBase
    #    else:
    #        print("Earlier fit for file number ",interactionFactorIdx,
    #                " not found, exiting...")
    #        sys.exit(1)
    #else:
    #    paramsFileBase = None

    spikeRaster = loadDataSet(dataFileBase, interactionFactorIdx, shuffle=Shuffle)
    ## find unique spike patterns and their counts
    #spikePatterns, patternCounts = np.unique(spikeRaster, return_counts=True, axis=1)
    fitCutFactor = 1
        
    nNeurons,tSteps = spikeRaster.shape
    ## manually split the dataset into train and test for pyWTAfit (has no in-built train/test)
    ## returns a list of lists as desired in C++ by boost
    #nrnspiketimes = spikeRasterToSpikeTimes(spikeRaster[:,:tSteps//(fitCutFactor*2)])
    #nrnspiketimes_test = spikeRasterToSpikeTimes(spikeRaster[:,tSteps//(fitCutFactor*2):tSteps//fitCutFactor])
    nrnspiketimes = spikeRasterToSpikeTimes(spikeRaster,binsize)

    print("WTA model fitting for file number ",interactionFactorIdx," modes ",nModes)
    sys.stdout.flush()
    
    eta_b    = 0.1          # Learning rate hyperparameter
    eta_W    = 0.25         # Learning rate hyperparameter
    trainiter = 1           # number of times to iterate over all the training data

    # choose equal mode weights or pre-fitted mode weights (one of the two below)
    ## mixing_weights are not initial weights, but are used in a complicated way to set biases
    #mixing_weights = (1./nModes)*np.ones(nModes)
    #HMMStr = ''

    # mixing_weights are actually wModes pre-fitted by EMBasins (Prentice et al 2016)
    # load wModes from the fit for HMM
    HMMStr = ('_shuffled' if Shuffle else '') + \
                '_HMM' + (str(crossvalfold) if crossvalfold>1 else '_EMBasins_full') + \
                ('' if treeSpatial else '_notree')
    print('loading pre-fitted wModes from ',dataFileBase+HMMStr+'_modes'+str(nModes)+'.shelve')
    dataBase = shelve.open(dataFileBase+HMMStr+'_modes'+str(nModes)+'.shelve')
    wModes = dataBase['stationary_prob'].flatten()
    dataBase.close()
    mixing_weights = wModes

    W_star, b_star, Converg_avgW, readout_train, readout_test = \
      VIWTA_SNN.pyWTAcluster(nrnspiketimes, float(binsize), nModes, eta_b, eta_W, trainiter, mixing_weights)

    print("Mixture model fitted for file number",interactionFactorIdx)
    sys.stdout.flush()

    saveFit(dataFileBase,HMMStr,nModes,trainiter,W_star,b_star,Converg_avgW,readout_train,readout_test)

    print("mixture model saved for file number ",interactionFactorIdx,
                            ', nModes ',nModes,' out of ',maxModes)
    sys.stdout.flush()
