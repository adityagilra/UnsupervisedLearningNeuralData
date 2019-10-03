import numpy as np
import scipy.io
import shelve, sys, os.path

from EMBasins_sbatch import interactionFactorIdx,interactionFactor,nModes,maxModes,\
                            dataFileBase,loadDataSet,spikeRasterToSpikeTimes

import WTA_clustering.VIWTA_SNN as VIWTA_SNN

np.random.seed(100)

def saveFit(dataFileBase,nModes,W_star,b_star,Converg_avgW,readout_train,readout_test):
    dataBase = shelve.open(dataFileBase +'_WTA'+'_modes'+str(nModes)+'.shelve')
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

    spikeRaster = loadDataSet(dataFileBase, interactionFactorIdx, shuffle=True)
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
    
    eta_b    = 0.001        # Learning rate hyperparameter 
    eta_W    = 0.0004       # Learning rate hyperparameter
    mixing_weights = (1./nModes)*np.ones(nModes)

    W_star, b_star, Converg_avgW, readout_train, readout_test = \
      VIWTA_SNN.pyWTAcluster(nrnspiketimes, float(binsize), nModes, eta_b, eta_W, mixing_weights)

    print("Mixture model fitted for file number",interactionFactorIdx)
    sys.stdout.flush()

    saveFit(dataFileBase,nModes,W_star,b_star,Converg_avgW,readout_train,readout_test)

    print("mixture model saved for file number ",interactionFactorIdx,
                            ', nModes ',nModes,' out of ',maxModes)
    sys.stdout.flush()
