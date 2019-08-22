import numpy as np
import scipy.io
import shelve, sys, os.path

from MixtureModel import loadDataSet
import EMBasins
EMBasins.pyInit()

np.random.seed(100)

HMM = True                          # HMM or EMBasins i.e. without temporal correlations 
                                    # IndependentBasin or TreeBasin (for spatial correlations)
                                    #  is to be selected in EMBasins.cpp and `make`
crossvalfold = 2                    # currently only for HMM, k-fold validation? 1 for train only

interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.])

maxModes = 150
nModesList = range(1,maxModes+1,5)  # steps of 5, no need to go one by one

# for sbatch array jobs, $SLURM_ARRAY_TASK_ID is passed as first command-line argument
#  so give the sbatch array job with indexes corresponding to taskId
#   that you want to decode as below into interactionFactorIdx and nModesIdx
#  sbatch --array=0-629 submit_mixmod.sbatch   # 30 nModes * 21 datasets = 630 tasks
print(sys.argv)
if len(sys.argv) > 1:
    taskId = int(sys.argv[1])
    interactionFactorIdx = taskId // len(nModesList)
    nModesIdx = taskId % len(nModesList)
    interactionFactor = interactionFactorList[interactionFactorIdx]
    nModes = nModesList[nModesIdx]
else:
    interactionFactorIdx = 20       # experimental data
    nModes = 70                     # best nModes reported for exp data in Prentice et al 2016
#binsize = 200                       # number of samples per bin,
                                    #  @ 10KHz sample rate and a 20ms bin, binsize=200
                                    # 200 is needed if spiketimes were given in units of sampling indices @ 10kHz
binsize = 1                         # here, spikes are given pre-binned into a spikeRaster, just take binsize=1

# whether to fit mixture models and save data
# this takes the longest time, so beware
fitMixMod = True
# whether to continue more EM steps from earlier saved fits
loadEarlierFits = False

dataFileBaseName = 'Learnability_data/synthset_samps'

# to fit MixMod for specific dataset and nModes
# first 20 are generated, 21st is exp dataset
if interactionFactorIdx != 20:
    dataFileBase = dataFileBaseName + '_' + str(interactionFactorIdx+1)
else:
    dataFileBase = 'Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100'

def spikeRasterToSpikeTimes(spikeRaster):
    # from a spikeRaster create a neurons list of lists of spike times
    nNeurons,tSteps = spikeRaster.shape
    nrnSpikeTimes = []
    # multiply by binsize, so that spike times are given in units of sampling indices
    bins = np.arange(tSteps,dtype=int)*binsize
    for nrnnum in range(nNeurons):
        # am passing a list of lists, convert numpy.ndarray to list,
        #  numpy.ndarray is just used to enable multi-indexing
        nrnSpikeTimes.append( list(bins[spikeRaster[nrnnum,:] != 0]) )
    return nrnSpikeTimes

if HMM:
    def saveFit(dataFileBase,nModes,params,trans,emiss_prob,alpha,pred_prob,hist,samples,train_logli,test_logli):
        dataBase = shelve.open(dataFileBase+'_HMM'+(str(crossvalfold) if crossvalfold>1 else '')\
                                            +'_modes'+str(nModes)+'.shelve')
        dataBase['params'] = params
        dataBase['trans'] = trans
        dataBase['emiss_prob'] = emiss_prob
        dataBase['alpha'] = alpha
        dataBase['pred_prob'] = pred_prob
        dataBase['hist'] = hist
        dataBase['samples'] = samples
        dataBase['train_logli'] = train_logli
        dataBase['test_logli'] = test_logli
        dataBase.close()
else:
    def saveFit(dataFileBase,nModes,params,w,P,prob,logli,P_test):
        dataBase = shelve.open(dataFileBase+'_EMBasins_modes'+str(nModes)+'.shelve')
        dataBase['params'] = params
        dataBase['w'] = w
        dataBase['P'] = P
        dataBase['prob'] = prob
        dataBase['logli'] = logli
        dataBase['P_test'] = P_test
        dataBase.close()

if fitMixMod:
    ## currently not implemented passing earlier params to EMBasins.cpp
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

    spikeRaster = loadDataSet(dataFileBase, interactionFactorIdx, not HMM)
    ## find unique spike patterns and their counts
    #spikePatterns, patternCounts = np.unique(spikeRaster, return_counts=True, axis=1)

    nNeurons,tSteps = spikeRaster.shape
    print("Mixture model fitting for file number",interactionFactorIdx)
    sys.stdout.flush()
    niter = 100
    
    if HMM:
        spikeRaster = spikeRaster[:,:tSteps//2]
        tSteps = tSteps//2
        train_logli = np.zeros(shape=(crossvalfold,niter))
        test_logli = np.zeros(shape=(crossvalfold,niter))
        if crossvalfold > 1:
            nrnspiketimes = spikeRasterToSpikeTimes(spikeRaster)
            # translated from getHMMParams.m 
            # if I understand correctly:
            #  to avoid losing temporal correlations,
            #  we specify contiguous chunks of training data
            #  by specifying upper _hi and lower _lo boundaries, as below
            #  the rest becomes contiguous chunks of test data
            #  thus, HMM::logli(true) in EMBasins.cpp gives training logli
            #   and HMM::logli(false) gives test logli
            bins = np.arange(tSteps)*binsize
            shuffled_idxs = np.random.permutation(np.arange(tSteps,dtype=np.int32))
            ntest = int(tSteps/crossvalfold)
            for k in range(crossvalfold):
                test_idxs = shuffled_idxs[k*ntest:(k+1)*ntest]
                train_idxs = np.zeros(tSteps,dtype=np.int32)
                train_idxs[test_idxs] = 1
                
                # contiguous 1s form a training (or maybe test) chunk
                flips = np.diff(np.append([0],train_idxs))
                unobserved_lo = bins[ flips == 1 ]
                unobserved_hi = bins[ flips == -1 ]
                # just in case, a last -1 is not there to close the last chunk
                if (len(unobserved_hi) < len(unobserved_lo)):
                    unobserved_hi = np.append(unobserved_hi,[tSteps])

                params,trans,emiss_prob,alpha,pred_prob,hist,samples,train_logli_this,test_logli_this = \
                    EMBasins.pyHMM(nrnspiketimes, unobserved_lo, unobserved_hi,
                                        float(binsize), nModes, niter)
                train_logli[k,:] = train_logli_this.flatten()
                test_logli[k,:] = test_logli_this.flatten()
        # no cross-validation, train on full data
        else:
            nrnspiketimes = spikeRasterToSpikeTimes(spikeRaster)
            params,trans,emiss_prob,alpha,pred_prob,hist,samples,train_logli_this,test_logli_this = \
                EMBasins.pyHMM(nrnspiketimes, np.ndarray([]), np.ndarray([]),
                                    float(binsize), nModes, niter)
            train_logli[0,:] = train_logli_this.flatten()
            test_logli[0,:] = test_logli_this.flatten()
        # Save the fitted model
        saveFit(dataFileBase,nModes,params,trans,emiss_prob,alpha,pred_prob,hist,samples,train_logli,test_logli)

    # temporally independent EMBasins
    else:
        nrnspiketimes = spikeRasterToSpikeTimes(spikeRaster[:,:tSteps//4])
        nrnspiketimes_test = spikeRasterToSpikeTimes(spikeRaster[:,tSteps//4:tSteps//2])
        # note: currently I'm returning logli_test in P_test (see my current mods in EMBasins.cpp)
        # train on some, test on some
        params,w,samples,state_hist,P,prob,logli,P_test = \
                EMBasins.pyEMBasins(nrnspiketimes, nrnspiketimes_test, float(binsize), nModes, niter)
        # Save the fitted model
        saveFit(dataFileBase,nModes,params,w,P,prob,logli,P_test)
        train_logli = logli
        test_logli = P_test

    print("Mixture model fitted for file number",interactionFactorIdx)
    sys.stdout.flush()    

    print("mixture model saved for file number ",interactionFactorIdx,
                            ', nModes ',nModes,' out of ',maxModes,
                            '.\n logL=',train_logli,'.\n logLTest=',test_logli)
    sys.stdout.flush()
