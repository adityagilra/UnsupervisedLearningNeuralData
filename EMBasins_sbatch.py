import numpy as np
import scipy.io
import shelve, sys, os.path

np.random.seed(100)

HMM = False                         # HMM or EMBasins i.e. with or without temporal correlations 
                                    # for with or without spatial correlations,
                                    #  select IndependentBasin or TreeBasin
                                    #  in EMBasins.cpp and `make`
shuffle = True                      # shuffle time bins in dataset if EMBasins (t-indep), not if HMM (t-dep)
crossvalfold = 2                    # currently only for HMM, k-fold validation? 1 for train only
treeSpatial = True                  # tree-based spatial correlations or no correlations

interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.])
interactionFactorList = np.append(interactionFactorList,[1.])

maxModes = 150
nModesList = range(1,maxModes+1,5)  # steps of 5, no need to go one by one

# for sbatch array jobs, $SLURM_ARRAY_TASK_ID is passed as first command-line argument
#  so give the sbatch array job with indexes corresponding to taskId
#   that you want to decode as below into interactionFactorIdx and nModesIdx
#  sbatch --array=0-659 submit_EMBasins.sbatch   # 30 nModes * 22 datasets = 660 tasks
print(sys.argv)
if len(sys.argv) > 1:
    taskId = int(sys.argv[1])
    nSeed = taskId // (len(interactionFactorList)*len(nModesList))
    remainTaskId = taskId % (len(interactionFactorList)*len(nModesList))
    interactionFactorIdx = remainTaskId // len(nModesList)
    nModesIdx = remainTaskId % len(nModesList)
    interactionFactor = interactionFactorList[interactionFactorIdx]
    nModes = nModesList[nModesIdx]
else:
    interactionFactorIdx = 20       # experimental data
    interactionFactor = interactionFactorList[interactionFactorIdx]
    nSeed = 0
    nModes = 70                     # best nModes reported for exp data in Prentice et al 2016
    print('You\'ve either imported EMBasins_sbatch.py or called it without CLI args,'
            'so setting default interactionFactorIdx and nModes ...')
print("nSeed=",nSeed)
print("interactionFactor=",interactionFactor)
print("nModes=",nModes)
#binsize = 200                       # number of samples per bin,
                                    #  @ 10KHz sample rate and a 20ms bin, binsize=200
                                    # 200 is needed if spiketimes were given in units of sampling indices @ 10kHz
binsize = 1                         # here, spikes are given pre-binned into a spikeRaster, just take binsize=1

# whether to fit mixture models and save data
# this takes the longest time, so beware
fitMixMod = True
# whether to continue more EM steps from earlier saved fits
loadEarlierFits = False

# to fit MixMod for specific dataset and nModes
# first 20 are generated, 21st is exp dataset
if interactionFactorIdx < 20:
    #dataFileBaseName = 'Learnability_data/synthset_samps'
    #dataFileBase = dataFileBaseName + '_' + str(interactionFactorIdx+1)
    dataFileBase = 'Learnability_data/generated_data_1_'\
                                        +str(interactionFactorIdx+1)+'_'\
                                        +str(nSeed)
elif interactionFactorIdx == 20:
    dataFileBase = 'Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100'
elif interactionFactorIdx == 21:
    dataFileBase = 'Prenticeetal2016_data/unique_natural_movie/data'

def spikeRasterToSpikeTimes(spikeRaster,binsize=binsize):
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

def spikeTimesToSpikeRaster(nrnSpikeTimes,binsteps):
    maxBins = 1
    nNeurons = len(nrnSpikeTimes)
    # loop since not a numpy array, each neuron has different number of spike times
    for nrnnum in range(nNeurons):
        maxBins = max( ( maxBins, int(np.amax(nrnSpikeTimes[nrnnum])/binsteps) + 1 ) )
    spikeRaster = np.zeros((nNeurons,maxBins))
    for nrnnum in range(nNeurons):
        spikeRaster[ nrnnum, (np.array(nrnSpikeTimes[nrnnum])/binsteps).astype(int) ] = 1.
    return spikeRaster

def loadDataSet(dataFileBase,interactionFactorIdx,shuffle=True,seed=100):
    # load the model generated dataset
    if interactionFactorIdx < 20:
        #retinaData = scipy.io.loadmat(dataFileBase+'.mat')
        #spikeRaster = retinaData['synthset']['smp'][0,0]
        #referenceRates = retinaData['synthset']['mv0'][0,0][0]
        #sampleRates = retinaData['synthset']['mv'][0,0][0]
        database = shelve.open(dataFileBase+'.shelve','r')
        spikeRaster = database['sample']        # neurons x timebins
        database.close()
    elif interactionFactorIdx == 20:
        retinaData = scipy.io.loadmat(dataFileBase+'.mat')
        spikeRaster = retinaData['bint']
        spikeRaster = np.reshape(np.moveaxis(spikeRaster,0,-1),(160,-1))
    elif interactionFactorIdx == 21:
        retinaData = scipy.io.loadmat(dataFileBase+'.mat')
        # see: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
        #  "For historic reasons, in Matlab everything is at least a 2D array, even scalars.
        #   So scipy.io.loadmat mimics Matlab behavior by default."
        # retinaData['data'][0,0] has .__class__ numpy.void, and has 'keys' (error on .keys() !):
        # description, experiment_date, spike_times, stimulus, hmm_fit
        #  the latter three are matlab structs, so similiar [0,0] indexing.
        #  load into matlab to figure out the 'keys'!
        nrnSpikeTimes = retinaData['data'][0,0]['spike_times'][0,0]['all_spike_times'][0]

        ## obsolete: when I was not converting to spikeRaster (needed for shuffling),
        ##  I int-divided by 200 and converted to list of lists
        ## spikeTimes are in bins of 1/10,000Hz i.e. 0.1 ms
        ## we bin it into 20 ms bins, so integer divide spikeTimes by 20/0.1 = 200 to get bin indices
        #nNeurons = len(nrnSpikeTimes)
        #nrnSpikeTimes = nrnSpikeTimes // 200
        #nrnspiketimes = []
        #tSteps = 0
        #for nrnnum in range(nNeurons):
        #    # am passing a list of lists via boost, convert numpy.ndarray to list
        #    spikeTimes = nrnSpikeTimes[nrnnum][0]
        #    tSteps = np.max( (tSteps,spikeTimes[-1]) )
        #    # somehow np.int32 gave error in converting to double in C++ via boost
        #    nrnspiketimes.append( list(spikeTimes.astype(np.float)) )

        # to shuffle time bins for this dataset, I need to convert spike times to spike raster
        spikeRaster = spikeTimesToSpikeRaster(nrnSpikeTimes,200)    # bin size = 20ms, i.e. 200 steps @ 10kHz sampling
    nNeurons,tSteps = spikeRaster.shape
    if shuffle:
        # randomly permute the full dataset
        # careful if fitting a temporal model and/or retina has adaptation
        # set seed to ensure repeatability of train/test split later
        np.random.seed(seed)
        shuffled_idxs = np.random.permutation(np.arange(tSteps,dtype=np.int32))
        spikeRaster = spikeRaster[:,shuffled_idxs]        
    return spikeRaster


if __name__ == "__main__":
    import HMM_tree_fitting.EMBasins as EMBasins
    EMBasins.pyInit()

    if HMM:
        def saveFit(dataFileBase,nModes,params,trans,P,emiss_prob,alpha,pred_prob,hist,samples,state_list,stationary_prob,train_logli,test_logli):
            dataBase = shelve.open(dataFileBase + ('_shuffled' if shuffle else '') \
                                                + '_HMM'+(str(crossvalfold) if crossvalfold>1 else '') \
                                                + ('' if treeSpatial else '_notree') \
                                                +'_modes'+str(nModes)+'.shelve')
            dataBase['params'] = params
            dataBase['trans'] = trans
            dataBase['P'] = P
            dataBase['emiss_prob'] = emiss_prob
            dataBase['alpha'] = alpha
            dataBase['pred_prob'] = pred_prob
            dataBase['hist'] = hist
            dataBase['samples'] = samples
            dataBase['state_list'] = state_list
            dataBase['stationary_prob'] = stationary_prob
            dataBase['train_logli'] = train_logli
            dataBase['test_logli'] = test_logli
            dataBase.close()
    else:
        def saveFit(dataFileBase,nModes,params,w,samples,state_list,state_hist,state_list_test,state_hist_test,P,P_test,prob,prob_test,train_logli,test_logli):
            dataBase = shelve.open(dataFileBase + ('_shuffled' if shuffle else '') \
                                                + '_EMBasins_full' \
                                                + ('' if treeSpatial else '_notree') \
                                                + '_modes'+str(nModes)+'.shelve')
                                                # _full since fitCutFactor = 1, not 2 below
            dataBase['params'] = params
            dataBase['w'] = w
            dataBase['samples'] = samples
            dataBase['state_list'] = state_list
            dataBase['state_hist'] = state_hist
            dataBase['state_list_test'] = state_list_test
            dataBase['state_hist_test'] = state_hist_test
            dataBase['P'] = P
            dataBase['P_test'] = P_test
            dataBase['prob'] = prob
            dataBase['prob_test'] = prob_test
            dataBase['train_logli'] = train_logli
            dataBase['test_logli'] = test_logli
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

        spikeRaster = loadDataSet(dataFileBase, interactionFactorIdx, shuffle=shuffle)
        ## find unique spike patterns and their counts
        #spikePatterns, patternCounts = np.unique(spikeRaster, return_counts=True, axis=1)
        if interactionFactorIdx <= 20:
            if HMM:
                fitCutFactor = 2
            else:
                fitCutFactor = 1
        elif interactionFactorIdx == 21:
            fitCutFactor = 1
            
        nNeurons,tSteps = spikeRaster.shape
        if HMM:
            # HMM will split the dataset in train and test sets based on crossvalfold
            spikeRaster = spikeRaster[:,:tSteps//fitCutFactor]
            tSteps = tSteps//fitCutFactor
            # returns a list of lists as desired in C++ by boost
            nrnspiketimes = spikeRasterToSpikeTimes(spikeRaster)
        else:
            # manually split the dataset into train and test for EMBasins (has no in-built cross-validation)
            # returns a list of lists as desired in C++ by boost
            nrnspiketimes = spikeRasterToSpikeTimes(spikeRaster[:,:tSteps//(fitCutFactor*2)])
            nrnspiketimes_test = spikeRasterToSpikeTimes(spikeRaster[:,tSteps//(fitCutFactor*2):tSteps//fitCutFactor])


        print("Mixture model fitting for interaction factor = ",
                interactionFactor,', nModes = ',nModes,', nSeed = ',nSeed)
        sys.stdout.flush()
        niter = 100
        
        if HMM:
            train_logli = np.zeros(shape=(crossvalfold,niter))
            test_logli = np.zeros(shape=(crossvalfold,niter))
            if crossvalfold > 1:
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
                    
                    # contiguous 1s form a test chunk, i.e. are "unobserved"
                    #  see state_list assignment in the HMM constructor in EMBasins.cpp
                    flips = np.diff(np.append([0],train_idxs))
                    unobserved_lo = bins[ flips == 1 ]
                    unobserved_hi = bins[ flips == -1 ]
                    # just in case, a last -1 is not there to close the last chunk
                    if (len(unobserved_hi) < len(unobserved_lo)):
                        unobserved_hi = np.append(unobserved_hi,[tSteps])

                    params,trans,P,emiss_prob,alpha,pred_prob,hist,samples,state_list,stationary_prob,train_logli_this,test_logli_this = \
                        EMBasins.pyHMM(nrnspiketimes, unobserved_lo, unobserved_hi,
                                            float(binsize), nModes, niter)
                    train_logli[k,:] = train_logli_this.flatten()
                    test_logli[k,:] = test_logli_this.flatten()
            else: # no cross-validation specified, train on full data
                params,trans,P,emiss_prob,alpha,pred_prob,hist,samples,state_list,stationary_prob,train_logli_this,test_logli_this = \
                    EMBasins.pyHMM(nrnspiketimes, np.ndarray([]), np.ndarray([]),
                                        float(binsize), nModes, niter)
                train_logli[0,:] = train_logli_this.flatten()
                test_logli[0,:] = test_logli_this.flatten()
            # Save the fitted model
            #  for cross-validation case only the last iteration's data is saved,
            #   except train and test logli have all iterations' data
            saveFit(dataFileBase,nModes,params,trans,P,emiss_prob,alpha,pred_prob,hist,samples,state_list,stationary_prob,train_logli,test_logli)

        # temporally independent EMBasins
        else:
            # train on nrnspiketimes, test on nrnspiketimes_test
            # perhaps I should use crossval() to test and train on k subsets,
            #  but EMBasins::crossval() perhaps not fully implemented in EMBasins.cpp
            params,w,samples,state_list,state_hist,state_list_test,state_hist_test,P,P_test,prob,prob_test,train_logli,test_logli = \
                    EMBasins.pyEMBasins(nrnspiketimes, nrnspiketimes_test, float(binsize), nModes, niter)
            # P and P_test are the Qmodes/Z (cf. my MixtureModel.py calcModePosterior())
            #  for each state in train_states and test_states
            # prob and test_prob are the Z's for each state in train_states and test_states
            # Save the fitted model
            saveFit(dataFileBase,nModes,params,w,samples,state_list,state_hist,state_list_test,state_hist_test,P,P_test,prob,prob_test,train_logli,test_logli)

        print("Mixture model fitted for interaction factor = ",
                interactionFactor,', nModes = ',nModes,', nSeed = ',nSeed)
        sys.stdout.flush()    

        print("mixture model saved for interaction factor ",interactionFactor,
                                ', nSeed ',nSeed,
                                ', nModes ',nModes,' out of ',maxModes,
                                '.\n logL=',train_logli,'.\n logLTest=',test_logli)
        sys.stdout.flush()
