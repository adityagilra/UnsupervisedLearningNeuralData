import numpy as np
import scipy.io
import shelve, sys, os.path

from MixtureModel import MixtureModel, loadDataSet, loadFit

np.random.seed(100)

interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.])

maxModes = 150
nModesList = range(1,maxModes+1,5)          # steps of 5, no need to go one by one

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

if fitMixMod:
    if loadEarlierFits:
        if os.path.isfile(dataFileBase+'_mixmod_modes'+str(nModes)+'.shelve'):
            print("Continuing from fitted model for file number ",interactionFactorIdx)
            paramsFileBase = dataFileBase
        else:
            print("Earlier fit for file number ",interactionFactorIdx,
                    " not found, exiting...")
            sys.exit(1)
    else:
        paramsFileBase = None

    spikeRaster = loadDataSet(dataFileBase, interactionFactorIdx)
    nNeurons,tSteps = spikeRaster.shape
    ## find unique spike patterns and their counts
    #spikePatterns, patternCounts = np.unique(spikeRaster, return_counts=True, axis=1)

    mixMod = MixtureModel(nModes,spikeRaster,
                            tStepsFit=tSteps//4,tStepsTest=tSteps//4,
                            loadParamsFileBase=paramsFileBase)

    nRepeat = 100
    for i in range(nRepeat):
        # start with maximization if using mixMod.PCond's initialization
        # or expectation if using mixMod.wModes and mixMod.mProb 's initialization
        # Prentice et al 2016 start with expectation, I've used their initialization distribution
        mixMod.expectation()
        mixMod.maximization()
        print("Mixture model fitting for file number",interactionFactorIdx,"repeat",i)
        sys.stdout.flush()
    #logL = mixMod.calcLogLikelihood()      # done as part of mixMod.saveFit() below
    
    # Save the fitted model
    logL,logLTest = mixMod.saveFit(dataFileBase)
    print("mixture model fitting and saving for file number ",interactionFactorIdx,
                                    ', nModes ',nModes,' out of ',maxModes,
                                    '. logL=',logL,', logLTest=',logLTest)
    sys.stdout.flush()
