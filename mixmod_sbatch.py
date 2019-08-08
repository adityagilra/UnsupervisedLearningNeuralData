import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import scipy.io

import shelve, sys
from sklearn.manifold import MDS
from sklearn import discriminant_analysis as DA

from MixtureModel import MixtureModel, loadDataSet

np.random.seed(100)

interactionFactorList = np.arange(0.,2.,0.1)
interactionFactorList = np.append(interactionFactorList,[1.])

maxModes = 150
nModesList = range(1,maxModes+1,5)          # steps of 5, no need to go one by one

# for sbatch array jobs, $SLURM_ARRAY_TASK_ID is passed as first command-line argument
#  so give the sbatch array job with indexes corresponding to taskId
#   that you want to decode as below into interactionFactorIdx and nModesIdx
#  sbatch --array=0-630 submit_L2L.sbatch   # 30 nModes * 21 datasets=630
print(sys.argv)
if len(sys.argv) > 1:
    taskId = int(sys.argv[1])
    interactionFactorIdx = taskId // len(interactionFactorList)
    nModesIdx = taskId % len(interactionFactorList)
    interactionFactor = interactionFactorList[interactionFactorIdx]
    nModes = nModesList[nModesIdx]
else:
    interactionFactorIdx = 20       # experimental data
    nModes = 70                     # best nModes reported for exp data in Prentice et al 2016

# whether to fit mixture models and save data
# this takes the longest time, so beware
fitMixMod = True

dataFileBaseName = 'Learnability_data/synthset_samps'

# to fit MixMod for specific dataset and nModes
if interactionFactorIdx != 20:
    dataFileBase = dataFileBaseName + '_' + str(interactionFactorIdx+1)
else:
    dataFileBase = 'Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100'

if fitMixMod:
    spikeRaster = loadDataSet(dataFileBase, interactionFactorIdx)
    lessDataBy = 4          # take only fraction of the data by this number
    nNeurons,tSteps = spikeRaster.shape
    ## find unique spike patterns and their counts
    #spikePatterns, patternCounts = np.unique(spikeRaster, return_counts=True, axis=1)

    mixMod = MixtureModel(nModes,spikeRaster[:,:tSteps//lessDataBy])
    nRepeat = 40
    for i in range(nRepeat):
        mixMod.maximization()
        mixMod.expectation()
        print("Mixture model fitting for file number",interactionFactorIdx,"repeat",i)
        sys.stdout.flush()
    logL = mixMod.calcLogLikelihood()
    
    # Save the fitted model
    mixMod.saveFit(dataFileBase)
    print("mixture model fitting and saving for file number",interactionFactorIdx,
                        'nModes',nModes,'out of',maxModes)
    sys.stdout.flush()
