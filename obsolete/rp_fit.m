% Usage:
% model = rp_fit('/nfs/scistore12/gaspgrp/agilra/UnsupervisedLearningNeuralData/Learnability_data/synthset_samps_11.mat')

function model = rp_fit(filename)
    load(filename);
    nNeurons = size(synthset.smp,1);
    tSteps = size(synthset.smp,2);
    idxTrain = int64(tSteps/4);
    spikesTrain = synthset.smp(:,1:idxTrain);
    spikesTest = synthset.smp(:,idxTrain+1:end);
    
    model = maxent.createModel(nNeurons,'rp')
    % training to 2 SD instead of 1 on https://orimaoz.github.io/maxent_toolbox/maxent_example.html
    % else training to 1 SD did not happen even for ~1.5 hours.
    model = maxent.trainModel(model,spikesTrain,'threshold',1.5)
    
    empirical_distribution = maxent.getEmpiricalModel(spikesTest);
    model_logprobs = maxent.getLogProbability(model,empirical_distribution.words);
    test_dkl = maxent.dkl(empirical_distribution.logprobs,model_logprobs);
    fprintf('Kullback-Leibler divergence from test set: %f\n',test_dkl);
end