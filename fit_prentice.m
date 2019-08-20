filename = "Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100";
retinaData = load(filename+".mat",'bint');

%binsize = 200; % binsize is # of samples / 20 ms time bin @ 10 kHz sampling rate.
binsize = 1;
nbasins = 70;

spikeRaster = retinaData.bint;
[numtrials,numneurons,numbins] = size(spikeRaster);
bins = 1:numbins*numtrials;
nrnSpikeTimes = cell(numneurons,1);

for nrnnum = 1:numneurons
    for trialnum = 1:numtrials
        spikeTimes = bins(diff([0,squeeze(spikeRaster(trialnum,nrnnum,:))']) == 1) + ...
                        (trialnum-1)*numbins;
        nrnSpikeTimes{nrnnum} = [nrnSpikeTimes{nrnnum},spikeTimes];
    end
end

% validation/test set is statistically very different from training set --
%  I got poor test log-likelihood despite high training log-likelihood
% maybe because similar images were presented temporally together during the experiment?
% to overcome this, I'm randomly permuting the full dataset
% achtung: problematic if fitting a temporal model and/or retina has adaptation
%shuffled_idxs = np.random.permutation(np.arange(tSteps,dtype=np.int32))
%spikeRaster = spikeRaster[:,shuffled_idxs]

% Achtung: I think the number of params returned is different,
%  check again with code in EMBasins.cpp
[freq,w,m,P,logli,prob] = ...
    EMBasins(nrnSpikeTimes, binsize, nbasins, 100);
