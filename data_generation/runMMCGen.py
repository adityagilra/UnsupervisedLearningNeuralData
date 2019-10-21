# FORWARD MONTE CARLO GATEWAY ROUTINE
# nS is the number of samples, sampling_freq is how many spin flips are
# done before another sample is collected. mv1 are mean value expectations
# in {0, 1} convention, cv1 are connected covariances, smp is {0, 1} array
# of samples and e1 is the time course of the every sampled at sampling
# freq. 
# sts(1) = variance of energy, sts(2) = mean magnetization, sts(3) =
# mean energy
# cooldownF is an approximate fraction of samples to equilibrate MCMC
# before real sampling starts; enter 0 to do no equilibration 

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import shelve, sys, os.path

np.random.seed(100)

from mxMaxentTGen import pyMaxentTGen, pyInit

def runMMCGen(coupls,nN,nS,frac,ffunc,state,seed=None,cooldownF=0.,suppressOut=False):
    frac = np.int(np.round(frac))
    nS = np.int(np.round(nS))
    if (nS <= 0):
        print('The number of samples must be positive')
        sys.exit(1)
    nfeatures = len(coupls)
    print('synthset shape = ',coupls.shape)
    if seed is None:
        seed = np.int(np.random.uniform() * 1000000)
    
    if not suppressOut:
        print('ISING MMC GEN: spins: {}, samples: {}, samp_freq: {}, burn-in frac: {}'\
                                .format(nN, nS, frac, cooldownF))
    
        # initial st.expec_scaled (1st arg) must be of size nfeatures
        i1, i2, esample, sts, sample = pyMaxentTGen(np.zeros(nfeatures), ffunc,
                                                        coupls, nN, 0, 0,
                                                        np.int(np.round(nS*cooldownF)),
                                                        np.int(np.round(nS*(1-cooldownF))),
                                                        frac, seed)
        # sample is originally numdata x numneurons, let's transpose it
        sample = sample.T

    mv1 = i2[:nN]
    cv1 = i2[nN:]

    # correct the energy estimate
    
    e0 = hamilGen(ffunc, np.zeros((nN,1)), coupls.flatten())
    sts = -sts - e0
    
    if not suppressOut:
        print('ISING MMC GEN: spins: {}, samples: {}, Mean energy {}, variance {}, magnetization {}.'\
                                .format(nN, nS, esample[2], esample[0], esample[1]))

    return (i1, mv1, cv1, esample, sts, sample)

# Variables for this are {0,1}!!!
def hamilGen(ffunc, state, coupls):
    if ffunc == 'KSpikeIsing':
        energy = -np.matmul(np.append(state, KSpikeIsing(state), axis=0).T, coupls)
    else:
        print('not implemented ffunc = ',ffunc)
        sys.exit(1)
    return energy

def KSpikeIsing(data):
    # If data is a Nx1 vector, computes the value of each of the K-spike
    # features, K=0,...,N  plus each of the two point correlations,
    #  in a [N(N+1)/2+1] x 1 vector.
    # If data is a NxT vector, computes the feature values at each column,
    #  in a [N(N+1)/2+1] x T vector
    # accepts {-1,1} or {0,1} data
    if np.amin(data) == -1:
        data = (data+1)/2
    N, T = data.shape
    SpikeCounts = np.sum(data,axis=0).astype(np.int)
    Features = np.zeros((N*(N+1)/2+1,T))
    for k in range(T):
        q = np.where(data[:,k]!=0)[0]
        for i in q:
            for j in range(i-1):
                # Aditya: I may be wrong in translating from matlab:
                # Features(N+1+ (q(i)-1)*(q(i)-2)/2 + q(j),k) = 1;
                Features[N + i*(i-1)/2 + q[j],k] = 1
    for k in range(T):
        Features[SpikeCounts[k],k] = 1
   
    return Features

if __name__=="__main__":
    # very important to call pyInit() first else seg faults
    pyInit()
    
    # load the fit
    # synthset_k_X_Y_Z.mat in the zip means:
    # X == number of neurons (always 4, meaning 120 neuron groups)
    # Y == the replicate (the subset of 120 neurons from data being studied; there is a lot of overlap of neurons within the group)
    # Z == integer representing alpha factor, alpha = synthset.factor, scales all correlations up and down.
    # In each synthset, there are parameters of the model fit (K-pairwise), which are [synthset.hs; synthset.js] in matlab
    fitData = scipy.io.loadmat('synthsets/synthset_k_4_8_10.mat')
    synthset_hs = fitData['synthset']['hs'][0,0]        # shape 120 x 1
    synthset_js = fitData['synthset']['js'][0,0]        # shape <.> x 1
    synthset_mean_rates = fitData['synthset']['mv0'][0,0].T

    figMM, axesMM = plt.subplots(nrows=1, ncols=1, figsize=(4,4))

    for seed in range(1):        
        # samples would be in smp_mc matrix (binary)
        # This is for 120 neurons, draw 100000 samples by recording a sample,
        #  doing 100 MC steps, recording a sample etc
        #  (so 100 is the sampling frequency).
        # KSpikeIsing is the form of the model, 0 is state.
        # The round(rand()...) stuff is the initial random seed for MC.
        # append synthset_hs and synthset_js along axis=0 as needed by mxMaxentTGen.cpp
        i1, mv1, cv1, esample, sts, sample = \
            runMMCGen(np.append(synthset_hs,synthset_js,axis=0), 120, 100000, 100,
                            'KSpikeIsing', state=0, seed=np.int(seed*1000000))
        #database = shelve.open('../Learnability_data/generated_data_alpha'\
        #                        +str(seed),flag='c')
        #database['sample'] = sample
        #shelve.close()

        axesMM.plot(synthset_mean_rates, np.mean(sample,axis=1), 'ko')
    
    plt.show()
