import numpy as np
import scipy.io
import shelve, sys, os.path

import VIWTA_SNN
VIWTA_SNN.pyInit()

np.random.seed(100)

#__________________________________________________________________________
# Description: Performs training and testing of the VI-WTA SNN.
#              This is simply a Python wrapper that calls the Boost-ed
#              function VIWTA_SNN (written in C++).
#__________________________________________________________________________

# -- Specify Hyper-Parameter Values: --
dataset  = 'daily'
m        = 10         #number of readout neurons in VIWTA circuit (i.e. # of latent modes)
eta_b    = 0.001      #Learning rate hyperparameter 
eta_W    = 0.0004     #Learning rate hyperparameter
save     = True

if dataset == 'minutely':
   savename = 'VIWTA_minutely_etab' + str(eta_b) + '_etaW' + str(eta_W)
elif dataset == 'daily':
   savename = 'VIWTA_daily_etab' + str(eta_b) + '_etaW' + str(eta_W)

mixing_weights = (1./m)*np.ones(m)

# THIS IS NOW OBSOLETE! Earlier the datafile was hard-coded into the C++ code,
#  I modified it, so look at the ../WTAcluster_sbatch.py code to see usage
# -- Train & Test via VI (Variational Inference) WTA Circuit: --
W_star, b_star, Converg_avgW, readout_train, readout_test = \
      VIWTA_SNN.pyWTAcluster(m, eta_b, eta_W, mixing_weights)

print("W_star, b_star, Converg_avgW, readout_train, readout_test")
print(W_star, b_star, Converg_avgW, readout_train, readout_test)
 
if save:
    dataBase = shelve.open(savename+'.shelve')
    dataBase['W_star'] = W_star
    dataBase['b_star'] = b_star
    dataBase['Converg_avgW'] = Converg_avgW
    dataBase['readout_train'] = readout_train
    dataBase['readout_test'] = readout_test
    dataBase.close()
