# UnsupervisedLearningNeuralData

Fitting various models to neural data in an unsupervised manner (stimuli for data are unknown).

-------------

## Using Jason Prentice and Adrianna Loback's code for fitting (as provided by Gasper Tkacik):  
First I need to compile BasinModel.cpp, etc. for linux.  
Just compile, don't link, hence -c:  
EMBasins.cpp uses Matlab's matrix.h and mex.h, hence the -I, see:  
 https://www.mathworks.com/help/matlab/matlab_external/mat-file-library-and-include-files.html  
Also matlab complained when mex-ing, and suggested -fPIC  
g++ -I/usr/local/MATLAB/R2019a/extern/include/  -fPIC -c EMBasins.cpp  
g++  -fPIC -c BasinModel.cpp  
g++  -fPIC -c TreeBasin.cpp  
.o files are created and I don't need to link them, as I will mex them for Matlab.  
    
Now in matlab, as per Adrianna's documentation:  
mex -largeArrayDims -I/usr/local/include -I/usr/local/Cellar/boost/1.68.0 -lgsl -lgslcblas EMBasins.cpp BasinModel.o TreeBasin.o  
Now mex completed successfully, so I suppose Boost libraries (correct version 1.68.0 too) were already present on IST cluster!  
  
Now copied EMBasins.mexa64 (note Adrianna's one had mexmaci, not mexa -- it was for mac) to my working directory, and ran Matlab from there.  
In matlab on the CLI, I tried: params = EMBasins(), now it crashed with an internal error,  
 but I think that's because I didn't pass any params!  
 At least, it doesn't say EMBasins not found.  

From matlab when EMBasins() is called, actually the mexFunction() inside EMBasins.cpp gets called. See:  
https://www.mathworks.com/help/matlab/apiref/mexfunction.html  
Currently mexFunction() creates an HMM model, but I could modify it to use the IndependentBasin.  
However, the IndependentBasin constructor does not seem to take in spike trains. Does it generate data itself?  
    
Actually mexFunction() contains commented code to use MixtureModel i.e. EMBasins class, not HMM class:  
 Perhaps correlations / tree term can be removed by modifying this statement at the top of EMBasins.cpp  
 // Selects which basin model to use  
 typedef TreeBasin BasinType;  
 to  
 typedef IndependentBasin BasinType;  
Thus I can switch from HMM to EMBasins to remove time-domain correlations,  
 and TreeBasin to IndependentBasin to remove space-domain correlations.  
    
    

