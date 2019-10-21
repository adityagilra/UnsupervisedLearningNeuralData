/*
 * General code to perform coordinate-descent Schapire-style maxent learning for data 
 * consisting of binary vectors.  Optimized for data where "1"'s are sparse.  
 * The maxent features can be arbitrary functions, aside from the restrictions that they must be binary and that the first N 
 * features (where N is the dimension of the binary vectors) must be the N "coordinate" features. 
 * The remaining features are specified through a feature function with prototype:
 *
 * void featureFunction(int* prevSample, int nonzero_vars, int* nonzero_features, int NUMVARS)
 * 
 * prevSample = a binary vector expressed as an array containing its nonzero coordinates [I need to make this clearer -- DA]
 * nonzero_vars = number of nonzero coordinates in the binary vector
 * nonzero_features = number of features which are nonzero (output)
 * NUMVARS = dimension of the binary vector
 *
 *
 *INPUTS: 
 * exmatlab = empirical expectations to be used for fitting maxent
 * NUMVARS = number of cells/spins
 * NUMRUNS = number of outer loop MC-learning iterations to do
 * TCOUNT = number of inner loop learning iterations to do
 * NUMSAMPS = size of MCMC sampling run during learning
 * NUMDATA = size of final (output) MCMC sampling run
 * SKIP = number of MCMC steps to traverse and skip for each one saved
 * SEED = seed for randomizer
 * NUMFCNS = number of features (including the obligatory NUMVARS means)
 * lambda_guess = initial guess for coupling constants (NUMFNCS x 1 array)(optional)
 * 
 *OUTPUTS:
 * lambda_out = final coupling constants for maxent model (NUMFNCS x 1 array)
 * expect_out = expectation values of features in final maxent model (NUMFNCS x 1 array)
 * mc_final = full MCMC sampling run taken from final fitted maxent model (NUMVARS*NUMDATA x 1 array) (optional)
 * sts = energy of each sample from the final MCMC sampling run (NUMDATA x 1 arrays) (optional)
 * esample = statistics from the MCMC sampling run -- magnetization, mean energy, variance of energy (3 x 1 array) (optional)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "mt19937ar.h"
#include "maxentTGen.h"

// Choose either MATLAB or PYTHON to link to via Boost
//#define MATLAB
#define PYTHON

#ifdef MATLAB
//#include "matrix.h"
#include "mex.h"
#endif

#ifdef MATLAB

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // defaults
    int NUMVARS = 2; 
    int NUMRUNS = 10;  
    int TCOUNT = 20; 
    int NUMSAMPS = 10000; 
    int NUMDATA = NUMSAMPS;  
    int SKIP = NUMVARS; 
    int SEED = 77;  
    int NUMFCNS = 0; 
    int sampFlag = 0; // flag to decide whether to run & output a final MCMC sample at the end.  Will be set to 1 below if function is called with >2 output args
    int status;
    char *ffunc;
    
    //read in data
    if (nrhs == 0) return;
    double* exmatlab = (double*)mxGetPr(prhs[0]);
    double* lambda_guess = 0;
    int*    init_guess   = 0;
    int nN = mxGetN(prhs[0]);
    int nM = mxGetM(prhs[0]);
    int sz = (int)max(nN, nM);
    mwSize strl = mxGetN(prhs[8])*sizeof(mxChar)+1;
    ffunc = (char*)mxMalloc(strl);
    if (nrhs > 1)
    {
        NUMVARS = (int)mxGetScalar(prhs[1]);
        NUMRUNS = (int)mxGetScalar(prhs[2]);
        TCOUNT = (int)mxGetScalar(prhs[3]);
        NUMSAMPS = (int)mxGetScalar(prhs[4]);
        NUMDATA = (int)mxGetScalar(prhs[5]);
        SKIP = (int)mxGetScalar(prhs[6]);
        SEED = (int)mxGetScalar(prhs[7]);
        if(status = mxGetString(prhs[8], ffunc, strl)) return;
        NUMFCNS = sz;
        if (nrhs >= 10)
        {
            lambda_guess = (double*)mxGetPr(prhs[9]);
        }
        if (nrhs == 11)
        {
            init_guess  = (int*)mxGetPr(prhs[10]);
        }
    }
    // end reading in data
    // setup output data
    plhs[0] = mxCreateDoubleMatrix(sz, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(sz, 1, mxREAL);
    double* lambda_out = mxGetPr(plhs[0]);
    double* expect_out = mxGetPr(plhs[1]);
    double* sts = 0;
    double* esample = 0;
    double* mc_final = 0;
    if(nlhs>2) { // # output args determines whether to save final sample
        sampFlag = 1;
        plhs[2] = mxCreateDoubleMatrix(3,1,mxREAL);
        plhs[3] = mxCreateDoubleMatrix(NUMDATA,1,mxREAL);
        plhs[4] = mxCreateDoubleMatrix(NUMDATA*NUMVARS,1,mxREAL);
        esample = mxGetPr(plhs[2]);
        sts = mxGetPr(plhs[3]);
        mc_final = mxGetPr(plhs[4]); 
    }
    // end setup output data
    
    montestats st;
    monteruns mr;
    mr.prevSample = 0;
    mr.prevSampleDiff = 0;
    mr.plus = 0;
    mr.minus = 0;
    mr.ffunc = ffunc;
    // allocate memory and initialize sampling statistics
    setupMonteSt(&st, NUMFCNS, NUMVARS, NUMSAMPS);
    setupSample(&st, &mr, NUMFCNS,SEED);
    // done allocating memory and initializing
   // mexPrintf("mxMaxentT.cpp:\nVariables %d, functions %d, runs %d, samples %d, lsteps %d.\n", NUMVARS, NUMFCNS, NUMRUNS, NUMSAMPS, TCOUNT);
  //  mexPrintf("First input mean %g, first cov %g. ",exmatlab[0], exmatlab[NUMVARS]);
   // if (lambda_guess == 0) mexPrintf("\n"); else mexPrintf("Using supplied initial guess.\n");
    
    double* ex = (double*)malloc(sizeof(double) * NUMFCNS);

    st.expec_scaled =  (double*) malloc(sizeof(double) * NUMFCNS);
    for (int i = 0; i < NUMFCNS; i++) st.expec_scaled[i] = exmatlab[i];
        
    // set the initial coupling constants (to the default guess or the provided initial guess) and do an initial sampling run
    initialGuess(&st, &mr, NUMVARS, NUMFCNS, NUMSAMPS, lambda_guess, SKIP);
    for(int numruns = 0; numruns < NUMRUNS; numruns++) 
    {    
        setupSample(&st, &mr, NUMFCNS,SEED); // initialize MCMC sampler
        for(int mc = 0; mc < NUMSAMPS; mc++) 
        {
            st.g_monte_lambda[mc] = sample(&mr, st.lambda, NUMVARS,SKIP); // take sample
            if (processSamples(&st, &mr, mc) < 0) return; // update statistics based on sample
        }
        st.sum_gexpg_monte_eta = 0;
        st.sum_energyexpg_monte_eta = 0;
        
        for (int tsample = 0; tsample < TCOUNT; tsample++) 
        {
            learn_alg(&st, NUMFCNS, NUMSAMPS); // run inner loop iteration
        }
        for(int j = 0; j < NUMFCNS; j++) 
        {
            st.lambda[j] += st.eta[j]; // update the coupling constants
        }
    }
    // final sample to generate expectation values, sample will be stored and output if sampFlag = 1
    finalGuess(&st, &mr, NUMFCNS, NUMDATA, NUMVARS, ex, sts, esample, mc_final, SKIP, sampFlag,SEED,init_guess);  
    
    // output the final coupling constants and expectation values
    for (int i = 0; i < NUMFCNS; i++)
    {
        lambda_out[i] = st.lambda[i];
        expect_out[i] = ex[i];
    }
    cleanupMonteSt(&st,&mr, NUMFCNS);
    free(ex);
}

#endif

#ifdef PYTHON

#include<boost/python.hpp>
// numpy.hpp needs boost >= 1.63.0, on IST cluster: `module load boost` to get 1.70.0
#include<boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

template <typename T>
np::ndarray writePyOutputMatrix(T* arrptr, int rows, int cols) {
// convert C++ 2D array (passing only pointer to first element) to a Python numpy array of rows x cols
    
    // https://www.boost.org/doc/libs/1_71_0/libs/python/doc/html/numpy/tutorial/ndarray.html
    np::dtype dt = np::dtype::get_builtin<T>();   

    //py::tuple shape = py::make_tuple(value.size());    // size gives only rows
    py::tuple shape = py::make_tuple(rows,cols);

    //py::tuple stride = py::make_tuple(sizeof(double));
    //py::object own;
    // https://www.boost.org/doc/libs/1_71_0/libs/python/doc/html/numpy/tutorial/fromdata.html
    // from_data uses the same location for both arrays in C and python
    //np::ndarray arr = np::from_data (value,dt,shape,stride,own);
    
    np::ndarray arr = np::empty(shape, dt);
    //std::cerr << "rows = " << rows << " cols = " << cols << std::endl;
    for (int i=0; i<rows; i++)
        for (int j=0; j<cols; j++) {
            arr[i][j] = arrptr[i*cols+j];
            //std::cerr << i << " " << j << " " << py::extract<char const *>(py::str((arr[i][j]))) << " ?= " << arrptr[i*cols+j] << std::endl;
            }
    return arr;
}

std::vector<double> getVec(np::ndarray arr) {
    int input_size = arr.shape(0);
    std::vector<double> v(input_size);
    for (int i=0; i<input_size; ++i)
        v[i] = py::extract<double>(arr[i]);
    return v;
}

std::vector<std::vector<double>> getMat(np::ndarray arr) {
    int nrows = arr.shape(0);
    int ncols = arr.shape(1);
    std::vector<std::vector<double>> v(nrows, std::vector<double>(ncols,0.0));
    for (int i=0; i<nrows; ++i)
        for (int j=0; j<ncols; ++j)
        v[i][j] = py::extract<double>(arr[i][j]);
    return v;
}

py::list pyMaxentTGen(np::ndarray & arr, py::str strl, np::ndarray & lambda_guess_arr,
                        int NUMVARS=2, int NUMRUNS=10, int TCOUNT=20, int NUMSAMPS=10000,
                        int NUMDATA=10000, int SKIP=2, int SEED=77) {

    // defaults
    //int NUMVARS = 2; 
    //int NUMRUNS = 10;  
    //int TCOUNT = 20; 
    //int NUMSAMPS = 10000; 
    //int NUMDATA = NUMSAMPS;  
    //int SKIP = NUMVARS; 
    //int SEED = 77;  
    //int NUMFCNS = 0; 
    //double* lambda_guess = 0;
    int*    init_guess = 0;
    
    std::cout << "reading in args start" << std::endl;
    
    // flag to decide whether to run & output a final MCMC sample at the end,
    //  will be set to 1 below if function is called with >2 output args, default = 0;
    int sampFlag = 0;
    int status;
    char *ffunc;
    
    //read in data
    int nN = arr.shape(0);
    int sz = nN;
    int NUMFCNS = sz;
    ffunc = py::extract<char *>(strl);
    // end reading in data
    
    // setup output data
    sampFlag = 1;
    double* esample = (double*)malloc(sizeof(double)*3);
    double* sts = (double*)malloc(sizeof(double)*NUMDATA);
    double* mc_final = (double*)malloc(sizeof(double)*NUMDATA*NUMVARS);
    // end setup output data
        
    montestats st;
    monteruns mr;
    mr.prevSample = 0;
    mr.prevSampleDiff = 0;
    mr.plus = 0;
    mr.minus = 0;
    mr.ffunc = ffunc;
    // allocate memory and initialize sampling statistics
    setupMonteSt(&st, NUMFCNS, NUMVARS, NUMSAMPS);
    setupSample(&st, &mr, NUMFCNS,SEED);
    // done allocating memory and initializing
    // printf("mxMaxentT.cpp:\nVariables %d, functions %d, runs %d, samples %d, lsteps %d.\n", NUMVARS, NUMFCNS, NUMRUNS, NUMSAMPS, TCOUNT);
    // printf("First input mean %g, first cov %g. ",arr[0], arr[NUMVARS]);
    // if (lambda_guess == 0) printf("\n"); else printf("Using supplied initial guess.\n");
    
    double* ex = (double*)malloc(sizeof(double) * NUMFCNS);
    st.expec_scaled = (double*) malloc(sizeof(double) * NUMFCNS);
    for (int i = 0; i < NUMFCNS; i++)
        st.expec_scaled[i] = py::extract<double>(arr[i]);

    double* lambda_guess = (double*)malloc(sizeof(double) * len(lambda_guess_arr));
    for (int i = 0; i < len(lambda_guess_arr); i++)
        lambda_guess[i] = py::extract<double>(lambda_guess_arr[i][0]);

    std::cerr << "reading in args end " << std::endl;

    // set the initial coupling constants (to the default guess or the provided initial guess) and do an initial sampling run
    std::cout << "before lambda" << std::endl;
    initialGuess(&st, &mr, NUMVARS, NUMFCNS, NUMSAMPS, lambda_guess, SKIP);
    std::cout << "after lambda" << std::endl;
    for(int numruns = 0; numruns < NUMRUNS; numruns++) 
    {    
        setupSample(&st, &mr, NUMFCNS,SEED); // initialize MCMC sampler
        for(int mc = 0; mc < NUMSAMPS; mc++) 
        {
            st.g_monte_lambda[mc] = sample(&mr, st.lambda, NUMVARS,SKIP); // take sample
            if (processSamples(&st, &mr, mc) < 0) return py::list(); // update statistics based on sample
        }
        st.sum_gexpg_monte_eta = 0;
        st.sum_energyexpg_monte_eta = 0;
        
        for (int tsample = 0; tsample < TCOUNT; tsample++) 
        {
            learn_alg(&st, NUMFCNS, NUMSAMPS); // run inner loop iteration
        }
        for(int j = 0; j < NUMFCNS; j++) 
        {
            st.lambda[j] += st.eta[j]; // update the coupling constants
        }
    }
    // final sample to generate expectation values, sample will be stored and output if sampFlag = 1
    finalGuess(&st, &mr, NUMFCNS, NUMDATA, NUMVARS, ex, sts, esample, mc_final, SKIP, sampFlag, SEED, init_guess);  
    
    py::list outlist = py::list();
    // output the final coupling constants and expectation values    
    outlist.append(writePyOutputMatrix(st.lambda, sz, 1)); // lambda_out
    outlist.append(writePyOutputMatrix(ex, sz, 1)); // expect_out
    // ** Return Learned Parameters **
    outlist.append(writePyOutputMatrix(esample, 3, 1));
    outlist.append(writePyOutputMatrix(sts, NUMDATA, 1));
    outlist.append(writePyOutputMatrix(mc_final, NUMDATA, NUMVARS));

    cleanupMonteSt(&st,&mr, NUMFCNS);
    free(ex);
    free(sts);
    free(esample);
    free(mc_final);
    free(lambda_guess);

    return outlist;
}

void pyInit() {
    // https://www.boost.org/doc/libs/1_71_0/libs/python/doc/html/numpy/tutorial/simple.html
    // Initialise the Python runtime, and the numpy module. Failure to call these results in segmentation errors!
    Py_Initialize();
    np::initialize();
    std::cout << "Initialized python and numpy" << std::endl;
}

// the module name specified here must match the name of the .so created via the Makefile
// else: `from mxMaxentTGen import pyMaxEntTGen` gives
// ImportError: dynamic module does not define init function (initmxMaxentTGen)
// see the boost::python answer at:
//  https://stackoverflow.com/questions/24226001/importerror-dynamic-module-does-not-define-init-function-initfizzbuzz
BOOST_PYTHON_MODULE(mxMaxentTGen)
{
   using namespace boost::python;
   def("pyMaxentTGen",pyMaxentTGen);
   def("pyInit",pyInit);
}

#endif

void setupMonteSt(montestats* st,int NUMFCNS, int NUMVARS, int NUMSAMPS)
{
    // allocate memory for sampling statistics
    st->sum_fjexpg_monte_eta = (double *) malloc(NUMFCNS * sizeof(double));
    st->Z_monte_eta = 0;
    st->Z_monte_lambda = 0;
    st->g_monte_eta = (double *) malloc(NUMSAMPS * sizeof(double));
    st->g_monte_lambda = (double *) malloc(NUMSAMPS * sizeof(double));
    st->eta = (double *) malloc(NUMFCNS * sizeof(double));
    st->lambda = (double *) malloc(NUMFCNS * sizeof(double));
    //st->expec_scaled = (double *) malloc(NUMFCNS * sizeof(double));

    st->count_monte = (int *) malloc(NUMFCNS * sizeof(int));
    st->bound_monte = (int *) malloc(NUMFCNS * sizeof(int));
    st->sInds_monte = (int **) malloc(NUMFCNS * sizeof(int *));
    // initialize sampling statistics
    for(int j = 0; j < NUMFCNS; j++) 
    {
        st->bound_monte[j] = 1;
        st->count_monte[j] = 0;
        st->sInds_monte[j] = (int *) malloc(st->bound_monte[j] * sizeof(int));
    }

    
    st->sum_gexpg_monte_eta = 0;
    st->sum_energyexpg_monte_eta = 0;
    st->rel_entropy_monte_theta_lambda = 0;
    st->mean_energy_monte_eta = 0;
}

void cleanupMonteSt(montestats* st,monteruns* mr, int NUMFCNS)
{
    free(st->sum_fjexpg_monte_eta);
    free(st->g_monte_eta);
    free(st->g_monte_lambda);
    free(st->eta);
    free(st->lambda);
    free(st->expec_scaled);
    free(st->count_monte);
    free(st->bound_monte);
    for (int i = 0; i < NUMFCNS; i++)
    {
        free(st->sInds_monte[i]);
    }
    free(st->sInds_monte);
    free(mr->prevSample);
    free(mr->prevSampleDiff);
    free(mr->plus);
    free(mr->minus);
}


void setupSample(montestats* st, monteruns* mr, int NUMFCNS,int SEED) 
{
    if (mr->prevSample != 0) free(mr->prevSample);
    if(mr->prevSampleDiff != 0) free(mr->prevSampleDiff);
    if(mr->plus != 0) free(mr->plus);
    if(mr->minus != 0) free(mr->minus);
    else 
    {
        mr->seed = SEED;  
        // initialize the random number generator
        init_genrand(mr->seed);
    }
        
    mr->prevSample = (int*)malloc(sizeof(int)*NUMFCNS);
    mr->prevSampleDiff = (int*)malloc(sizeof(int)*NUMFCNS);
    mr->plus = (int*)malloc(sizeof(int)*NUMFCNS);
    mr->minus = (int*)malloc(sizeof(int)*NUMFCNS);
    mr->sampleCountFcns = 0;
    mr->sampleCountVars = 0;
    
    for(int j = 0; j < NUMFCNS; j++)
    {
        st->count_monte[j] = 0;
        st->bound_monte[j] = 1;
        st->sum_fjexpg_monte_eta[j] = 0;
        st->eta[j] = 0;
    }
        
    st->Z_monte_lambda = 0;
    st->Z_monte_eta = 0;
}


double sample(monteruns* mr, double* lambda, int NUMVARS, int SKIP)
{
    double Dblx = 0.0;
    int isOne = 0; 
    int eqVar = 0;
    double p = 0.0;
    double r = 0.0;
    int dummy = 0;
    int dummy2 = 0;
    int nplus = 0;
    int nminus = 0;
    for(int n = 0; n < SKIP; n++) {
        int i = (int)floor(NUMVARS * genrand_real2());
        p = 0.0;
        isOne = 0;
        for(int j = 0; j < mr->sampleCountVars; j++)
        {
            int k = mr->prevSample[j];
            if(k == i) {
                isOne = 1;
                eqVar = j;
            }
            else {
                mr->prevSampleDiff[j-isOne] = mr->prevSample[j];
            }
        }
        diffFunction(mr->prevSampleDiff, mr->sampleCountVars-isOne, i,NUMVARS,mr->plus,&nplus,mr->minus,&nminus,mr->ffunc);
        for(int j=0;j<nplus;j++) p += lambda[mr->plus[j]];
        for(int j=0;j<nminus;j++) p -= lambda[mr->minus[j]];
//         if(isOne) {
//             featureFunction(mr->prevSample,mr->sampleCountVars,&dummy,NUMVARS,mr->ffunc);
//             featureFunction(mr->prevSampleDiff,mr->sampleCountVars-1,&dummy2,NUMVARS,mr->ffunc);
//             for(int j=mr->sampleCountVars;j<dummy;j++) {
//                 p += lambda[mr->prevSample[j]];
//             }
//             for(int j=mr->sampleCountVars-1;j<dummy2;j++) {
//                 p -= lambda[mr->prevSampleDiff[j]];
//             }
//         } else if(!isOne) {
//             mr->prevSampleDiff[mr->sampleCountVars] = i;
//             featureFunction(mr->prevSample,mr->sampleCountVars,&dummy,NUMVARS,mr->ffunc);
//             featureFunction(mr->prevSampleDiff,mr->sampleCountVars+1,&dummy2,NUMVARS,mr->ffunc);
//             for(int j=mr->sampleCountVars;j<dummy;j++) {
//                 p -= lambda[mr->prevSample[j]];
//             }
//             for(int j=mr->sampleCountVars+1;j<dummy2;j++) {
//                 p += lambda[mr->prevSampleDiff[j]];
//             }
//         }          
        p = 1.0/(1.0 + exp(-lambda[i] - p));
        r = genrand_real2();
        if(r < p) { 
            // a p chance of seting the i_th coordinate to 1 if it is not already 1 
            if(!isOne) {
                mr->prevSample[mr->sampleCountVars] = i;
                (mr->sampleCountVars)++;
            }
        } else if(isOne) { // a 1-p chance of changing it to zero if it is 1
            for(int j = eqVar; j < mr->sampleCountVars - 1; j++) { mr->prevSample[j] = mr->prevSample[j+1]; }
            mr->sampleCountVars--;
        }
    }
        
    for(int i = 0; i < mr->sampleCountVars; i++) {
        int k = mr->prevSample[i];
        Dblx += lambda[k];         
    }
    featureFunction(mr->prevSample,mr->sampleCountVars,&dummy,NUMVARS,mr->ffunc);
    mr->sampleCountFcns = dummy;
    for(int i=mr->sampleCountVars; i< mr->sampleCountFcns; i++) {
        Dblx += lambda[mr->prevSample[i]]; 
    }
    return Dblx;
}


int processSamples(montestats* st, monteruns* mr, int mc)
{
    // compute base partition function for sample and reset deviation measures 
    st->Z_monte_lambda += exp(st->g_monte_lambda[mc]);
    st->g_monte_eta[mc] = 0;
    st->Z_monte_eta++;
    // done partition function 
            
    if(mr->sampleCountFcns > 0) 
    {
        for(int j = 0; j < mr->sampleCountFcns; j++) 
        {
            int i0 = mr->prevSample[j];
            // allocate more memory for locations of nonzero features in MCMC sampling run if necessary
            if(st->count_monte[i0] >= st->bound_monte[i0]) 
            {
                st->bound_monte[i0] *= 2;
                st->sInds_monte[i0] = (int *) realloc((void *) (st->sInds_monte[i0]), st->bound_monte[i0] * sizeof(int));
                if(st->sInds_monte[i0] == 0) 
                {
                    printf("Couldn't allocate to sInds_monte[%d], bound=%d, count=%d\n", i0, st->bound_monte[i0], st->count_monte[i0]);
                    return -1;
                }
            }
            // done allocating memory 
            // update sampling run statistics 
            st->sInds_monte[i0][st->count_monte[i0]] = mc;
            st->count_monte[i0]++;
            st->sum_fjexpg_monte_eta[i0]++;
            // done sampling run statistics
        }
    }
    return 0;
}


void learn_alg(montestats* st, int NUMFCNS, int NUMSAMPS) {
    int minJ = 0;
    double minDel = 0;
    int mc = 0;
    double estimfj = 0;
    double expdelta = 0;
    double DelL = 0;
    double mindelta = 0;
    double oldg = 0;
    double newg = 0; 
    
    // START OPTIMIZING PARAMETERS
    // find the best lambda to change
    for(int j = 0; j < NUMFCNS; j++) {
        estimfj = st->sum_fjexpg_monte_eta[j] / st->Z_monte_eta;
        expdelta = st->expec_scaled[j] * (1 - estimfj);
        expdelta /= (1 - st->expec_scaled[j]) * estimfj;
        DelL = -log(expdelta) * st->expec_scaled[j];
        DelL += log(1 + (expdelta - 1) * estimfj);
        
        if(DelL < minDel) {
            minDel = DelL;
            minJ = j;
            mindelta = log(expdelta);
        }
    }
    // DONE OPTIMIZING PARAMETERS 
    
    // START UPDATING
    st->eta[minJ] += mindelta;
    // start Monte Carlo run updates
    for(int i0 = 0; i0 < st->count_monte[minJ]; i0++) {
        mc = st->sInds_monte[minJ][i0];
        oldg = st->g_monte_eta[mc];
        st->g_monte_eta[mc] += mindelta;
        newg = st->g_monte_eta[mc];
        
        st->Z_monte_eta += exp(newg) - exp(oldg);
        
        // start monte checks
        st->sum_gexpg_monte_eta += newg * exp(newg) - oldg * exp(oldg);
        st->sum_energyexpg_monte_eta += st->g_monte_lambda[mc] * (exp(newg) - exp(oldg));
        // done monte checks 
    } 
    // done Monte Carlo run updates
    
    // start function expectation updates
    for(int j = 0; j < NUMFCNS; j++) {
        st->sum_fjexpg_monte_eta[j] = 0;
        for(int i0 = 0; i0 < st->count_monte[j]; i0++) {
            st->sum_fjexpg_monte_eta[j] += exp(st->g_monte_eta[st->sInds_monte[j][i0]]);
        }
    }
    // done function expectation udpates
    // start monte
    st->rel_entropy_monte_theta_lambda = -log(st->Z_monte_eta / NUMSAMPS) + st->sum_gexpg_monte_eta / st->Z_monte_eta;
    st->mean_energy_monte_eta = st->sum_energyexpg_monte_eta / st->Z_monte_eta;
    // done monte
    /* DONE UPDATING */
    return;
}


void initialGuess(montestats* st, monteruns* mr, int NUMVARS, int NUMFCNS, int NUMSAMPS, double* lambda_guess,int SKIP) 
{
    // start estimating the h's without the J's
    int i;
    int j = 0;
    int k = 0;
    for(; j < NUMVARS; j++)
    {
            if (lambda_guess == 0) st->lambda[j] = log(st->expec_scaled[j] / (1 - st->expec_scaled[j]));
            else st->lambda[j] = lambda_guess[j];
    }
    for(; j < NUMFCNS; j++) 
    {
        if (lambda_guess == 0) st->lambda[j] = 0;
        else st->lambda[j] = lambda_guess[j];
    }
    // done estimating the h's 
    for(int mc = 0; mc < NUMSAMPS; mc++) 
    {
        sample(mr, st->lambda, NUMVARS,SKIP);
    }
}  


void finalGuess(montestats* st, monteruns* mr, int NUMFCNS, int NUMDATA, int NUMVARS, double* ex, double* sts, double* esample, double* mc_final, int SKIP, int sampFlag,int SEED,int* init_guess)
{
    double esq=0; // sum squared energy
    double emn=0; // sum energy
    double mmn=0; // sum number of spins
    setupSample(st, mr, NUMFCNS,SEED);
    for (int i = 0; i < NUMFCNS; i++) ex[i] = 0;
    if (init_guess != 0)
    {
        int dummy = 0;
        for (int i = 0; i < NUMVARS; i++)
        {
            if (init_guess[i] == 1)
            {
                 mr->prevSample[mr->sampleCountVars] = i;
                (mr->sampleCountVars)++;
    //            printf("%d ", i);
            }
        }
        featureFunction(mr->prevSample,mr->sampleCountVars,&dummy,NUMVARS,mr->ffunc);
        mr->sampleCountFcns = dummy;
     //   printf("Dummy %d.",dummy);
    }
    for (int mc = 0; mc < NUMDATA; mc++)
    {
        sample(mr,st->lambda,NUMVARS,SKIP);
        if(sampFlag == 1) { 
            // record MC samples and energies
            sts[mc]=0;
            for(int j=0; j < NUMVARS; j++) mc_final[mc * NUMVARS + j] = 0;
            for(int k=0; k < mr->sampleCountFcns; k++)
            {
                int j = mr->prevSample[k];
                if(k < mr->sampleCountVars) mc_final[mc * NUMVARS + j] = 1;
                
                // this is the energy statistics. Need to INVERT the sign and subtract the baseline energy of all patterns quiet
                sts[mc] += st->lambda[j];
            }
        // adding up statistics
        mmn += mr->sampleCountVars;
        emn += sts[mc];
        esq += sts[mc] * sts[mc];
        }
        // adding up feature expectations 
        if(mr->sampleCountFcns > 0) 
        {
            for(int j = 0; j < mr->sampleCountFcns; j++) 
            {
                ex[mr->prevSample[j]]++;
            }
        }
    }
    // output final expectations and (optionally) final sampling statistics
    for(int j = 0; j < NUMFCNS; j++)  ex[j] /= NUMDATA;
    if(sampFlag == 1) {
        esample[0] = (esq / NUMDATA) - (emn / NUMDATA) * (emn / NUMDATA);
        esample[1] = emn / NUMDATA;
        esample[2] = mmn / (NUMDATA*NUMVARS);
    }
}


double max(double a, double b)
{
    if (a > b) return a; 
    else return b;
}


double min(double a, double b)
{
    if (a < b) return a; 
    else return b;
}


void featureFunction(int* prevSample, int nonzero_vars, int* nonzero_features,int NUMVARS,char* ffunc)
{
    int a,b;
    if(!strcmp(ffunc,"KSpike")) {
        *nonzero_features = nonzero_vars + 1;
        prevSample[nonzero_vars] = NUMVARS+nonzero_vars;
    }
    else if(!strcmp(ffunc,"KSpikeIsing")) {      
        *nonzero_features = nonzero_vars + 1;
        prevSample[nonzero_vars] = NUMVARS+nonzero_vars;
        for(int i=0;i<nonzero_vars;i++) {
            for(int j=0;j<i;j++) {
                a = min(prevSample[i],prevSample[j]);
                b = max(prevSample[i],prevSample[j]);
                prevSample[*nonzero_features] = 2*NUMVARS+((b * (b-1))/2 + (a+1));
                *nonzero_features = *nonzero_features + 1;
            }
        }   
    }
    else if(!strcmp(ffunc,"Ising")) {
        *nonzero_features = nonzero_vars;
        for(int i=0;i<nonzero_vars;i++) {
            for(int j=0;j<i;j++) {
                a = min(prevSample[i],prevSample[j]);
                b = max(prevSample[i],prevSample[j]);
                prevSample[*nonzero_features] = NUMVARS-1+((b * (b-1))/2 + (a+1));
                *nonzero_features = *nonzero_features + 1;
            }
        }
    }
    else {
        printf("Undefined Feature Function:\n%15s\n",ffunc);
        exit(0);
    }
}


void diffFunction(int* prevSampleDiff,int nonzero_vars, int flip ,int NUMVARS,int* plus, int* nplus,int* minus,int* nminus ,char* ffunc)
{
    int a,b;
    if(!strcmp(ffunc,"KSpike")) {
        plus[0] = NUMVARS+nonzero_vars+1;
        minus[0] = NUMVARS+nonzero_vars;
        *nplus = 1;
        *nminus = 1;
    }
    else if(!strcmp(ffunc,"KSpikeIsing")) {
        plus[0] = NUMVARS+nonzero_vars+1;
        minus[0] = NUMVARS+nonzero_vars;
        *nplus = 1;
        *nminus = 1;
        for(int j=0;j<nonzero_vars;j++) {
            a = min(flip,prevSampleDiff[j]);
            b = max(flip,prevSampleDiff[j]);
            plus[j+1] = 2*NUMVARS+((b * (b-1))/2 + (a+1));
            (*nplus)++;
        }
    }
    else if(!strcmp(ffunc,"Ising")) {
        *nplus = 0;
        *nminus = 0;
        for(int j=0;j<nonzero_vars;j++) {
            a = min(flip,prevSampleDiff[j]);
            b = max(flip,prevSampleDiff[j]);
            plus[j+1] = NUMVARS-1+((b * (b-1))/2 + (a+1));
            (*nplus)++;
        }
    }
    else {
        printf("Undefined Feature Function:\n%15s\n",ffunc);
        exit(0);
    }
}
