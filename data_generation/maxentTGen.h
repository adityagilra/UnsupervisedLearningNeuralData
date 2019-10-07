/**
 * A header file for sampler algorithms.
 */
 
#ifndef MAXENTT_H
#define MAXENTT_H

/* statistics of the MCMC sampling run */
struct montestats
{
    double* sum_fjexpg_monte_eta; // numerator of eq 19 in paper, <fj exp(theta-theta')>
    double Z_monte_eta; // deviation normalization constant for the histogram Monte Carlo (updated as learning steps proceed from a given MC sampling run, reset at next run)
    double Z_monte_lambda; // sampling estimate of partition function for the current values of the coupling consts = sum_samples exp (g_monte_lambda[sample])
    double* g_monte_eta;  // accumulated deviation in energy g_monte_lambda for each MCMC sample
    double* g_monte_lambda; // energy, i.e. sum of features times coupling constants sum_{i=features} lambda_i f_i for each MCMC sample.  
    double* eta; // accumulated deviations to coupling constants lambda
    double* lambda; // current coupling constants for maxent features
    double* expec_scaled; // empirical expectations to be matched by algorithm

    int* count_monte; // array of NUMFCNS counters keeping track of number of nonzero values for each feature; keeps track of the sizes of sInds_monte[][]
    int* bound_monte; // array of NUMFCNS allocated memory sizes for nonzero monte carlo arrays sInds_monte[][]
    int** sInds_monte; // array of NUMFCNS arrays of sample numbers for which features are nonzero in the MCMC samples.  Dynamically expanded as needed.
    
    double sum_gexpg_monte_eta; // diagnostic stats to check that learning is proceeding properly
    double sum_energyexpg_monte_eta; // diagnostic stats to check that learning is proceeding properly
    double rel_entropy_monte_theta_lambda; // entropy statistic 
    double mean_energy_monte_eta; // energy statistic
};

/* state of the current MCMC sample */
struct monteruns
{
    int* prevSample; // array containing position numbers of the nonzero features (i.e. some subset of 0 to NUMFCNS-1)
    int* prevSampleDiff;
    int* plus; //array for differential feature function, indicates which features are turned on when a spin is flipped
    int* minus; //arry for differential feature function, indicates which features are turned off when a spin is flipped
    int sampleCountFcns; // number of total nonzero features
    int sampleCountVars; // number of nonzero coordinate features
    int seed; // seed for sampler
    char* ffunc; //name of feature function
};

void setupMonteSt(montestats* st, int NUMFCNS, int NUMVARS, int NUMSAMPS); //allocate memory for sampling statistics and initialize
void setupSample(montestats* st, monteruns* mr, int NUMFCNS, int SEED); //initialize sampling state
double sample(monteruns* mr, double* lambda, int NUMVARS, int SKIP); // take an MC sample
int processSamples(montestats* st, monteruns* mr, int mc); //update sampling run statistics and allocated memory after each sample
void learn_alg(montestats* st, int NUMFCNS, int NUMSAMPS); //run learning step for algorithm (Schapire)
void initialGuess(montestats* st, monteruns* mr, int NUMVARS, int NUMFCNS, int NUMDATA, double* initialguess, int SKIP); // make an initial guess of the coupling constants
void finalGuess(montestats* st, monteruns* mr, int NUMFCNS, int NUMDATA, int NUMVARS, double* ex, double* sts, double* esample, double* sample, int SKIP, int sampFlag, int SEED, int* init_guess);
double max(double a, double b);
double min(double a, double b);
void featureFunction(int* prevSample, int nonzero_vars, int* nonzero_features, int NUMVARS, char* ffunc); // generic feature function
void cleanupMonteSt(montestats* st,monteruns* mr, int NUMFCNS);
void diffFunction(int* prevSampleDiff,int nonzero_vars, int flip ,int NUMVARS,int* plus, int* nplus,int* minus,int* nminus ,char* ffunc);
#endif

