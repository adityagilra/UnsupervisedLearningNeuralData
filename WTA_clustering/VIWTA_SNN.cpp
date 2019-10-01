//_________________________________________________________________
//  VIWTA_SNN.cpp
//  Performs both training *and* testing in one package of
//  VI-WTA (variational inference winner-take-all circuit)
//  spiking network model that implements
//  unsupervised learning of its implicit generative model.
//  This version allows learning of the mixing weights p_k = exp(\pi_k),
//  i.e. it does *NOT* assume uniform priors on the mixing weights.
//
// NOTE: This version is fully online, in that it outputs the
//       readout responses of the VI (variational inference)
//       WTA circuit readout neurons during *training*.
//       (This code also outputs the responses during
//       testing, i.e. frozen using the learned parameters).
// NOTE: This version is for local use with Matlab (mex file).
// # of INPUTS:  3
// # of OUTPUTS: 5
//
//  Copyright © 2019 adrianna. All rights reserved.
//  NOTE: This version has been modified to output
//        latent mode (i.e. readout neuron spike) probabilities,
//        to be used as input to LSTM (for Xenesis).
//_________________________________________________________________

#include "VIWTA_SNN.h"

#include "matrix.h"
#include "mex.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <exception>
#include <random>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

template <typename T>
void writeOutputMatrix(int pos, vector<T> value, int N, int M, mxArray**& plhs) {
    mxArray* out_matrix = mxCreateDoubleMatrix(N,M,mxREAL);
    double* pr = mxGetPr(out_matrix);
    for (typename vector<T>::iterator it=value.begin(); it!=value.end(); ++it) {
        *pr++ = (double) *it;
    }
    plhs[pos] = out_matrix;
    return;
}

// *****************************************************************
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    // ** Command-Line Argument & Initializations: **
    int N            = 250;                //Afferent neuron population size
    int m            = 10;                 //# of latent states (# readout neurons)
    double binsize   = 200;                //Time bin width (transformed time units: 10 kHz)
    double eta_b     = *mxGetPr(prhs[0]);  //Learning rate for b_k values (hyperparam)
    double eta_W     = *mxGetPr(prhs[1]);  //Learning rate for W_ki values (hyperparam)
    double* mexArray = mxGetPr(prhs[2]);   //Pointer to Matlab vector of mixing weight values
    
    string datafile = "ST_VIM_GBPUSD_Daily.txt"; //simulated or actual (data) spike times
    
    // ** VI-WTA Spiking Circuit Model **
    cout << "Initializing VI-WTA object..." << endl;
    WTACircuitModel WTA_obj(datafile, binsize, N, m, eta_b, eta_W, mexArray); //instantiate
    
    // ** Train on ALL Observed Data & Return Learned Model Params: **
    paramsStruct<double> learned_W_b = WTA_obj.train_via_STDP(binsize);
    cout << "Finished training via STDP update protocol." << endl;
    
    // ** Return Learned Parameters for Matlab Format: **
    vector<double>& learned_W = *learned_W_b.W_star.data();
    writeOutputMatrix(0, learned_W, m, N, plhs);
    writeOutputMatrix(1, learned_W_b.b_star, m, 1, plhs);
    int n_timebins = WTA_obj.return_ntimebins();
    writeOutputMatrix(2, learned_W_b.Converg_avgW, n_timebins, 1, plhs);
    cout << "Finished writing learned parameters..." << endl;
    
    // ** Return the Readout Responses during *Training* (in Matlab Format): **
    vector<double>& readout_training = *learned_W_b.readout.data();
    writeOutputMatrix(3, readout_training, m, n_timebins, plhs);
    
    // ** Now Test the Trained Network & Return Readout Neuron Responses: **
    cout << "Beginning testing on data using learned parameters..." << endl;
    myMatrix<double> output_test = WTA_obj.test_WTA(binsize);

    // ** Return Outputs for Matlab Format: **
    vector<double>& readout_test = *output_test.data();
    writeOutputMatrix(4, readout_test, m, n_timebins, plhs);
    cout << "Finished writing test readout response outputs..." << endl;
}
// *****************************************************************

// ************************** RNG Methods **************************
//Constructor:
RNG::RNG() {
    // Initialize mersenne twister RNG
    rng_pr = gsl_rng_alloc(gsl_rng_mt19937);
}
//Destructor:
RNG::~RNG() {
    gsl_rng_free(rng_pr);
}
//RNG::uniform
double RNG::uniform(double min, double max) {
    double u = gsl_rng_uniform(rng_pr);
    return ((max-min)*u)+min;
}
//RNG::discrete
int RNG::discrete(const vector<double>& p) {
    double u = gsl_rng_uniform(rng_pr);
    double c = p[0];
    int ix=0;
    while (c<u) {
        ix++;
        c += p[ix];
    }
    return ix;
}
//RNG::bernoulli
bool RNG::bernoulli(double p) {
    return (gsl_rng_uniform(rng_pr) < p);
}
//RNG::randperm
vector<int> RNG::randperm(int nmax) {
    //Description: This fn is analogous to Matlab's randperm(nmax)
    vector<int> nvals (nmax);
    for (int i=0; i<nmax; i++) {
        nvals[i] = i;
    }
    for (int i=0; i<nmax; i++) {
        // select random integer ix between i and nmax-1
        // swap i with ix
        unsigned long ix = i + gsl_rng_uniform_int(rng_pr, nmax-i);
        int tmp   = nvals[i];
        nvals[i]  = nvals[ix];
        nvals[ix] = tmp;
    }
    return nvals;
}
//RNG:gaussian
double RNG::gaussian(double sigma, double mu) {
    double g = gsl_ran_gaussian(rng_pr, sigma);
    return g+mu;
}
// *****************************************************************

// ******************* WTACircuitModel Methods *********************
//Constructor:
WTACircuitModel::WTACircuitModel(const string& filename, double binsize, int N, int m, double eta_b, double eta_W, double* mexArray) : N(N), m(m), eta_b(eta_b), eta_W(eta_W)
{
    rng = new RNG();
    
    //-- Instantiate constant parameters: --
    mu_w    = -21.3;
    sigma_w = 2;
    r_net   = 1;      //Units: spikes per *time bin*
    c       = 1;
    
    //-- Load the population spike time data from the input filename: --
    cout << "Loading in spike time data..." << endl;
    vector<Spike> all_spikes; //Init cache
    ifstream infile;
    infile.open(filename);
    double st;                //Denotes current spike time (10 kHz)
    int nidx;
    
    while( infile >> st >> nidx ) {
        Spike s;
        s.time = st;  //Units: 10 kHz
        s.bin  = floor(s.time/binsize);
        s.neuron_ind = nidx;
        all_spikes.push_back(s);
    }
    infile.close();
    
    //-- Now sort spikes to be in chronological order: --
    all_spiketimes = sort_spikes(all_spikes);
    cout << "Sorted spikes..." << endl;       //all_spiketimes is protected data of WTA_obj
    
    //-- Initialize W matrix \in R^{m \times N}: --
    vector<double> W_int_vec; //prelim
    //const gsl_rng_type* T;  //can alternatively use
    //gsl_rng* r;
    //gsl_rng_env_setup();
    //T = gsl_rng_default;
    //r = gsl_rng_alloc(T);
    for (int ind=0; ind<(m*N); ind++) {
        double u = 0.1*((double) rand() / (double) RAND_MAX) + 0.45; //initial \pi_{ki}
        W_int_vec.push_back( log(u/(1-u)) );
        //W_int_vec.push_back(rng->gaussian(sigma_w, mu_w));
        //W_int_vec.push_back(mu_w + gsl_ran_gaussian(r, sigma_w));
        deltaW.push_back(0);
    }
    W.assign(W_int_vec,m,N);
    
    //-- Load in or initialize m_vec values (constraints): --
    vector<double> v_pre;
    for (int k=0; k<m; k++) {
        v_pre.push_back(mexArray[k]);
    }
    m_vec = v_pre;
    
    //-- Initialize intrinsic excitabilities, {b_k}: --
    for (int k=0; k<m; k++) {
        double A_k = compute_A_Wk(k);
        double m_k = m_vec[k];
        b.push_back(-A_k + log(m_k));
    }
}

//WTACircuitModel Destructor:
WTACircuitModel::~WTACircuitModel() {
    delete rng;
}

//WTACircuitModel::sort_spikes
//Sorts input spike times in ascending chronological order (uses <algorithm> header)
vector<Spike> WTACircuitModel::sort_spikes(vector<Spike>& a_spikes) { //use call-by-reference to modify input vec
    sort(a_spikes.begin(), a_spikes.end(), SpikeComparison());
    return a_spikes;
}

//WTACircuitModel::compute_A_Wk
double WTACircuitModel::compute_A_Wk(int k) {
    double A_Wk = 0; //Init
    for (int i=0; i<N; i++) {
        A_Wk += log(1 + exp(W.at(k,i)));
    }
    return c*A_Wk;
}

//WTACircuitModel::train_via_STDP
paramsStruct<double> WTACircuitModel::train_via_STDP(double binsize) {
    paramsStruct<double> params_learned;
    Spike last_s  = all_spiketimes.back();
    int first_bin = all_spiketimes[0].bin;
    int n_T       = last_s.bin;
    n_timebins    = (n_T-first_bin+1);              //total # of discrete time bins
    vector<double> Converg_avgW(n_T+1 - first_bin);
    
    myMatrix<double> rho_Cache;                     //Initialize cache of outputs during *training*
    vector<double> rho_vec;
    for (int ind=0; ind<(m*n_timebins); ind++) {
        rho_vec.push_back(0);
    }
    rho_Cache.assign(rho_vec,m,n_timebins);
    
    for (int n_t=first_bin; n_t<=n_T; n_t++) {      //n_t denotes the current time bin index
        //--
        //std::vector<bool> y(m,0);                 //may later expand scope
        
        //(0a) Compute x(n_t) \in \R^N for the current time bin n_t:
        compute_unweighted_spkvec(n_t, binsize);    //updates vector<double> current_x \in \R^N of WTA_obj
        
        //(0b) Compute i(n_t) (which is independent of k)
        compute_current_inhibition();
        
        //double check_homog_netrate = 0; //check that inhibition is implementing homogeneous network rate
        //if (abs(check_homog_netrate-r_net)>0.001) { cerr << "Non-homogeneous network rate" << endl; }
        
        //** Apply STDP Updates: **//
        //Stochastic E-Step Approximation:
        for (int k=0; k<m; k++) {
            // -- Homeostatic Plasticity Step: --
            double rho_k = compute_rho_k(k);
            double delta_bk = eta_b * ( m_vec[k] - rho_k);
            b[k] += delta_bk;
        } //end over k \in [m]
        compute_current_inhibition();         //update variational posterior w/ new b_k terms
        
        //Stochastic M-Step Approximation:
        for (int k=0; k<m; k++) {
            double rho_kt = compute_rho_k(k); //uses updated b_k terms (variational posterior)
            rho_Cache.assign_entry(k, n_t-first_bin, rho_kt);
            
            // -- Update synaptic weights W_ki \forall i \in [N]: --
            for (int i=0; i<N; i++) {
                    double deltaW_ki = eta_W * rho_kt * (current_x[i] - (1/(1+exp(-W.at(k,i)))));
                    deltaW[i*m + k] = deltaW_ki;
                    W.addto(k, i, deltaW_ki);
                    //if (W.at(k,i)<0) { cerr << "Negative W_ki obtained for readout neuron " << k << endl; }
                } //end over i \in [N]
            } //end over k \in [m]
        
        //** To Check Convergence, Compute Mean of \delta W_ik: **
        Converg_avgW[n_t - first_bin] = calcAvg_deltaW();
        
        cout << "Finished n_t = " << n_t << endl;
        //--
    } //end over time bins n_t
    
    //Assign learned parameters
    params_learned.W_star = W;
    params_learned.b_star = b;
    params_learned.Converg_avgW = Converg_avgW;
    params_learned.readout = rho_Cache;
    return params_learned;
}

//WTACircuitModel::compute_unweighted_spkvec
void WTACircuitModel::compute_unweighted_spkvec(int n_t, double binsize) {
    //Initializations:
    double t = binsize*(n_t+1); //will use as proxy for discretizing time
    int spk_ind = 0;
    std::vector<double> x(N,0.0);
    Spike last_spk = all_spiketimes.back();
    int n_T        = last_spk.bin;
    
    Spike s = all_spiketimes[spk_ind];
    if (n_t < n_T) {
        while (s.bin <= n_t) {
            x[s.neuron_ind] += EPSP_kernel(t-s.time, binsize);
            spk_ind++;
            s = all_spiketimes[spk_ind]; //move to next afferent spike
        }
    }
    else { //This is if assuming training on ALL of the data
        vector<Spike>::iterator it;
        for (it = all_spiketimes.begin(); it != all_spiketimes.end(); ++it) {
            x[it->neuron_ind] += EPSP_kernel(t - it->time, binsize);
        }
    }
    
    current_x = x; //update for time bin n_t
}

//WTACircuitModel::EPSP_kernel //simplest step function version
double WTACircuitModel::EPSP_kernel(double delta, double binsize) {
    double y;
    if (delta <= binsize) { y=1; }
    else {y=0;}
    return y;
}

//WTACircuitModel::EPSP_kernel //Biologically-plausible version (double exponential PSP kernel)
//double WTACircuitModel::EPSP_kernel(double delta) {
//    double y = D*(exp(-delta/tau1) - exp(-delta/tau2));
//    if (y<0) { cerr << "Error: Negative value obtained for non-negative EPSP function.\n"; }
//    return y;
//}

//WTACircuitModel::compute_current_inhibition
void WTACircuitModel::compute_current_inhibition() {
    double temp_summand = 0; //init
    for (int j=0; j<m; j++) {
        temp_summand += exp( compute_uk_hat(j) );
    }
    if (temp_summand < 0) { cerr << "Error: Negative values for log argument.\n"; }
    current_inhib = log(temp_summand) - log(r_net); //update global lateral feedback inhibition
}

//WTACircuitModel::compute_uk_hat
double WTACircuitModel::compute_uk_hat(int k) {
    double dotprod = 0; //=dot product of W_k and x
    //Compute dot product
    for (int i=0; i<N; i++) {
        dotprod += ( W.at(k,i) * current_x[i] );
    }
    return (dotprod + b[k]); //+b[k] for variational approximation (homeostatic plasticity)
}

//WTACircuitModel::compute_rho_k
double WTACircuitModel::compute_rho_k(int k) {
    double rho_k = exp( compute_uk_hat(k) - current_inhib );
    return rho_k;
}

//WTACircuitModel::calcAvg_deltaW
double WTACircuitModel::calcAvg_deltaW() {
    double sum = 0.0;
    for(std::size_t i = 0; i < deltaW.size(); i++)
        sum += deltaW.at(i);
    return sum/deltaW.size();
}

//WTACircuitModel::test_WTA (modifed to output readout spike probabilities, to be used potentially for input to e.g. LSTM)
myMatrix<double> WTACircuitModel::test_WTA(double binsize) {
    Spike last_s  = all_spiketimes.back();
    int first_bin = all_spiketimes[0].bin;
    int n_T       = last_s.bin;
    n_timebins    = (n_T-first_bin+1);             //Total # of discrete time bins
    cout << "First bin = " << first_bin << " and n_T = " << n_T << endl; //check

    myMatrix<double> rho_Cache;                    //Initialize
    vector<double> rho_vec;
    for (int ind=0; ind<(m*n_timebins); ind++) {
        rho_vec.push_back(0);
    }
    rho_Cache.assign(rho_vec,m,n_timebins);
    
    for (int n_t=first_bin; n_t<=n_T; n_t++) {    //n_t denotes the current time bin index
        //--
        //(0a) Compute x(n_t) \in \R^N for the current time bin n_t:
        compute_unweighted_spkvec(n_t, binsize); //updates vector<double> current_x \in \R^N of WTA_obj
        
        //(0b) Compute i(n_t) (which is independent of k)
        compute_current_inhibition();
        
        for (int k=0; k<m; k++) {
            //(1) Compute \rho_k(n_t)
            double rho_kt = compute_rho_k(k);
            rho_Cache.assign_entry(k, n_t-first_bin, rho_kt);
        } //end over k \in [m] (should now have fully updated vector y(n_t) \in \F_2^N)
        
        cout << "Finished n_t = " << n_t << endl;
        //--
    }
    return rho_Cache;
}
// *****************************************************************
