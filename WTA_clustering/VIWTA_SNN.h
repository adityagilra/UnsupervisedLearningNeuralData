//_________________________________________________________________
//  VIWTA_SNN.h
//  Copyright © 2019 adrianna. All rights reserved.
//_________________________________________________________________

#ifndef VIWTA_SNN_h
#define VIWTA_SNN_h

#include <iostream>
#include <gsl/gsl_rng.h>
#include <vector>
#include <string>
#include <map>

using namespace std;

// *********************** myMatrix ****************************
template <class T>
class myMatrix
{
public:
    myMatrix();
    myMatrix(vector<T>&, int, int);
    void assign(vector<T>&, int, int);
    int get_m();
    int get_N();
    
    const T& at(const int k, const int i) const;       //Subscripted index
    const T& at(const int k) const;                    //Linear index
    void addto(const int k, const int i, T a);         //Add a to the (k,i)-th entry
    void assign_entry(const int k, const int i, T a);  //Assign value a to the (k,i)-th entry
    
    vector<T>* data();
    
private:
    vector<T> matrix_data;
    int m,N; // m = #rows, N = #columns
};

// -- myMatrix Definition: --
template <class T>
myMatrix<T>::myMatrix() : m(0),N(0) {}; //Constructor def

template <class T>
myMatrix<T>::myMatrix(vector<T>& _data, int _m, int _N) : m(_m), N(_N) {
    if (_data.size() != _m*_N) {
        cerr << "Matrix dimensions must agree." << endl;
        m = 0; N = 0;
    } else {
        matrix_data = _data;
    }
}

template <class T>
void myMatrix<T>::assign(vector<T>& _data, int _m, int _N) {
    if (_data.size() != _m*_N) {
        cerr << "Matrix dimensions must agree." << endl;
        m = 0; N = 0;
    } else {
        m = _m; N = _N;
        matrix_data = _data;
    }
    return;
}

template <class T>
int myMatrix<T>::get_m() { return m; }

template <class T>
int myMatrix<T>::get_N() { return N; }

template <class T>
const T& myMatrix<T>::at(const int k, const int i) const {
    return matrix_data[i*m + k];
}

template <class T>
void myMatrix<T>::addto(const int k, const int i, T a) {
    matrix_data[i*m + k] += a;
}

template <class T>
void myMatrix<T>::assign_entry(const int k, const int i, T a) {
    matrix_data[i*m + k] = a;
}

template <class T>
vector<T>* myMatrix<T>::data() { return &matrix_data; }
// *************************************************************

// ********************* paramsStruct **************************
template<typename T>
struct paramsStruct
{
    myMatrix<T> W_star;     //the learned feedforward weights
    vector<T> b_star;       //the learned b_k* values
    vector<T> Converg_avgW;
    vector<T> Converg_avgb;
    myMatrix<T> readout;    //cache of the readout during *training*
};
// *************************************************************

// ************************** RNG ******************************
class RNG
{
public:
    RNG();
    ~RNG();
    double uniform(double min, double max);
    int discrete(const vector<double>&);
    bool bernoulli(double);
    vector<int> randperm(int);
    double gaussian(double sigma, double mu);
private:
    gsl_rng* rng_pr;
};
// *************************************************************

// ************************* Spike *****************************
struct Spike
{
    double time; 
    int bin;
    int neuron_ind;
};
// *************************************************************

// ********************* SpikeComparison ***********************
class SpikeComparison
{
public:
    bool operator() (const Spike& lhs, const Spike& rhs) const
    {
        return (lhs.bin < rhs.bin);
    }
};
// *************************************************************

// ********************* WTACircuitModel ***********************
class WTACircuitModel
{
public:
    WTACircuitModel(const string& filename, double binsize, int N, int m, double _etab, double _etaW, double* mx);
    paramsStruct<double> train_via_STDP(double binsize);
    myMatrix<double> test_WTA(double binsize);
    int return_ntimebins() { return n_timebins; };
    ~WTACircuitModel();
protected:
    //Constant params:
    int N;          //afferent input neuron population size
    int m;          //number of readout neurons (optimal # latent states)
    double mu_w;    //mean for initializing synaptic weights
    double sigma_w; //std for initializing synaptic weights
    double c;       //constant for weight update rule (see Eq. 5 in Nessler et al., 2013; need c>1)
    double eta_b;   //learning rate for b_k terms
    double eta_W;   //learning rate  for W_ki terms
    double r_net;   //total network instantaneous firing rate (constant)
    RNG* rng;
    vector<double> m_vec; //a priori specified mean activations
    
    //Dynamic/learned params:
    myMatrix<double> W;
    vector<double> b;             //b_k = -A(W_k) + \hat{b_k} + \beta_k for variational posterior
    double current_inhib;         //current common lateral inhibition (I(n_t))
    
    //Data-handling:
    vector<Spike> all_spiketimes; //will cache all spike times (in ascending order)
    vector<double> current_x;
    vector<Spike> sort_spikes(vector<Spike>&);
    int n_timebins;
    
    //Functions to compute A(W_k):
    vector<double> A_vec;
    double compute_A_Wk(int k);
    
    //To check convergence of learning:
    vector<double> deltaW;        //will cache for each i and/or k (will use to take mean)
    vector<double> deltab;
    
    //Training via STDP:
    double EPSP_kernel(double delta, double binsize);
    void   compute_unweighted_spkvec(int n_t, double binsize);
    void   compute_current_inhibition();
    double compute_uk_hat(int k);
    double compute_rho_k(int k);
    double calcAvg_deltaW();      //for convergence tracking
};
// *************************************************************

#endif /* VIWTA_SNN_h */
