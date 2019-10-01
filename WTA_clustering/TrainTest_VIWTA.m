function [Output] = TrainTest_VIWTA(varargin)
%__________________________________________________________________________
% Description: Performs training and testing of the VI-WTA SNN.
%              This is simply a Matlab wrapper that calls the mexed
%              function VIWTA_SNN (written in C++).
%__________________________________________________________________________

%% -- Specify Hyper-Parameter Values: --
V.dataset  = 'daily';               
V.m        = 10;         %# of readout neurons in VIWTA circuit (i.e. # of latent modes)
V.eta_b    = 0.001;      %Learning rate hyperparameter 
V.eta_W    = 0.0004;     %Learning rate hyperparameter
V.save     = 1;          %0 = do NOT save
V          = parseargs(V, varargin{:});

if strcmp(V.dataset,'minutely')
   savename = ['VIWTA_minutely_etab' strrep(num2str(V.eta_b),'.','pt') ... 
               '_etaW' strrep(num2str(V.eta_W),'.','pt') '.txt'];
elseif strcmp(V.dataset,'daily')
   savename = ['VIWTA_hourly_etab' strrep(num2str(V.eta_b),'.','pt') ... 
               '_etaW' strrep(num2str(V.eta_W),'.','pt') '.txt'];
end 

mixing_weights = (1/V.m)*ones(1,V.m);

%% -- Train & Test via VI (Variational Inference) WTA Circuit: --
% Call Mex Code:
[W_star, b_star, Converg_avgW, readout_train, readout_test] = ...
      VIWTA_SNN(V.eta_b, V.eta_W, mixing_weights);

Output.W_star        = W_star;
Output.b_star        = b_star;
Output.Converg_avgW  = Converg_avgW;
Output.readout_train = readout_train;
Output.readout_test  = readout_test;  
 
if V.save
    csvwrite(savename,readout_test'); 
    save(strrep(savename,'.txt','.mat'),'Output'); 
end

end %main
