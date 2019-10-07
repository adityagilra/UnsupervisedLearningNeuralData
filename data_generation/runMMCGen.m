% FORWARD MONTE CARLO GATEWAY ROUTINE
% nS is the number of samples, sampling_freq is how many spin flips are
% done before another sample is collected. mv1 are mean value expectations
% in {0, 1} convention, cv1 are connected covariances, smp is {0, 1} array
% of samples and e1 is the time course of the every sampled at sampling
% freq. 
% sts(1) = variance of energy, sts(2) = mean magnetization, sts(3) =
% mean energy
% cooldownF is an approximate fraction of samples to equilibrate MCMC
% before real sampling starts; enter 0 to do no equilibration 
function [mv1 cv1 esample sts sample] = runMMCGen(coupls,nN,nS,frac,seed,ffunc,cooldownF,state, suppressOut)    
    frac = round(double(frac)); coupls = double(coupls); nS = round(double(nS));
    if (nS <= 0) error('The number of samples must be positive'); end
    if (numel(coupls) ~= size(coupls,1) && numel(coupls) ~= size(coupls,2)) error('Coupls is wrong size');end
    nfeatures = numel(coupls);
    if (isempty(seed)); seed = round(rand * 1000000); end    
    if (nargin <=7 ); cooldownF = 0;end;
    if (isempty(cooldownF)); cooldownF = 0; end;
    if (nargin < 9) suppressOut = 0; else suppressOut = 1; end;
    
    if (~suppressOut) disp(sprintf('**** MXISINGMMCGEN: spins: %d, samples: %d, samp_freq: %d, burn-in frac: %g ****', nN, nS, frac, cooldownF)); end
    
    tic
    if(nargout>2)
        if (nargin >= 8)
            [i1 i2 esample sts sample] = mxMaxentTGen(zeros(1,nfeatures), int32(nN),int32(0),int32(0),int32(round(nS*cooldownF)),int32(round(nS*(1-cooldownF))),int32(frac),int32(seed),ffunc,coupls, int32(state));
        else
            [i1 i2 esample sts sample] = mxMaxentTGen(zeros(1,nfeatures), int32(nN),int32(0),int32(0),int32(round(nS*cooldownF)),int32(round(nS*(1-cooldownF))),int32(frac),int32(seed),ffunc,coupls);
        end
        sample = reshape(sample,nN,round(nS*(1-cooldownF)));
    elseif(nargout==2)
        if (nargin >= 8)
            [i1 i2] = mxMaxentTGen(zeros(1,nfeatures), int32(nN),int32(0),int32(0),int32(round(nS*cooldownF)),int32(round(nS*(1-cooldownF))),int32(frac),int32(seed),ffunc,coupls, int32(state));
        else
            [i1 i2] = mxMaxentTGen(zeros(1,nfeatures), int32(nN),int32(0),int32(0),int32(round(nS*cooldownF)),int32(round(nS*(1-cooldownF))),int32(frac),int32(seed),ffunc,coupls);
        end
    else(error('not enough args'));
    end
    ttime = toc; 
    mv1 = i2(1:nN);
    cv1 = i2((nN+1):end);
        
    % correct the energy estimate
    
    e0 = hamilGen(ffunc, zeros(nN,1), reshape(coupls, numel(coupls),1));
    sts = -sts - e0;
    
    if (~suppressOut) disp(sprintf('**** MXISINGMMCGEN: spins: %d, samples: %d, DONE in %ds. Mean energy %g, variance %g, magnetization %g. ****', nN, nS, round(ttime), esample(3), esample(1), esample(2)));end
    %mv1=2*myh1'-1;
    %cv1=zeros(nN,nN);ixx=4;for i=1:nN, for j=i+1:nN, cv1(i,j)=mxy1(ixx);ixx=ixx+4;end;end;
    %cv1=cv1+cv1';
    %for i=1:nN, cv1(i,i)=myh1(i)-myh1(i)^2;end;
    %cv1=4 * cv1 - repmat(mv1,nN,1) - repmat(mv1',1,nN) - 1;
    
    % compute the connected correlations
    %cv1 = cv1 - mv1' * mv1;
