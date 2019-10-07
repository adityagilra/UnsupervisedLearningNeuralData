% Calculates the maximum entropy //Ising model// -> firing count constrained model for given means / covariances.
% expts are means and covariances packed into a linear array by the
% unparametrizeIsing() function. Numruns is the number of monte-carlo 
% cycles; numsamples is the number of binary patters drawn in each 
% MC simulation; numlearn is the number of learning steps per cycle.
% rs is the result of previous tmaxent runs, such that, if supplied, 
% the algorithm will continue with the last cycle in the given rs. The
% rs parameter can be left out as well (empty).
%
% Typically, set numsamps comparable to the number of samples in the
% estimated dataset, up to 10 times that number (usually order 100k-1M).
% Numlearn should start high, perhaps 50, and be decreased upon subsequent
% runs, down to perhaps 5 or 10. 
%
% The return value will contain magnetic fields hs, couplings js, MC
% estimate of the means (mv) and covariances (cv), and l1 norms of
% the deviation of mv and cv (i.e. the maximally deviating correlation
% and mean). 
%
%% Note: means mean(data), and covariances cov(data), are always estimated
% on data that is {-1, 1}, NOT {0, 1}.
%% NOTE alteration: this modified routine uses {0, 1} data, NOT {-1,1}
% ****
%% Note: this routine needs mxMaxEntTGen mex-ed first! 
% ****
%
function [rs] = tmaxentGen(expts, numvars, numruns,numlearn, numsamps, skip, seed, betas,ffunc,rs)
    if (nargin < 10) rs=[]; end;
    rnds=round(rand(1,1) * 1000000);
    if (nargin < 1) cpls = []; expts = []; return; end
    if (nargin > 1) 
        if (numel(numruns) ~= 1 || numel(numvars) ~= 1 || numel(numlearn) ~= 1 || ...
            numruns < 0 || numvars < 1 || numlearn < 0) 
            disp('Wrong input parameters');
            return;
        end
        if(numel(numsamps) == 2)
            numsampsI = numsamps(1);
            numsampsF = numsamps(2);
        elseif(numel(numsamps) == 1)
            numsampsI = numsamps;
            numsampsF = numsamps;
        else
            disp('Wrong input parameters');
        end
    end
    if (max(size(expts))~=numel(expts))
        disp('Wrong expectation array dimension.');
        return;
    end
    expts  = double(reshape(expts, 1, numel(expts)));
    expts1 = expts;
    sz     = numvars;
    
    if (mod(numruns,10)~=0) disp('The number of runs must be a multiple of 10.'); return; end;
    
    numruns = int32(numruns);
    numsamps = int32(numsamps);
    numlearn = int32(numlearn);
    i0=0;
    intmem = zeros(length(expts1),1);
    if (nargin==10 & ~isempty(rs)) i0=length(rs); intmem=rs(end).cpls_raw; betas=rs(end).betas; end;
    
    if (nargin == 1) [a1 a2] = mxMaxentTGen(expts1);
    else
        for i=(i0+1):numruns/10,
            tic();
          %  if (i == 1) [a1 a2] = mxMaxentT(expts1, 10, numsamps, numlearn);
            if (isempty(betas)) 
                if (i~=1)
                    if(i==floor(numruns/10))
                        [a1 a2 esample sts sample] = mxMaxentTGen(expts1, numvars,10,numlearn,numsampsI,numsampsI,skip,seed,ffunc,intmem);
                    else
                        [a1 a2] = mxMaxentTGen(expts1, numvars,10,numlearn,numsampsI,numsampsF,skip,seed,ffunc,intmem);
                    end
                    %[a1 a2] = mxMaxentTGen(expts1, 10, numsamps, numlearn,sz, intmem);
                else
                    [a1 a2] = mxMaxentTGen(expts1, numvars,10,numlearn,numsampsI,numsampsF,skip,seed,ffunc);
                    %[a1 a2] = mxMaxentTGen(expts1, 10, numsamps, numlearn,sz);
                end
            else
                if(i==floor(numruns/10))
                    [a1 a2 esample sts sample] = mxMaxentTGen(expts1, numvars,10,numlearn,numsampsI,numsampsF,skip,seed,ffunc,intmem);
                else 
                    [a1 a2] = mxMaxentTGen(expts1, numvars,10,numlearn,numsampsI,numsampsI,skip,seed,ffunc,intmem);
                end
            end
            intmem = a1;
            
            
            hs = expts(1:sz)';
            js = expts(sz+1:end)';
            
            rs(i).time = toc();
            rs(i).cpls_raw = a1;
            rs(i).expt_raw = a2;
            [hsx jsx mvx cvx] = convert2(a1,a2,sz);
            rs(i).hs=hsx;
            rs(i).js=jsx;
            rs(i).mv=mvx;
            rs(i).cv=cvx;
            rs(i).l1_cv=max(abs(cvx-js));
            rs(i).l1_mv=max(abs(mvx-hs));
            rs(i).betas=betas;
            mv0 = expts(1:sz);
            cv0 = expts((sz+1):end);
            rs(i).mv0 = mv0;
            rs(i).cv0 = cv0;
            rs(i).file_name = sprintf('rs%d.mat',rnds);
            
            
            rs(i).num_samples = numsamps;
            rs(i).num_learn   = numlearn; % to here
            
            disp(sprintf('**** TMAXENTGEN: Iteration %d / %d   |   file rs%d.mat.\n**** TMAXENTGEN: L1 cov:  %g  L2 cov:  %g\n**** TMAXENTGEN: L1 mean: %g   L2 mean: %g\n', i, numruns/10,rnds, max(abs((cvx-js))),norm(((cvx-js))), max(abs(mvx-hs)),norm(mvx-hs)));
            save(rs(i).file_name, 'rs');
           % disp([ 'iteration ' num2str(i)]);
        end
    end
    if (nargin==10) 
        if(numruns == 0 || numlearn == 0)
            [a1 a2 esample sts sample] = mxMaxentTGen(expts1, numvars,0,numlearn,numsampsI,numsampsF,skip,seed,ffunc,rs.cpls_raw);
            
            e0 = hamilGen(ffunc, zeros(nN,1), rs.cpls_raw);
            esample = -esample - e0;
    
            rs(end).esample = esample;
            rs(end).sts = sts;
            rs(end).sample = reshape(sample,numvars,numsampsF);
        end
        
    end
end

function [hsx jsx mv cv cpls exptso] = convert2(a1, a2, sz)
    hsx = a1(1:sz);
    jsx = a1((sz+1):end);
    mv = a2(1:sz);
    cv = a2((sz+1):end);
    cpls = a1;
    exptso = a2;
end