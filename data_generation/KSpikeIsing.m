function Features = KSpikeIsing(data,string)
% If data is a Nx1 vector, computes the value of each of the K-spike
% features, K=0,...,N  plus each of the two point correlations, in a [N(N+1)/2+1] x 1 vector. If data is a NxT vector, computes the
% feature values at each column, in a [N(N+1)/2+1] x T vector
% accepts {-1,1} or {0,1} data
if min(min(data)) == -1
    data = (data+1)/2;
end
[N T] = size(data);
SpikeCounts = sum(data,1);
if(nargin == 2 && isequal(string,'mean'))
    Features = zeros(N*(N+1)/2+1,1);
    for k = 1:T;
        q = find(data(:,k));
        for i=1:numel(q)
            for j=1:(i-1)
                Features(N+1+ (q(i)-1)*(q(i)-2)/2 + q(j)) = Features(N+1+ (q(i)-1)*(q(i)-2)/2 + q(j))+1;
            end
        end
    end
    for k = 1:T
        Features(SpikeCounts(k)+1) = Features(SpikeCounts(k)+1)+1;
    end
    Features = Features ./ T;
else
    Features = zeros(N*(N+1)/2+1,T);
    for k = 1:T;
        q = find(data(:,k));
        for i=1:numel(q)
            for j=1:(i-1)
                Features(N+1+ (q(i)-1)*(q(i)-2)/2 + q(j),k) = 1;
            end
        end
    end
    for k = 1:T
        Features(SpikeCounts(k)+1,k) = 1;
    end
   % find(Features)
end

% change this it won't work, c2 needs to be an individual count of when
% i=j=1, not an average : c2 = lowtrig(cov(data'));
