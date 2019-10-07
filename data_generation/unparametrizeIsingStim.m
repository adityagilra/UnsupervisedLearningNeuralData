function [cpls]=unparametrizeIsingStim(hs,js, ss)
    n=length(hs);
    [i1 i2]=size(ss);
    cpls=zeros(n*(n+1)/2 + n * i2,1);
    cpls(1:n)=hs;
    ixx=n+1;
    
        for i=1:n,
            for j=i+1:n,
                cpls(ixx)=js(i,j);
                ixx=ixx+1;
            end;
            for j=1:i2,
                cpls(ixx)=ss(i,j);
                ixx=ixx+1;
            end
        end
end