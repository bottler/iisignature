%Simple matlab function to match iisignature.sig .
%Returns the signature up to level m of path in R^d 
%given by n points. The path is given as an array
%of shape [n,d].
%Doesn't return level 0, which is always the number 1.
function [sig] = sig(path,m)
    [n,d]=size(path);
    o = {zeros(1,d)};
    for i=2:m
        o{i}=zeros(1,d^i);
    end
    diffs=diff(path,1,1);   
    for i=1:(n-1)
        d=diffs(i,:);
        stepsig={d};
        last=d;
        for j=2:m
            last=kron(d,last)/j;
            stepsig{j}=last;
        end
        for lev=m:-1:1
            o{lev}=o{lev}+stepsig{lev};
            for llev=1:(lev-1)
                o{lev}=o{lev}+kron(o{lev-llev},stepsig{llev});
            end
        end
    end
    %sig=o; %To return a cell array, with the levels separately
    sig=cat(2,o{:}); %To return a single array
end
