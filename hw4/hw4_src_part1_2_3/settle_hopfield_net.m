function [ data] = settle_hopfield_net( w, data, iters)
% [data] = settle_hopfield_net(w, data, iters) settles a hopfield network
%  with initial configurations data, weight matrix w, and the number of
%  iterations iters.
%  
% INPUTS:
%   w ............. : an (v by v) weight matrix where v is the number of
%                       visible units
%   data .......... : an (n by v) data matrix where n is the number of
%                       datapoints and v is the number of visible units
%   iters ......... : number of iterations to run the settling algorithm
%
% OUTPUTS:
%   data .......... : an (n by v) matrix which is the network
%                       configurations after iters iterations of the 
%                       settling algorithm

% The data comes in as 0s and 1s. Convert to -1s and 1s
data = convert_0_to_neg1(data);


[n,p] = size(data);

for i=1:iters
    for d=1:n
        node = randi([1,p], 1,1);
  
        % Update state of node
        Unode = data(d,:)*w(:,node);
        data(d,node) = sign(Unode);

        % Check edge case when sum is zero
        if(data(d,node) == 0)
            data(d,node) = 1;
        end
    end
end

end

