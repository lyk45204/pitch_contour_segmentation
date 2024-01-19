function [currentState, p] = hmmViterbi(M, A, s)
% Implmentation function of Viterbi algorithm. 
% Input:
%   M: k x n emmision data matrix M=E*X
%   A: k x k transition matrix
%   s: k x 1 starting probability (prior)
% Output:
%   z: 1 x n latent state
%   p: 1 x n probability
% Modify Mo Chen's hmmViterbi_ according to Matlab's hmmviterbi to speed up code(sth4nth@gmail.com).
[k,n] = size(M);
% allocate space
pTR = zeros(k,n);
A = log(A);
M = log(M);
pTR(:,1) = 1:k;
v = log(s(:))+M(:,1);

for t = 2:n

    % use for loop to avoid calling max a lot of times 
    v_max = zeros(k,1);
    for state = 1:k
        bestVal = -inf;
        bestPTR = 1;
        for inner = 1:k
            val = A(inner,state) + v(inner);
            if val > bestVal
                bestVal = val;
                bestPTR = inner;
            end
        end
        v_max(state) = bestVal;
        % save the best transition information for later backtracking
        pTR(state,t) = bestPTR;
    end

    v = v_max;
    v = v(:)+M(:,t);
end

[v_best,idx_best] = max(v);


% Now back trace through the model
% z = Z(idx,:);
% p = exp(v);
currentState = zeros(1,n);
currentState(n) = idx_best;
for count = n-1:-1:1
    currentState(count) = pTR(currentState(count+1),count+1);
    if currentState(count) == 0
        error(message('stats:hmmviterbi:ZeroTransitionProbability', currentState( count + 1 )));
    end
end
