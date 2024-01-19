function [z, p] = hmmViterbi_DT(M, A, s)
% Implmentation function of Viterbi algorithm. 
% Input:
%   M: k x n emmision data matrix M=E*X
%   A: k x k transition matrix
%   s: k x 1 starting probability (prior)
% Output:
%   z: 1 x n latent state
%   p: 1 x n probability
% Written by Mo Chen (sth4nth@gmail.com).
% I modified it to fit that the transitional probability is dynamic (k*k*n)
[k,n] = size(M);
Z = zeros(k,n);
A = log(A);
M = log(M);
Z(:,1) = 1:k;
v = log(s(:))+M(:,1);
for t = 2:n
    [v,idx] = max(bsxfun(@plus,A(:,:,t),v),[],1);    % 13.68
    v = v(:)+M(:,t);
    Z = Z(idx,:);
    Z(:,t) = 1:k;
end
[v,idx] = max(v);
z = Z(idx,:);
p = exp(v);
