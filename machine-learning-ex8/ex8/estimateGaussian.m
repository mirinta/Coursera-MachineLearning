function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

 

mu = mean(X)'; % n by 1 vector, the result of mean(...) is a 1 by n vector (summation of all rows)

sigma2 = var(X)' * (m-1) / m; % n by 1 vector, the result of var(...) is a 1 by n vector divided by m-1


% using a for-loop to compute mu and sigma2: 

%mu = (sum(X))' / m; % n by 1 vector, the result of sum(A) is a 1 by n vector (summation of all rows)

%muMatrix = []; % initialize a matrix to store the mean values

%for i = 1 : m, 
  
%  muMatrix = [muMatrix ; mu'];

%end % after the loop, temp is a m by n matrix, ith column corresponds to the mean value of ith feature

%sigma2 = (sum((X - muMatrix) .^2))' / m; % n y 1 vector, the sum(...) is a 1 by n vector (summation of all rows)



% =============================================================


end
