function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% m is the number of training examples, n is the number of features, idx is a m by 1 vector

for i = 1 : K,
  
  exampleID = find(idx == i); % ? by 1 vector, each element represents the ID of the example which is assigned to centroid i
  
  examples = X(exampleID, :); % ? by n vector, each row represent a example that is assigned to centroid i
  
  exampleNum = size(examples, 1); % real number, the number of examples which are assigned to centroid i
  
  exampleSum = sum(examples); % 1 by n vector, the sum of all the examples which are assigned to centroid i
 
  centroids(i, :) = exampleSum / exampleNum; % update centroid i 

end




% =============================================================


end

