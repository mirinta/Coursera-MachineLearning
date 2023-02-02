function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X, 1); % real number, the number of training examples

for i = 1 : m,
  
  example = []; % initialize an empty matrix, prepare to store the ith example
  
  for j = 1 : K, 
    
    example = [example; X(i, :)]; % K by n matrix (after the inner loop), each row represents the ith example
   
  end
  
  sqrDist = sum((example - centroids) .^2, 2); % K by 1 vector, squared distances between ith example and each centroid
  
  [minValue, minIndex] = min(sqrDist); % find the smallest value and its index
  
  idx(i) = minIndex; % assign the index to the ith element of idx ( K by 1 vector)

end
 
 





% =============================================================

end

