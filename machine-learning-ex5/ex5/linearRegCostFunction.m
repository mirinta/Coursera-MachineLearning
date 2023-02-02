function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X is a m by n+1 matrix, y is a m by 1 vector, theta is a n+1 by 1 vector

% ====================== Compute J ======================

predictions = X * theta; % m by 1 vector, predictions of the hypothesis over all training examples

errors = predictions - y; % m by 1 vector, difference between predictions and desired outputs

sqrErrors = errors .^ 2; % squared error

costNoReg = sum(sqrErrors) / (2 * m); % real number, value of the original cost function (without regularization term)

costReg = sum(theta(2:end) .^ 2) * lambda / (2 * m); % real number, value of the regularization terminal

J = costNoReg + costReg; % real number

% ====================== Compute grad ======================

partialNoReg = X' * errors / m; % n+1 by 1 vector, partial derivatives of the original cost function (without regularization term)

partialReg = [0; theta(2:end)] * lambda / m; % n+1 by 1 vector, partial derivatives of the regularization term (not regularize theta0)

grad = partialNoReg + partialReg; % real number




% =========================================================================

grad = grad(:);

end
