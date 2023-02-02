function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% ===============compute J ===============

predictions = sigmoid(X * theta); % m by 1 vector, predictions over all training examples

costs = - y .* log(predictions) - (1-y) .* log(1-predictions); % m by 1 vector

regularization = sum(theta(2:end) .^2) * lambda / (2 * m); % real number, value of the regularization term

J = sum(costs) / m + regularization;

% ===============compute grad ===============

errors = predictions - y; % m by 1 vector, defference between predictions and y

partialsWithoutRegularizationTerm = X' * errors; % n+1 by 1 vector

partialsRegularizationTerm = [0; theta(2:end)]; % n+1 by 1 vector

grad = partialsWithoutRegularizationTerm / m + partialsRegularizationTerm * lambda / m;


% =============================================================

end
