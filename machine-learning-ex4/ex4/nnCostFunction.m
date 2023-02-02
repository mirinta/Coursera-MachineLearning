function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Theta1 is a hiddenLayerSize by n+1 matrix
% Theta2 is a outputLayerSize by hiddenLayerSize+1 matrix

% ====================== Compute J with regularization term ======================

X = [ones(m,1) X]; % add ones to X, m by n+1 matrix contains all training examples

a1 = X; % set a1 of the input layer, m by n+1 matrix

z2 = a1 * Theta1'; % inputs of the hidden layer over all training examples, m by hiddenLayerSize matrix 

a2 = sigmoid(z2); % outputs of the hidden layer over all training examples, m by hiddenLayerSize matrix

a2 = [ones(m,1) a2]; % add ones to a2, m by hiddenLayerSize+1 matrix

z3 = a2 * Theta2'; % inputs of the output layer over all training examples, m by outputLayerSize matrix

a3 = sigmoid(z3); % outputs of the output layer over all training examples, m by outputLayerSize matrix

yMatrix = eye(num_labels)(y,:); % expected outputs over all training examples, m by outputLayerSize matrix 

cost = - yMatrix .* log(a3) - (1.0 - yMatrix) .* log(1.0 - a3);

JNoReg = sum(sum(cost)) / m; % J without regularization term

sqrWeights = sum(sum(Theta1(:,2:end) .^2)) + sum(sum(Theta2(:,2:end) .^2)); % sum squared weights (except the first column) in Theta1 and Theta2

reg = sqrWeights * lambda / (2.0 * m) % value of the regularization term

J = JNoReg + reg;

% ====================== Compute Theta1_grad and Theta2_grad with regularization term ======================

% ====================== vectorial implementation ======================
 
delta3 = a3 - yMatrix; % m by outputLayerSize matrix, errors between predictions and labels

delta2 = (delta3 * Theta2)(:,2:end) .* sigmoidGradient(z2); % m by hiddenLayerSize matrix, remove the fist column of delta3 * Theta2

Delta2 = delta3' * a2; % outputLayerSize by hiddenLayerSize+1 matrix

Delta1 = delta2' * a1; % hiddenLayerSize by n+1 matrix

Theta1_grad = Delta1 / m + [zeros(hidden_layer_size,1) Theta1(:,2:end)] * lambda / m;

Theta2_grad = Delta2 / m + [zeros(num_labels,1) Theta2(:,2:end)] * lambda / m;

% ====================== using a for-loop to compute Theta1_grad and Theta2_grad ======================

%Delta1 = zeros(size(Theta1)); % initialize Delta1, hiddenLayerSize by n+1 matrix

%Delta2 = zeros(size(Theta2)); % initialize Delta2, outputLayerSize by hiddenLayerSize+1 matrix

%for i = 1 : m, % for loop, given ith training example
  
%  ra1 = X(i,:)'; % n+1 by 1 vector
   
%  rz2 = Theta1 * ra1; % hiddenLayerSize by 1 vector, inputs of the hidden layer
 
%  ra2 = sigmoid(rz2); % hiddenLayerSize by 1 vector, outputs of the hidden layer 
  
%  ra2 = [1; ra2]; % hiddenLayerSize+1 by vector, add one to ra2
  
%  rz3 = Theta2 * ra2; % outputLayerSize by 1 vector, inputs of the output layer
  
%  ra3 = sigmoid(rz3); % outputLayerSize by 1 vector, outputs of the output layer
  
%  yDesired = yMatrix(i,:)'; % outputLayerSize by 1 vector, desired output of the output layer
  
%  delta3 = ra3 - yDesired; % outputLayerSize by 1 vector, compute the prediction error
  
%  delta2 = (Theta2' * delta3)(2:end, 1) .* sigmoidGradient(rz2); % hiddenLayerSize by 1 vector
  
%  Delta1 = Delta1 + delta2 * ra1';
  
%  Delta2 = Delta2 + delta3 * ra2';
 
% end

%Theta1_grad = Delta1 / m + [zeros(hidden_layer_size,1) Theta1(:,2:end)] * lambda / m;

%Theta2_grad = Delta2 / m + [zeros(num_labels,1) Theta2(:,2:end)] * lambda / m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
