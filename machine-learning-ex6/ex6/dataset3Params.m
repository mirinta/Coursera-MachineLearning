function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

minError = 10000.0; % initialize a minimum error

params = [0.01 , 0.03, 0.1, 0.3, 1, 3, 10, 30]; % all possible values for C and sigma

for CVal = params,
  
	for sigmaVal = params,
    
		model =  svmTrain(X, y, CVal, @(x1, x2) gaussianKernel(x1, x2, sigmaVal)); % train the SVM model, given C and sigma
    
		predictions = svmPredict(model , Xval); % prediction of the cross-validation dataset
    
		error = mean(double(predictions ~= yval)); % prediction error of the cross-validation dataset
    
		if minError > error, 
      
			minError = error; % if the current error is less than the minError, update the value of the minimum error
      
			C = CVal; % update the value of C
      
			sigma = sigmaVal; % update the value of sigma
      
		end
    
	end
  
end


% =========================================================================

end
