function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


    predictions = pval < epsilon; % ? by 1 vector, ? is the size of the cross-validation dataset
    
    totalPreP = sum(predictions); % real number, the number of predicted positive exmaples
    
    totalActP = sum(yval); % real number, the number of actual positive examples 
    
    tp = sum((predictions == 1) & (yval ==1)); % real number, the number of true positives;
    
    prec = tp / (totalPreP + 1e-100); % real number, precision, add 1e-100 to prevent from a division by zero warning
    
    rec = tp / (totalActP + 1e-100); % real number, recall, add 1e-100 to prevent from a division by zero warning
    
    F1 = 2 * prec * rec / (prec + rec + 1e-100); % real number, F1 score, add 1e-100 to prevent from a division by zero warning
    
    % another way to compute F1, precision, recall:
    % fp = sum((predictions == 1) & (yval == 0)); % the number of false positives 
    % fn = sum((predictions == 0) & (yval == 1)); % the number of false negtives 
    % prec = tp / (tp + fp);
    % rec = tp / (tp+fn);
    % F1 = 2 * prec * rec / (prec + rec);
    
    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
