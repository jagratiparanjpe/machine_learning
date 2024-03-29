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
%c(1) = 1; sigma_vec(1) = 0.3;

range = 8;
c = [0.01 0.03 0.1 0.3 1, 3, 10 30];
sigma_vec = [0.01 0.03 0.1 0.3 1, 3, 10 30];


k = 0;
parameter_vec = zeros(size(range ** 2, 3));
for i = 1:range
  for j = 1:range
    model= svmTrain(X, y, c(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j))); 
    prediction = svmPredict(model, Xval);
    prediction_error = mean(double(prediction ~= yval));
    possible_vec(++k,:) = [c(i), sigma_vec(j), prediction_error]; 
    
  end
end

% find the minimum value of the prediction error 
[min_value, min_index] = min(possible_vec(:, 3));
C = possible_vec(min_index, 1);

sigma = possible_vec(min_index , 2);


% =========================================================================

end
