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

A = zeros(64, 2);
error_val = zeros(64, 1);
v = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]; % column order in for loop

k = 1;
for i = 1:8,
    for j = 1:8,
        C = v(i);
        sigma = v(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        A(k, :) = [C sigma];
        error_val(k) = mean(double(predictions ~= yval));
        k = k + 1;
    end
end

[a, min_index] = min(error_val);
C = A(min_index, 1);
sigma = A(min_index, 2);



% =========================================================================

end
