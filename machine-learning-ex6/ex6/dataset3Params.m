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


%c = [0.01*C, 0.03*C, 0.1*C, 0.3*C, 1*C, 3*C, 10*C, 30*C];

c = [0.01*C 0.03*C 0.1*C 0.3*C 1*C 3*C 10*C 30*C]';

sig1 = [0.01*sigma 0.03*sigma 0.1*sigma 0.3*sigma 1*sigma 3*sigma 10*sigma 30*sigma]';

x1 = [1 2 1]; x2 = [0 4 -1];
row = 0;

for i = 1 : length(c)
  for j = 1 : length(sig1)
  
  row = row + 1;
  model= svmTrain(X, y, c(i), @(x1, x2) gaussianKernel(x1, x2, sig1(j))); 
  predictions = svmPredict(model, Xval);
  error = mean(double(predictions ~= yval));
  
  results(row , :) = [ c(i) sig1(j) error ];   
  
  
  end
end  

final_result = sortrows( results , 3);

C = final_result(1,1);
sigma = final_result(1,2);




% =========================================================================

end
