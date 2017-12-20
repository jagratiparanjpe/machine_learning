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





hx = X * theta; % Calculate hx function 

regparameter = (lambda/(2*m)) * (sumsq(theta(2:end)));



error = (hx - y).^2; % Calculate squared error 
M = 2*m;
J = sum(error)/M + regparameter; % cost function for given value of theta

% gradient calulation

diff = hx - y;
org_grad = (1.0/m) * (X' * diff);
reg_gard = (lambda/m) * theta;
reg_gard(1) = 0;

grad = org_grad + reg_gard;








% =========================================================================

grad = grad(:);

end
