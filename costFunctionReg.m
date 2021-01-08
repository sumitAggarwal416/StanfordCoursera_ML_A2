function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
j = length(theta);
z = X * theta;
cost = -y' * log(sigmoid(z)) - (1 -y') * log(1 - sigmoid(z));
% You need to return the following variables correctly 
%J = (cost * (1 / m)) + (lambda/(2 * m)) * (theta(2:j))' * theta(2:j); 
J = (cost * (1 / m)) + (lambda/(2 * m)) * sum(theta .^ 2);
grad = zeros(size(theta));

thetaZero = theta;
thetaZero(1) = 0;

grad = (((sigmoid(z) - y)' * X) * (1 / m)) + (lambda / m) * thetaZero';

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
