function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad       = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
%dimension_theta   = length(theta);
%theta_squared_sum = 0;
%
%for i=2:length(theta),
%theta_squared_sum = theta_squared_sum + ( theta(i) * theta(i) );
%end;
%
%regularization_term = (lambda/(2*m)) * theta_squared_sum;
%
%%% Gradient
%
%z = theta'*X';
%J = (1/m) * sum(  -y.*log(sigmoid(z')) - (1-y).*log(1-sigmoid(z')) ) + regularization_term; %
%
%theta(1) = (1/m) * sum( (sigmoid(z') - y) .* X(:,1));
%
%for i=2:length(theta),
%theta(i) = (1/m) * sum( (sigmoid(z') - y) .* X(:,i) + (lambda/m)*theta(i) );
%end;
%grad = theta;

theta_squared_sum = sum(theta(2:end).^2);
regularization_term = (lambda/(2*m)) * theta_squared_sum;

%%% Cost Function

z = theta'*X';
J = (1/m) * sum(  -y.*log(sigmoid(z')) - (1-y).*log(1-sigmoid(z')) ) + regularization_term; 

%%% Gradient

theta(1) = 0;
grad = (1/m) * X' * (sigmoid(z') - y) + (lambda/m) * theta;

% =============================================================

end
