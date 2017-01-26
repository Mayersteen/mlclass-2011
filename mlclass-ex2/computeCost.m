function [J] = computeCost(theta, X, y)

m = length(y);

z = theta'*X';
J = (1/m) * sum(  -y.*log(sigmoid(z')) - (1-y).*log(1-sigmoid(z')) ); 

end