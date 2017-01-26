cd 'F:\Dropbox\ml-class\mlclass-ex4'

clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;

load('ex4data1.mat');
m = size(X, 1);

load('ex4weights.mat');

nn_params = [Theta1(:) ; Theta2(:)];

lambda = 1;

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1) X];

a1 = X;

z2 = a1*Theta1';
a2 = [ones(m,1) sigmoid(z2)];

z3 = a2*Theta2';
a3 = sigmoid(z3);

y1 = eye(num_labels)(y,:);
J = -(1/m) * sum( sum(y1 .* log(a3) + (1-y1).*log(1-a3)) );

vec = [Theta1(size(Theta1,1)+1:end) Theta2(size(Theta2,1)+1:end)];
Reg = lambda/(2*m) * sum( vec .^ 2);

J = J + Reg;

delta3 = a3 - y1;

res2   = reshape(Theta2(:)(num_labels+1:end),num_labels,hidden_layer_size);
delta2 = (res2' * delta3')' .* sigmoidGradient(z2);

Delt1 = (a1' * delta2)';
Delt2 = (a2' * delta3)';

reg1 = (lambda/m) * Theta1(:,2:end) .^ 2;
reg2 = (lambda/m) * Theta2(:,2:end) .^ 2;

Theta1_grad = (1/m) * Delt1;
Theta2_grad = (1/m) * Delt2;

Theta1_grad = Theta1_grad(:,2:end) + reg1;
Theta2_grad = Theta2_grad(:,2:end) + reg2;
