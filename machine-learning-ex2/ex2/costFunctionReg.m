function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
hypothesis = 1 ./ (1+exp(-X*theta));

JRegTerm = (lambda/(2*m)) * ( sum(theta(2:size(theta)) .^ 2) );
J = (1/m)*(sum(  (-y.*(log(hypothesis))) - ((1-y).*(log(1-hypothesis))) ) ) + JRegTerm;

n = size(theta)-1; % since it includes theta0 also
temp_sum = sum(((hypothesis)-y) .* X(1:m,:),1);   % 1x n+1 dimensional vector
grad(1) = temp_sum(1)' ./ m ;
grad(2:size(theta)) = temp_sum(2:size(theta))' ./ m + (lambda/m)*theta(2:size(theta)); 



% =============================================================

end
