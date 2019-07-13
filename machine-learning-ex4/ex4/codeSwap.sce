clc;
clear;
close;

function g = sigmoid(z)
//SIGMOID Compute sigmoid functoon
//   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));
end



function theta = thetaInit(m,n)
    theta = rand(m,n,'normal')
endfunction

function val = sigmoidDiff(net)
    val =  sigmoid(net).*(1-sigmoid(net))
endfunction
loadmatfile('irisDataset.mat')

X = irisInputs';
y = irisTargets';

//load trainingData.dat
// loads data into X variable and outputs in y variable
// ---------Architecture--------------------------------------
num_h = 4;       // number of nodes in each hidden layer
n = size(X,2);   // number of inputs
p = size(y,2);   // number of outputs
m = size(X,1);   // no. of training exaples(patterns) in matrix
h = 7;           // no.of nodes in the hidden layer
eta = 0.3;       // learning rate
Xnew = [ones(m,1),X];

// numTheta = 6;
//----generate random initial thetas -----
theta1 = thetaInit(h,n+1);
theta2 = thetaInit(h,h+1);
theta3 = thetaInit(h,h+1);
theta4 = thetaInit(h,h+1);
theta5 = thetaInit(h,h+1);
theta6 = thetaInit(p,h+1);

//----Calculating outward---------
net2 = Xnew * theta1'; // X(m*n+1) * theta1'((n+1),h1)
fNet2 = sigmoid(net2);
net3 = [ones(size(fNet2,1),1),fNet2]*theta2';
fNet3 = sigmoid(net3);
net4 = [ones(size(fNet3,1),1),fNet3]*theta3';
fNet4 = sigmoid(net4);
net5 = [ones(size(fNet4,1),1),fNet4]*theta4';
fNet5 = sigmoid(net5);
net6 = [ones(size(fNet5,1),1),fNet5]*theta5';
fNet6 = sigmoid(net6);
net7 = [ones(size(fNet6,1),1),fNet6]*theta6';
fNet7 = sigmoid(net7);


//------Calculating deltas -------
// at o/p neuron
delta7 = (y-fNet7) .* sigmoidDiff(net7);

// at hidden neurons
delta6 = (delta7 * theta6(:,2:$)) .* sigmoidDiff(net6);
delta5 = (delta6 * theta5(:,2:$)) .* sigmoidDiff(net5);
delta4 = (delta5 * theta4(:,2:$)) .* sigmoidDiff(net4);
delta3 = (delta4 * theta3(:,2:$)) .* sigmoidDiff(net3);
delta2 = (delta3 * theta2(:,2:$)) .* sigmoidDiff(net2);
//delta1 = (delta2 * theta1(:,2:$)) .* sigmoidDiff(net1);

//------ Calculating gradients-----
//deltaTheta7 = eta * (delta7*fNet6);
deltaTheta6 = zeros(3,8);
for i = 1:size(fNet7,1)
    for j = 1:size(fNet7,2)
        deltaTheta6(1:3) = deltaTheta6(1:3) + eta* fNet7(i,j)*delta7(i,1:3);
    end
end




