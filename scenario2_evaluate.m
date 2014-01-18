clc;
clear;
delfigs;
prwaitbar off;

nist_data = prnist(0:9,1:1000);

%% NIST EVAL

clc;

iter = 5;       % Number of performance evaluations
num_test = 50;  % Number of test objects per class

errors = zeros(1:iter, 1);

for i = 1:iter
    % Generate a random training set with 10 objects per class 
    n_trn = gendat(nist_data, 0.01);
    % Calculate trainings prdataset object
    trn = my_rep2(n_trn);
    
    % Train SVC classifier
    w = svc(trn);
    
    % Evaluate performance of classifier
    E = nist_eval('my_rep2', w, num_test);
    
    errors(i) = E;
end

errors'