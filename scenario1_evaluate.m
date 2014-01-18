clc;
clear;
delfigs;
prwaitbar off;
prmemory(64000000);
nist_data = prnist(0:9,1:1000);

%% NIST EVAL

clc;

iter = 1;        % Number of performance evaluations
num_test = 100;  % Number of test objects per class

errors = zeros(1:iter, 1);

for i = 1:iter
    % Generate a random training set with 400 objects per class 
    n_trn = gendat(nist_data, 0.8);
    % Calculate trainings prdataset object
    trn = my_rep1(n_trn);
    
    % Train SVC classifier
    w = trn * libsvc([],(proxm([],'r',2.9)),1);
    %w = libsvc(trn,(proxm([],'r',2.9)),1);
    
    % Evaluate performance of classifier
    E = nist_eval('my_rep1', w, num_test);
    
    errors(i) = E;
end

errors'