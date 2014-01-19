clc;
clear;
delfigs;
prwaitbar off;

nist_data = prnist(0:9,1:1000);

%% NIST EVAL
prmemory(64000000);
clc;

iter = 3;        % Number of performance evaluations
num_test = 100;  % Number of test objects per class

errors = zeros(iter, 1);
errors2 = zeros(iter, 1);

for i = 1:iter
    % Generate a random training set with 400 objects per class 
    n_trn = gendat(nist_data, 0.8);
    % Calculate trainings prdataset object

    trn_unscaled = my_rep1(n_trn);

    mapping_scale = scalem(trn_unscaled, 'c-mean');
    trn_scaled = trn_unscaled*mapping_scale;
    
    % domain +/- 0.023 / 0.023,0.021
    
    % Perform PCA
    [mapping_pca, frac] = pcam(trn_scaled, 53);
    
    % Return the dataset after performing PCA
    trn_pca = trn_scaled*mapping_pca;
    
    % Train SVC classifier
    % w = fisherc(trn_pca);
    w = libsvc(trn_pca,(proxm([],'r',2.9)),1);
    w2 = libsvc(trn_pca,(proxm([],'p',2)),1);
    
    w_pca = mapping_scale*mapping_pca*w;
    w2_pca = mapping_scale*mapping_pca*w2;
    %w_pca = mapping_pca*w;
    
    % Evaluate performance of classifier
    E = nist_eval('my_rep1', w_pca, num_test);
    E2 =  nist_eval('my_rep1', w2_pca, num_test);
    
    errors(i) = E;
    errors2(i) = E2;
end

errors'
errors2'
