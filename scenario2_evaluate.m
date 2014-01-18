clc;
clear;
delfigs;
prwaitbar off;

nist_data = prnist(0:9,1:1000);

%% NIST EVAL

clc;

iter = 4;       % Number of performance evaluations
num_test = 50;  % Number of test objects per class

errors = zeros(1:iter, 2);

for i = 1:iter
    % Generate a random training set with 10 objects per class 
    n_trn = gendat(nist_data, 0.01);
    % Calculate trainings prdataset object
    trn = my_rep2(n_trn);
    
    % Train SVC classifier
    w_fisher = fisherc(trn);
    w_svc = svc(trn);
    
    % Evaluate performance of classifier
    E1 = nist_eval('my_rep2', w_fisher, num_test);
    E2 = nist_eval('my_rep2', w_svc, num_test);
    
    errors(i,1) = E1;   % Fisher
    errors(i,2) = E2;   % SVC
end

errors

load gong.mat;
soundsc(y);

%% Feature selection
clc;

iter = 4;       % Number of performance evaluations
num_test = 50;  % Number of test objects per class

errors = zeros(1:iter, 1);

for i = 1:iter
    % Generate a random training set with 10 objects per class 
    n_trn = gendat(nist_data, 0.01);
    % Calculate trainings prdataset object

    trn_unselected = my_rep2(n_trn);

    [mapping, R] = featself(trn_unselected);
    disp(R);

    trn_featsel = trn_unselected*mapping;

    % Train SVC classifier
    %w_fisher = fisherc(trn_featsel);
    w_svc = svc(trn_featsel);
    
    %w_fisher_map = mapping*w_fisher;
    w_svc_map = mapping*w_svc;
    
    %E1 = nist_eval('my_rep2', w_fisher_map, num_test);
    E2 = nist_eval('my_rep2', w_svc_map, num_test);
    
    %errors(i,1) = E1;   % Fisher
    errors(i,1) = E2;   % SVC
end

errors'

%% PCA
clc;

iter = 4;       % Number of performance evaluations
num_test = 50;  % Number of test objects per class

errors = zeros(1:iter, 1);


for i = 1:iter
    % Generate a random training set with 400 objects per class 
    n_trn = gendat(nist_data, 0.01);
    % Calculate trainings prdataset object

    trn_unscaled = my_rep2(n_trn);

    mapping_scale = scalem(trn_unscaled, 'c-mean');
    trn_scaled = trn_unscaled*mapping_scale;
    
    % Perform PCA
    [mapping_pca, frac] = pcam(trn_scaled, 60);
    
    % Return the dataset after performing PCA
    trn_pca = trn_scaled*mapping_pca;
    
    % Train SVC classifier
    w = svc(trn_pca);
    %w = libsvc(trn,(proxm([],'r',2.9)),1);
    
    w_pca = mapping_scale*mapping_pca*w;
    
    % Evaluate performance of classifier
    E = nist_eval('my_rep2', w_pca, num_test);
    
    errors(i) = E;
end

errors'
notify('test');

%% Code for error graph increasing pixels

avg_error = [4  0.218  0.243,	% 4x4
	 		 6  0.231  0.250,	% 6x6
	 		 8  0.4775 0.2345	% 8x8
	 		 9  0.5525 0.2335	% 9x9
	 		 10 0.403  0.2470,	% 10x10
	 		 11 0.2730 0.2010,  % 11x11
	 		 12 0.2370 0.2420,	% 12x12
	 		 13 0.2220 0.2370,
	 		 14 0.2145	0.2235,
	 		 15 0.2260 0.2125,
	 		 16 0.2215 0.1795,
	 		 17 0.1960 0.2270,
	 		 18 0.1905 0.1925];

clf;
hold on
    plot(avg_error(:,1), avg_error(:,2),'--k');
    plot(avg_error(:,1), avg_error(:,3),'-k');
    title('', 'FontSize', 12);
    xlabel('Size of downscaled image (px)', 'FontSize', 14);
    ylabel('Error Rate', 'FontSize', 14);
    h_legend = legend('Fisher', 'SVC');
    set(h_legend,'FontSize',14);
hold off