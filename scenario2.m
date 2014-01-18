clc;
clear;
delfigs;
prwaitbar off;

nist_data = prnist(0:9,1:1000);

%%
% Select train and test set
n_trn = gendat(nist_data, 0.01);
n_tst = gendat(nist_data, 0.05);

trn = my_rep2(n_trn); disp('Finished calculating train data...');
tst = my_rep2(n_tst); disp('Finished calculating test data...');

num_measurements = 5;
E = zeros(num_measurements, 5);

clc;

%W = trn*{knnc,parzenc,fisherc,nmc,svc};
%E = tst*W*testc

% Testing SVC with different kernels

% SVC kernels tested:
% polynomial (p), exponential (e), radial (r), sigmoid (s),
% distance(d), minkowski (m), city-block (c)
%W = trn*{svc([],(proxm([],'p',1)),1),svc([],(proxm([],'e',1)),1),svc([],(proxm([],'r',1)),1),svc([],(proxm([],'s',1)),1),svc([],(proxm([],'d',1)),1),svc([],(proxm([],'m',1)),1),svc([],(proxm([],'c',1)),1)};
%E = tst*W*testc

W = svc(trn);
[E,C] = testc(tst, W);
disp(E);

% Error percentage per digit

cs = classsizes(tst);
errors = zeros(10,1);

clc;

for i = 1:10
   error = C(i)/cs(i);
   errors(i,1) = error;
   fprintf('%1.2f & ', error);
end

%fprintf('%f', E);
%errors'

%%
for i = 1:num_measurements
    W = trn*{knnc([],3),parzenc,fisherc,nmc,svc};
    E(i,1:5) = tst*W*testc;
end

avg_error = mean(E,1);
clc;
fprintf('KNN classifier: %f\n', E(1));
fprintf('Parzen classifier: %f\n', avg_error(2));
fprintf('Fisher classifier: %f\n', avg_error(3));
fprintf('NM classifier: %f\n', avg_error(4));
fprintf('SVC classifier: %f\n', avg_error(5));

%% Benchmarking

n_trn = gendat(nist_data, 0.01);
n_tst = gendat(nist_data, 0.05);

trn = my_rep2(n_trn); disp('Finished calculating train data...');
tst = my_rep2(n_tst); disp('Finished calculating test data...');

w = svc(trn, proxm([],'p',1));
%E = tst*W*testc

E = nist_eval('my_rep2.m', w, 10);


%% ROC Curve

E = roc(tst, w);
plote(E);




