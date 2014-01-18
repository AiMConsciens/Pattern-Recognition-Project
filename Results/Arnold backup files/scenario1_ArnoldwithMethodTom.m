%% Start DIPimage - Arnold

addpath('C:\Program Files\DIPimage 2.5.1\common');

dip_initialise;
dipimage;
prmemory(64000000);

%%

clc;
clear;
delfigs;
prwaitbar off;

nist_data = prnist(0:9,1:1000);

%%
% Select train and test set
[n_trn,n_tst] = gendat(nist_data,0.8);

trn = my_rep(n_trn); disp('Finished calculating train data...');
tst = my_rep(n_tst); disp('Finished calculating test data...');

%%
num_measurements = 4;
E = zeros(num_measurements, 4);

clc;

W = trn*{libsvc([],(proxm([],'p',1)),1),libsvc([],(proxm([],'s',1)),1),libsvc([],(proxm([],'d',1)),1),libsvc([],(proxm([],'r',1)),1)};
%E = tst*W*testc

%% Error percentage per digit
W = svc(trn);
[E,C] = testc(tst, W);

clc;
cs = classsizes(tst);
errors = zeros(10,1);

for i = 1:10
   error = C(i)/cs(i);
   errors(i,1) = error;
   fprintf('Error for digit %i = %f\n', i-1, error);
end

fprintf('Total error = %f\n', E);
errors'

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

%%



