clear;
delfigs;
prwaitbar off;

a = prnist([0:9],[1:2:1000]);
%show(a);

% Preprocessing of letter images
% See page 17 of Lab Course Manual
preproc = im_box([],0,1)*im_resize([],[9 9],'bicubic')*im_box([],1,0);
a = a*preproc;

%a = im_resize(a,[8 8],'bilinear');
pr_a = prdataset(a);

%make scale-mappings of the original pixel-dataset
mapping1 = scalem(pr_a, 'variance');
mapping2 = scalem(pr_a, 'c-mean');
%scale the original pixel-datasets
scaledPixels1 = pr_a*mapping1;
scaledPixels2 = pr_a*mapping2;
% Perform Principal Component Analysis
% Extraction features gives ~95% of the variance
[mapping1, frac1] = pcam(scaledPixels1, 36);
[mapping2, frac2] = pcam(scaledPixels2, 33);
[mapping3, frac3] = pcam(pr_a,33);
% pcaData# now contains an affine PCA mapping
pcaData1 = scaledPixels1*mapping1;
pcaData2 = scaledPixels2*mapping2;
pcaData3 = pr_a * mapping3;

% Calculate all the image features
features = im_features(a, a, 'all');
mapping = scalem(features, 'variance');
scaledData = features*mapping;

% Perform Principal Component Analysis
% Extraction 10 features gives ~95% of the variance
% I suppose this is the best number of features to extract
[mapping, frac] = pcam(scaledData, 10);
% w now contains an affine PCA mapping
pcaData = scaledData*mapping;

% Perform cross-validation to check performance
% 10-fold cross-validation

%Fully written: pixel dataset, abbreviations: PCA dataset
[train1,test1,Itrain1,Itest1] = gendat(pcaData,0.8);
[train2,test2,Itrain2,Itest2] = gendat(pr_a,0.8);
[train3,test3,Itrain3,Itest3] = gendat(pcaData1,0.8);
[train4,test4,Itrain4,Itest4] = gendat(pcaData2,0.8);
[train5,test5,Itrain5,Itest5] = gendat(pcaData3,0.8);

%for j=0.1:0.1:3,
%    wparz = parzenc(trn,j);
%    pparz = test*wparz*testc;
%    result([j]) = pparz;
%end
%%
%All images used preproc and are rotated and resized

%trainingset1: PCA of scaled features
%trainingset2: pixels
%trainingset3: PCA of scaled pixels with variance
%trainingset4: PCA of scaled pixels with c-mean
%trainingset5: PCA of pixels


%Used dimensions in function pcam per size and per mapping
%20 objects per class
% size/percent/method       feature_map     pix_map_var     pix_map_c-mean
% 8 * 8 / 95 / nearest      9               36              39
% 8 * 8 / 95 / bilinear     9               32              27
% 8 * 8 / 95 / bicubic

%500 objects per class
% size/percent/method       feature_map     pix_map_var     pix_map_c-mean
% 8 * 8 / 95 / nearest      
% 8 * 8 / 95 / bilinear     10              36              33
% 8 * 8 / 95 / bicubic


%Classification error of parzenc
[W1,H1] = parzenc(train1);
[W2,H2] = parzenc(train2);
[W3,H3] = parzenc(train3);
[W4,H4] = parzenc(train4);
[W5,H5] = parzenc(train5);
E1 = test1*W1*testc
E2 = test2*W2*testc
E3 = test3*W3*testc
E4 = test4*W4*testc
E5 = test5*W5*testc

%25 objects / class
%image size 128 * 128 95%   E1 = 0.4    E2 = 0.26
%image size 8 * 8 90%       E1 = 0.58   E2 = 0.24   E3 = 0.38   E4 = 0.22
%image size 8 * 8 95%       E1 = 0.60   E2 = 0.38   E3 = 0.28   E4 = 0.30

%500 objects / class
%resize: bilinear
%image size 8 * 8 95%       E1 = 0.296   E2 = 0.041   E3 = 0.041   E4 = 0.038   E5 = 0.034
%image size 9 * 9 95%       E1 = 0.244   E2 = 0.032   E3 = 0.051   E4 = 0.038   E5 = 0.03

%resize: bicubical
%image size 8 * 8 95%       E1 = 0.247   E2 = 0.043   E3 = 0.05   E4 = 0.034   E5 = 0.04
%image size 9 * 9 95%       E1 = 0.224   E2 = 0.038   E3 = 0.06   E4 = 0.04 E5 = 0.031
%%
%Classification error of ldc
W1 = ldc(train1);
W2 = ldc(train2);
W3 = ldc(train3);
W4 = ldc(train4);
W5 = ldc(train5);
E1 = test1*W1*testc
E2 = test2*W2*testc
E3 = test3*W3*testc
E4 = test4*W4*testc
E5 = test5*W5*testc

%25 objects / class
%resize: nearest
%image size 128 * 128   95%     E1 = 0.34   E2 = 0.74
%image size 8 * 8       90%     E1 = 0.44   E2 = 0.48   E3 = 0.46   E4 = 0.28
%image size 8 * 8       95%     E1 = 0.42   E2 = 0.40   E3 = 0.22   E4 = 0.30
%resize: bilinear
%image size 8 * 8       95%     E1 = 0.40   E2 = 0.26   E3 = 0.26   E4 = 0.18

%500 objects / class
%resize: bilinear
%image size 8 * 8       95%     E1 = 0.348   E2 = 0.116   E3 = 0.1    E4 = 0.111  E5 = 0.113
%image size 9 * 9       95%     E1 = 0.279   E2 = 0.806   E3 = 0.113  E4 = 0.113  E5 = 0.104

%resize: bicubical
%image size 8 * 8 95%       E1 = 0.314   E2 = 0.104   E3 = 0.115   E4 = 0.101   E5 = 0.121
%image size 9 * 9 95%       E1 = 0.258  E2 = 0.682  E3 = 0.11   E4 = 0.12  E5 = 0.109
%%
%Classification error of qdc
W1 = qdc(train1);
W2 = qdc(train2);
W3 = qdc(train3);
W4 = qdc(train4);
W5 = qdc(train5);
E1 = test1*W1*testc
E2 = test2*W2*testc
E3 = test3*W3*testc
E4 = test4*W4*testc
E5 = test5*W5*testc

%25 objects / class 
%resize: nearest
%image size 128 * 128   95%     E1 = 0.36   E2 = 0.9
%image size 8 * 8       95%     E1 = 0.54   E2 = 0.9    E3 =  0.74  E4 = 0.68
%resize: bilinear
%image size 8 * 8       95%     E1 = 0.46   E2 = 0.86   E3 = 0.46   E4 = 0.50 

%500 objects / class
%resize: bilinear
%image size 8 * 8       95%     E1 = 0.308   E2 = 0.064   E3 =  0.045  E4 = 0.04   E5 = 0.033
%image size 9 * 9      95%     E1 = 0.26    E2 = 0.242   E3 = 0.066   E4 = 0.038  E5 = 0.033

%resize: bicubical
%image size 8 * 8 95%       E1 = 0.254   E2 = 0.056   E3 = 0.036   E4 = 0.034   E5 = 0.039
%image size 9 * 9 95%       E1 = 0.223   E2 = 0.147   E3 = 0.053   E4 = 0.042   E5 = 0.034

%%
%Classification error of fisher
W1 = fisherc(train1);
W2 = fisherc(train2);
W3 = fisherc(train3);
W4 = fisherc(train4);
W5 = fisherc(train5);
E1 = test1*W1*testc
E2 = test2*W2*testc
E3 = test3*W3*testc
E4 = test4*W4*testc
E5 = test5*W5*testc
%25 objects / class
%image size 128 * 128   95%     E1 = 0.38   E2 = 0.7
%image size 8 * 8       95%     E1 = 0.38   E2 = 0.36    E3 =  0.26  E4 = 0.38

%500 objects / class
%resize: bilinear
%image size 8 * 8       95%     E1 = 0.352   E2 = 0.144    E3 =  0.133  E4 = 0.13     E5 = 0.14
%image size 9 * 9       95%     E1 = 0.307   E2 = 0.153    E3 = 0.148   E4 = 0.141    E5 = 0.136

%resize: bicubical
%image size 8 * 8 95%       E1 = 0.32    E2 = 0.126  E3 = 0.153   E4 = 0.139  E5 = 0.146
%image size 9 * 9 95%       E1 = 0.286   E2 = 0.146  E3 = 0.137   E4 = 0.149  E5 = 0.134
%%
%Classification error of nmc
W1 = nmc(train1);
W2 = nmc(train2);
W3 = nmc(train3);
W4 = nmc(train4);
W5 = nmc(train5);
E1 = test1*W1*testc
E2 = test2*W2*testc
E3 = test3*W3*testc
E4 = test4*W4*testc
E5 = test5*W5*testc
%25 objects / class
%image size 128 * 128   95%     E1 = 0.46   E2 = 0.28
%image size 8 * 8       95%     E1 = 0.5    E2 = 0.36   E3 = 0.2    E4 = 0.34

%500 objects / class
%resize: bilinear
%image size 8 * 8       95%     E1 = 0.418    E2 = 0.19    E3 = 0.193   E4 = 0.168  E5 = 0.169
%image size 9 * 9       95%     E1 = 0.368    E2 = 0.179   E3 = 0.208   E4 = 0.172  E5 = 0.188

%resize: bicubical
%image size 8 * 8 95%       E1 = 0.381   E2 = 0.177   E3 = 0.212   E4 = 0.184   E5 = 0.199
%image size 9 * 9 95%       E1 = 0.336   E2 = 0.186   E3 = 0.22   E4 = 0.177   E5 = 0.178
%%
%Classification error of knnc, k is calculated with respect to
%leave-one-out error
W1 = knnc(train1);
W2 = knnc(train2);
W3 = knnc(train3);
W4 = knnc(train4);
W5 = knnc(train5);
E1 = test1*W1*testc
E2 = test2*W2*testc
E3 = test3*W3*testc
E4 = test4*W4*testc
E5 = test5*W5*testc
%25 objects / class
%image size 128 * 128   95%     E1 = 0.38   E2 = 0.26
%image size 8 * 8       95%     E1 = 0.52   E2 = 0.38   E3 = 0.22   E4 = 0.34

%500 objects / class
%resize: bilinear
%image size 8 * 8       95%     E1 = 0.303   E2 = 0.043   E3 = 0.045   E4 = 0.044  E5 = 0.036
%image size 9 * 9       95%     E1 = 0.249   E2 = 0.031   E3 = 0.057   E4 = 0.036  E5 = 0.035

%resize: bicubical
%image size 8 * 8 95%       E1 = 0.264   E2 = 0.044   E3 = 0.051   E4 = 0.033   E5 = 0.046
%image size 9 * 9 95%       E1 = 0.211   E2 = 0.042   E3 = 0.06   E4 = 0.042   E5 = 0.032
%%
%Classification error of ldc
W1 = fisherc(train1);
W2 = fisherc(train2);
W3 = fisherc(train3);
W4 = fisherc(train4);
W5 = fisherc(train5);
E1 = test1*W1*testc
E2 = test2*W2*testc
E3 = test3*W3*testc
E4 = test4*W4*testc
E5 = test5*W5*testc
%25 objects / class
%image size 128 * 128   95%     E1 = 0.38   E2 = 0.36   E3 = 0.26   E4 = 0.38

%500 objects / class
%resize: bilinear
%image size 8 * 8   95%         E1 = 0.352   E2 = 0.144   E3 = 0.133   E4 = 0.13   E5 = 0.14
%image size 9 * 9   95%         E1 = 0.307   E2 = 0.153   E3 = 0.148   E4 = 0.141   E5 = 0.136

%resize: bicubical
%image size 8 * 8 95%       E1 = 0.32   E2 = 0.126   E3 = 0.153   E4 = 0.139   E5 = 0.146
%image size 9 * 9 95%       E1 = 0.286   E2 = 0.146   E3 = 0.137   E4 = 0.149   E5 = 0.134

%%
W1 = adaboostc(train1,parzenc,5,[],0);
W2 = adaboostc(train2,parzenc,5,[],0);
W3 = adaboostc(train3,parzenc,5,[],0);
W4 = adaboostc(train4,parzenc,5,[],0);
W5 = adaboostc(train5,parzenc,5,[],0);
E1 = test1*W1*testc
E2 = test2*W2*testc
E3 = test3*W3*testc
E4 = test4*W4*testc
E5 = test5*W5*testc

%%
v = [parzenc*classc knnc*classc qdc*classc]*fisherc;
W1 = train1*v;
W2 = train2*v;
W3 = train3*v;parzenc*classc knnc*classc qdc*classc]*fisherc;
W4 = train4*v;
W5 = train5*v;
E1 = test1*W1*testc
E2 = test2*W2*testc
E3 = test3*W3*testc
E4 = test4*W4*testc
E5 = test5*W5*testc

%%
W1 = train1 * svc([],(proxm([],'p',1)),1);
E1 = train1*W1*classc;
