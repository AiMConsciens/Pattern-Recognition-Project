clear;
delfigs;
prwaitbar off;


%show(a);

% Preprocessing of letter images
% See page 17 of Lab Course Manual

%% Generate standard classifier outputs
% nearest / bilinear / bicubic
% 8X8,9X9,10X10,11X11,12X12,13X13
num_measurements = 6;
E = zeros(num_measurements, 6);

for i = 0:(num_measurements-1)
    a = prnist([0:9],[1:2:1000]);
    preproc = im_box([],0,1)*im_resize([],[(i+8) (i+8)],'bicubic')*im_box([],1,0);
    a = a*preproc;
    pr_a = prdataset(a);
    
    [train,test] = gendat(pr_a,0.8);

    clc;
    
    W = train * {parzenc,ldc,qdc,fisherc,nmc,knnc};
   for j = 1:num_measurements
        E(i+1,j) = test*W(j)*testc;
    end
end
%% Generate SVM outputs

num_measurements = 6;
number_classifiers = 2;
E = zeros(num_measurements, number_classifiers);

for i = 0:(num_measurements-1)
    a = prnist([0:9],[1:2:1000]);
    preproc = im_box([],0,1)*im_resize([],[(i+8) (i+8)],'bilinear')*im_box([],1,0);
    a = a*preproc;
    pr_a = prdataset(a);
    
    [train,test] = gendat(pr_a,0.8);

    clc;
    W = train * {libsvc([],(proxm([],'p',1)),1),libsvc([],(proxm([],'s',1)),1),libsvc([],(proxm([],'d',1)),1),libsvc([],(proxm([],'r',1)),1),libsvc([],(proxm([],'e',1)),1),libsvc([],(proxm([],'p',2)),1),libsvc([],(proxm([],'r',2.5)),1)};

    for j = 1:number_classifiers
        E(i+1,j) = test*W(j)*testc;
    end
end

%% Generate PCA Mappings
E = zeros(5,11,6);

for k = 1:6
    
    a = prnist([0:9],[1:2:1000]);
    preproc = im_box([],0,1)*im_resize([],[(k+8) (k+8)],'bicubic')*im_box([],1,0);
    a = a*preproc;
    pr_a = prdataset(a);

    %make scale-mappings of the original pixel-dataset
    mapping1 = scalem(pr_a, 'c-variance');
    mapping2 = scalem(pr_a, 'c-mean');
    mapping3 = scalem(pr_a, 'domain');

    %scale the original pixel-datasets
    scaledPixels1 = pr_a*mapping1;
    scaledPixels2 = pr_a*mapping2;
    scaledPixels3 = pr_a*mapping3;

    % Perform Principal Component Analysis
    % Extraction features gives ~95% of the variance
    [mapping1, frac1] = pcam(scaledPixels1, 46);
    [mapping2, frac2] = pcam(scaledPixels2, 42);
    [mapping3, frac3] = pcam(scaledPixels3, 42);
    [mapping4, frac4] = pcam(pr_a,42);

    % pcaData# now contains an affine PCA mapping
    pcaData1 = scaledPixels1*mapping1;
    pcaData2 = scaledPixels2*mapping2;
    pcaData3 = scaledPixels3*mapping3;
    pcaData4 = pr_a * mapping4;

    % Calculate all the image features
    %features = im_features(a, a, 'all');
    %mapping5 = scalem(features, 'variance');
    %mapping6 = scalem(features, 'c-variance');
    %mapping7 = scalem(features, 'c-mean');
    %mapping8 = scalem(features, 'domain');

    %scaledData5 = features*mapping5;
    %scaledData6 = features*mapping6;
    %scaledData7 = features*mapping7;
    %scaledData8 = features*mapping8;

    % Perform Principal Component Analysis
    % Extraction around 45 features gives ~95% of the variance
    % I suppose this is the best number of features to extract
    %[mapping5, frac5] = pcam(scaledData5, 9);
    %[mapping6, frac6] = pcam(scaledData6, 9);
    %[mapping7, frac7] = pcam(scaledData7, 1);
    %[mapping8, frac8] = pcam(scaledData8, 9);
    % w now contains an affine PCA mapping
    %pcaData5 = scaledData5*mapping5;
    %pcaData6 = scaledData6*mapping6;
    %pcaData7 = scaledData7*mapping7;
    %pcaData8 = scaledData8*mapping8;

    % Perform cross-validation to check performance
    % 10-fold cross-validation

    %Fully written: pixel dataset, abbreviations: PCA dataset

    %PCA on pixels
    [Train{1},Test{1}] = gendat(pcaData1,0.8);
    [Train{2},Test{2}] = gendat(pcaData2,0.8);
    [Train{3},Test{3}] = gendat(pcaData3,0.8);
    [Train{4},Test{4}] = gendat(pcaData4,0.8);
    %PCA on features
    %[Train{5},Test{5}] = gendat(pcaData5,0.8);
    %[Train{6},Test{6}] = gendat(pcaData6,0.8);
    %[Train{7},Test{7}] = gendat(pcaData7,0.8);
    %[Train{8},Test{8}] = gendat(pcaData8,0.8);
    %Raw pixels
    [Train{5},Test{5}] = gendat(pr_a,0.8);
    %Raw features
    %[Train{10},Test{10}] = gendat(features,0.8);

    % Generate standard classifiers output
    num_classifiers = 6;
    num_measurements = 5
    %E = zeros(num_measurements, num_classifiers);

        clc;
    for i = 1:num_measurements 
        W = Train{i} * {parzenc,ldc,qdc,fisherc,nmc,knnc};
        for j = 1:num_classifiers
            E(i,j,k) = Test{i}*W(j)*testc;
        end
    end

    % Generate SVM outcomes
    num_classifiers = 4;
    num_measurements = 5;
    %E = zeros(num_measurements, num_classifiers);

        clc;
    for i = 1:num_measurements 
        W = Train{i} * {libsvc([],(proxm([],'p',1)),1),libsvc([],(proxm([],'e',1)),1),libsvc([],(proxm([],'p',2)),1),libsvc([],(proxm([],'r',2.5)),1)};

        for j = 1:num_classifiers
            E(i,j+6,k) = Test{i}*W(j)*testc;
        end
    end
end

F = zeros(5,11,6);

for k = 1:6
    
    a = prnist([0:9],[1:2:1000]);
    preproc = im_box([],0,1)*im_resize([],[(k+8) (k+8)],'bilinear')*im_box([],1,0);
    a = a*preproc;
    pr_a = prdataset(a);

    %make scale-mappings of the original pixel-dataset
    mapping1 = scalem(pr_a, 'c-variance');
    mapping2 = scalem(pr_a, 'c-mean');
    mapping3 = scalem(pr_a, 'domain');

    %scale the original pixel-datasets
    scaledPixels1 = pr_a*mapping1;
    scaledPixels2 = pr_a*mapping2;
    scaledPixels3 = pr_a*mapping3;

    % Perform Principal Component Analysis
    % Extraction features gives ~95% of the variance
    [mapping1, frac1] = pcam(scaledPixels1, 46);
    [mapping2, frac2] = pcam(scaledPixels2, 42);
    [mapping3, frac3] = pcam(scaledPixels3, 42);
    [mapping4, frac4] = pcam(pr_a,42);

    % pcaData# now contains an affine PCA mapping
    pcaData1 = scaledPixels1*mapping1;
    pcaData2 = scaledPixels2*mapping2;
    pcaData3 = scaledPixels3*mapping3;
    pcaData4 = pr_a * mapping4;

    % Calculate all the image features
    %features = im_features(a, a, 'all');
    %mapping5 = scalem(features, 'variance');
    %mapping6 = scalem(features, 'c-variance');
    %mapping7 = scalem(features, 'c-mean');
    %mapping8 = scalem(features, 'domain');

    %scaledData5 = features*mapping5;
    %scaledData6 = features*mapping6;
    %scaledData7 = features*mapping7;
    %scaledData8 = features*mapping8;

    % Perform Principal Component Analysis
    % Extraction around 45 features gives ~95% of the variance
    % I suppose this is the best number of features to extract
    %[mapping5, frac5] = pcam(scaledData5, 9);
    %[mapping6, frac6] = pcam(scaledData6, 9);
    %[mapping7, frac7] = pcam(scaledData7, 1);
    %[mapping8, frac8] = pcam(scaledData8, 9);
    % w now contains an affine PCA mapping
    %pcaData5 = scaledData5*mapping5;
    %pcaData6 = scaledData6*mapping6;
    %pcaData7 = scaledData7*mapping7;
    %pcaData8 = scaledData8*mapping8;

    % Perform cross-validation to check performance
    % 10-fold cross-validation

    %Fully written: pixel dataset, abbreviations: PCA dataset

    %PCA on pixels
    [Train{1},Test{1}] = gendat(pcaData1,0.8);
    [Train{2},Test{2}] = gendat(pcaData2,0.8);
    [Train{3},Test{3}] = gendat(pcaData3,0.8);
    [Train{4},Test{4}] = gendat(pcaData4,0.8);
    %PCA on features
    %[Train{5},Test{5}] = gendat(pcaData5,0.8);
    %[Train{6},Test{6}] = gendat(pcaData6,0.8);
    %[Train{7},Test{7}] = gendat(pcaData7,0.8);
    %[Train{8},Test{8}] = gendat(pcaData8,0.8);
    %Raw pixels
    [Train{5},Test{5}] = gendat(pr_a,0.8);
    %Raw features
    %[Train{10},Test{10}] = gendat(features,0.8);

    % Generate standard classifiers output
    num_classifiers = 6;
    num_measurements = 5
    %E = zeros(num_measurements, num_classifiers);

        clc;
    for i = 1:num_measurements 
        W = Train{i} * {parzenc,ldc,qdc,fisherc,nmc,knnc};
        for j = 1:num_classifiers
            F(i,j,k) = Test{i}*W(j)*testc;
        end
    end

    % Generate SVM outcomes
    num_classifiers = 4;
    num_measurements = 5;
    %E = zeros(num_measurements, num_classifiers);

        clc;
    for i = 1:num_measurements 
        W = Train{i} * {libsvc([],(proxm([],'p',1)),1),libsvc([],(proxm([],'e',1)),1),libsvc([],(proxm([],'p',2)),1),libsvc([],(proxm([],'r',2.5)),1)};

        for j = 1:num_classifiers
            F(i,j+6,k) = Test{i}*W(j)*testc;
        end
    end
end
