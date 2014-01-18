clear;
delfigs;
prwaitbar off;

a = prnist([0:9],[1:40:1000]);
%show(a);

% Preprocessing of letter images
% See page 17 of Lab Course Manual
preproc = im_box([],0,1)*im_rotate*im_resize([],[128 128])*im_box([],1,0);
a = a*preproc;

% Calculate all the image features
features = im_features(a, a, 'all');

% Perform Principal Component Analysis
% w =       affine PCA mapping
% frac =    fraction of cumulative variance to retain

% Calculate distance matrix using statistics toolbox
distMatrix =    pdist(+features, 'euclidean');

% Perform Classical Multidimensional Scaling
%scaledMatrix =  cmdscale(distMatrix);

% The scaledMatrix already performed some kind of PCA
% We can use the best x-features and perform scatterplot

%scatterd(prdataset(scaledMatrix(:,[1 2]),getlab(a)))

% Perform Principal Component Analysis
%[w,frac] = pcam(scaledMatrix, 3);

%selFeatures = scaledMatrix(:,[1 2 3 4]);
%w = knnm(selFeatures, 2);


mapping = scalem(features, 'variance');
scaledData = features*mapping;

% Perform Principal Component Analysis
% Extraction 10 features gives ~95% of the variance
% I suppose this is the best number of features to extract
[w, frac] = pcam(scaledData, 10);



% Perform cross-validation to check performance
% 10-fold cross-validation

