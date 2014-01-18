clear;
delfigs;
prwaitbar off;

% START DISABLED CODE %
a = prnist([0:9],[1:40:1000]);
%show(a);

%Preprocessing of letter images
%See page 17 of Lab Course Manual
preproc = im_box([],0,1)*im_rotate*im_resize([],[128 128])*im_box([],1,0);
a = a*preproc;

% Calculate all the image features
features = im_features(a, a, 'all');

% Compute scaling map (on the variance)
mapping = scalem(features, 'variance');
scaledData = features*mapping;

% Perform Principal Component Analysis
% Extraction 10 features gives ~95% of the variance
[mapping, frac] = pcam(scaledData, 10);
% w now contains an affine PCA mapping

% Apply PCA mapping to scaled data
pcaData = scaledData*mapping;
% END DISABLED DATA %

% Generate a test set
[trn, test, Itrn, Itest] = gendat(pcaData, 0.8);

% Parzen classifier
[trn_pz, smooth] = parzenc(trn);

% Test classifier error
[error, num_misclassified] = testc(test*trn_pz, 'crisp');




% for h = 0.1:0.1:2
%     % Parzen classifier
%     trn_pz = parzenc(trn, h);
% 
%     % Test classifier error
%     [error, num_misclassified] = testc(test*trn_pz, 'crisp');
%     
%     fprintf('Error of %f for h = %i \n', error, h);
%     
% end

% Perform cross-validation to check performance
% 10-fold cross-validation

% Possible good features:
%   - Contour direction (chain code)
%   - Gradient map
%   - Geometric moments 
%   - Hu's absolute orthogonal moment invariants
%   - Zernike moments
%   - Spline curve
%   - Fourier desciptors

% Other Ideas:
%   - Perform closing operation before starting
%   - !!! http://www.sciencedirect.com/science/article/pii/S0734189X87801834
