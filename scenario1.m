%% Load data
clc;
clear;
delfigs;
prwaitbar off;

% Load for each 10 digits (0-9) 25 objects
% out of the total dataset of 1000 images
%a = prnist(0:9,1:40:1000);
a = prnist(0:9,1:1000);
fprintf('Finished loading %i digits\n', length(a));

%% Simple preprocessing
fprintf('Preprocessing data set...\n');
preproc = im_box([],0,1)*im_rotate*im_resize([],[128 128])*im_box([],1,0);
a = a*preproc;

% Convert the entire dataset to image objects
%printf('Converting dataset to images...\n');
%im = data2im(a);
fprintf('Finished preprocessing!\n');

%% Converting to DIPimage and computing pixel features

% Filter size for closing operation
closingFilter = 2;

% Empty matrix for the results
d = 12; 
M = zeros(length(a), (d^2)+1);

fprintf('Busy calculating pixel features...\n');

for i = 1:length(a)
    % Convert to DIPimage object
    dip_img = data2im(a(i));
    
    % Obtain the label from the object
    label = getlabels(a(i));
    
    % Get the numeric value of the digit in the image
    numlab = str2num(label(7));

    % Perform closing operation on the image
    % image_out = closing(image_in,filterSize,filterShape)
    dip_img = closing(dip_img, closingFilter, 'elliptic');
    
    % Perform gray-value stretching
    dip_img = stretch(dip_img);
    
    % Resize the image
    scaleFactor = d/size(dip_img,1);
    dip_img = resample(dip_img, scaleFactor);
    
    % 5x5 matrix containing pixel values (0-255)
    mat_img = im2mat(dip_img);
    
    % Populate the feature matrix
    M(i, 1) = numlab;   % First column is label
    
    for j = 0:(d-1)
        istart = (d*j)+2;
        iend = istart+(d-1);
        
        %fprintf('[%i, %i]\n', istart, iend); 
        M(i, istart:iend) = mat_img(j+1, 1:d);
    end
    
    %fprintf('Finished processing image %i...\n', i);
    
end

csvfilename = sprintf('Data/pxfeat_%iobj_%ipx_%iclsflt_stretch.csv', length(a), d, closingFilter);
%csvfilename = sprintf('Data/pxfeat_%iobj_%ipx_%iclsflt_nostretch.csv', length(a), d, closingFilter);
%csvfilename = sprintf('Data/pxfeat_%iobj_%ipx_noclose.csv', length(a), d);
%csvfilename = sprintf('Data/pxfeat_%iobj_%ipx_noclose_nostretch.csv', length(a), d);

csvwrite(csvfilename, M);
fprintf('Resuts saved to %s\n', csvfilename);
fprintf('All done!\n');

%% Classification

% Labels are in the first column
A = prdataset(M(:,2:144), M(:,1));

%mapping = scalem(A, 'variance');
%scaledData = A*mapping;

% Perform Principal Component Analysis
% 22 features corresponds to ~95% of the data
% Therefore applying PCA is not necessary?!
%[mapping, frac] = pcam(scaledData, 22);

% Apply PCA mapping to scaled data
%pcaData = scaledData*mapping;
% END DISABLED DATA %

% Generate a test set
[trn, tst, index_trn, index_tst] = gendat(A, 0.9);

w = fisherc(trn);
e = tst*w*testc

%% Classification V2

prmemory(100000000);

M = csvread('Data/pxfeat_10000obj_6px_10clsflt.csv');
A = prdataset(M(:,2:37), M(:,1)); % 26 or 37

%% Distance Matrix and PCA
mapping = scalem(A, 'variance');
scaledData = A*mapping;

[mapping, frac] = pcam(scaledData, 40);

% Apply PCA mapping to scaled data
pcaData = scaledData*mapping;
%%
[trn, tst, index_trn, index_tst] = gendat(A, 0.95);

% qdc = worst every time
% parzenc = best every time (closely followed by knnc)
W = trn*{knnc([],3),parzenc};
E = tst*W*testc

%W = svc(trn, proxm([],'r',1));

% Lowest error rate (still ~14%) achieved
% by using the 5x5 pixel feature data set and 
% using the parzenc classifier. PCA does not seem
% to be necessary.






