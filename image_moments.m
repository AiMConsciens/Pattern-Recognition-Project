clear;
delfigs;
prwaitbar off;

% Load NIST images
a = prnist([0:9],[1:40:1000]);

%Preprocessing of letter images
%See page 17 of Lab Course Manual
preproc = im_box([],0,1)*im_rotate*im_resize([],[128 128],'bilinear')*im_box([],1,0);
a = a*preproc;

% Convert the entire dataset to image objects
im = data2im(a);

% Empty matrix for the results
M = zeros(length(im), 13);

for i = 1:length(im)
    % Convert to DIPimage object
    dip_img = data2im(a(i));
    
    % Obtain the label from the object
    label = getlabels(a(i));
    
    % Get the numeric value of the digit in the image
    numlab = str2num(label(7));

    % Perform closing operation on the image
    % image_out = closing(image_in,filterSize,filterShape)
    dip_img = closing(dip_img, 10, 'elliptic');
    
    % Perform gray-value stretching
    dip_img = stretch(dip_img);
    
    % Threshold the image to obtain binary represtation
    binary = threshold(dip_img);
    msr = measure(binary, dip_img, ({'Size', 'Radius', 'Inertia', 'Mu', 'ConvexArea', 'Center', 'Gravity'}));
    
    % Default we set it to the first object
    j = 1;
    
    % Some threshold measures have multiple entries since there
    % are multiple objects. However the first one is also the largest
    if (size(msr, 1) ~= 1)
        % More than one object in the image
        % We select on the largest object
        [val, j] = max(msr.Size);
        fprintf('Multiple objects found, [%d,%d] selected %i\n', msr.Size(1), msr.Size(2), j);
        
        % Selected with the max size has index j
        
    end
    
    % These moments are very (!) large
    mu_21 = im_cmoment(+a(i), 2, 1);
    mu_12 = im_cmoment(+a(i), 1, 2);
    mu_22 = im_cmoment(+a(i), 2, 2);
     
    % Write matrix in the following format
    % [index label size gravityX gravityY intertia_m1 intertia_m2 mu_xx mu_yy mu_xy]
    M(i,:) = [i numlab msr.Size(j) msr.Gravity(1, j) msr.Gravity(2,j) msr.Inertia(1,j) msr.Inertia(2,j) msr.Mu(1,j) msr.Mu(3,j) msr.Mu(2,j) mu_21 mu_12 mu_22];
    
    
end

A = prdataset(M(:,4:end), M(:,2));
fprintf('Generated PR dataset A with %i objects and %i features\n', size(M,1), size(M,2)-2);

[train1, test1, Itrain1, Itrain2] = gendat(A, 0.8);

[W,R] = featself(A, 'NN', 0);

W1 = knnc(train1);

E1 = test1*W1*testc;
disp(E1);

% Save measurements to CSV file
c = clock;
filename = sprintf('measurements_%d%d%d_%d%d.csv', c(1), c(2), c(3), c(4), c(5));
csvwrite(strcat(filename), M);

% TODO:
%   - Downscale the images to perform faster
%   - Perform good calculation of Zernike moments
%   - Perform automatic testing of different classifiers


% Zernike moments + kNN classifier works best



% Measurement Features to use in MEASURE:
%  
%                 Name   Description
%        ------------------------------------------------------------------------
%                 Size - number of object pixels
%         CartesianBox - cartesian box size of the object in all dimensions
%              Minimum - minimum coordinates of the object
%              Maximum - maximum coordinates of the object
%            Perimeter - length of the object perimeter  (chain-code method)
%          SurfaceArea - surface area of object (3D)
%                Feret - maximum and minimum object diameters (2D)
%               Radius - statistics on radius of object (chain-code method)
%           ConvexArea - area of the convex hull (2D)
%      ConvexPerimeter - perimeter of the convex hull (2D)
%                  P2A - circularity of the object (2D & 3D)
%       PodczeckShapes - Podczeck shape descriptors (2D)
%            Convexity - area fraction of convex hull covered by object (2D)
%                  Sum - sum of object intensity (=mass) *
%                 Mass - mass of object (=sum of object intensity) *
%                 Mean - mean object intensity *
%               StdDev - standard deviation of object intensity *
%             Skewness - skewness (gamma_1) of object intensity *
%       ExcessKurtosis - excess Kurtosis (gamma_2) of object intensity *
%               MaxVal - maximum object intensity *
%               MinVal - minimum object intensity *
%               Center - coordinates of the geometric mean of the object
%              Gravity - coordinates of the center-of-mass of the grey-value object *
%              Inertia - moments of inertia of binary object
%          GreyInertia - grey-weighted moments of inertia of object *
%                   Mu - elements of the inertia tensor
%               GreyMu - elements of the grey-weighted inertia tensor *
%      CCBendingEnergy - bending energy of object perimeter (chain-code method)
%       DimensionsCube - extent along the principal axes of a cube
%   GreyDimensionsCube - extent along the principal axes of a cube (grey-weighted) *
%  DimensionsEllipsoid - extent along the principal axes of an ellipsoid
% GreyDimensionsEllipsoid - extent along the principal axes of an elliposid (grey-weighted)*
%            MajorAxes - principal axes of an object
%        GreyMajorAxes - principal axes of an object (grey-weigheted) *
%  
% Measurements marked with an * require a grey-value input image.