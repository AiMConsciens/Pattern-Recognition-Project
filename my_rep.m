% FUNCTION: my_rep(m)
%
% INPUT:
%   m = NIST measurement
% OUTPUT:
%   a = resulting dataset
function a = my_rep( m )
    
    fprintf('Preprocessing data set...\n');
    preproc = im_box([],0,1)*im_rotate*im_resize([],[128 128])*im_box([],1,0);
    m = m*preproc;
    fprintf('Finished preprocessing!\n');
    
    % Resulting image size = d*d pixels
    d = 5; 
    M = zeros(length(m), (d^2)+1);

    fprintf('Busy calculating pixel features...\n');

    for i = 1:length(m)
        % Convert to DIPimage object
        dip_img = data2im(m(i));

        % Obtain the label from the object
        label = getlabels(m(i));

        % Get the numeric value of the digit in the image
        numlab = str2num(label(7));

        % Perform closing operation on the image
        % image_out = closing(image_in,filterSize,filterShape)
        dip_img = closing(dip_img, 15, 'elliptic');

        % Perform gray-value stretching
        dip_img = stretch(dip_img);

        % Resize the image
        scaleFactor = d/size(dip_img,1);
        dip_img = resample(dip_img, scaleFactor);

        % 5x5 matrix containing pixel values (0-255)
        mat_img = im2mat(dip_img);

        % Populate the feature matrix
        M(i, 1) = numlab;   % First column is label

        for j = 0:4
            istart = (5*j)+2;
            iend = istart+4;

            %fprintf('[%i, %i]\n', istart, iend); 
            M(i, istart:iend) = mat_img(j+1, 1:5);
        end

        %fprintf('Finished processing image %i...\n', i);

    end

    fprintf('All done!\n');
    
    % Resulting dataset
    a = prdataset(M(:,2:26), M(:,1));
    
end

