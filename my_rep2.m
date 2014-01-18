function a = my_rep2( m )

    % Simple preprocessing
    m = m*im_resize([],[128 128])*im_box([],1,0);

    % Pixel features of size d*d
    d = 0; 
    
    % Converting to DIPimage and computing pixel features
    % Empty matrix for the results
    M = zeros(length(m), d^2 + 25);
    
    % Obtain the label from the object
    labels = getlabels(m);

    for i = 1:length(m)
        % Convert to DIPimage object
        dip_img = data2im(m(i));

        % Perform closing operation on the image
        % image_out = closing(image_in,filterSize,filterShape)
        %dip_img = closing(dip_img, close_size);
        dip_img = smooth(dip_img);
        
        % Perform gray-value stretching
        %dip_img = stretch(dip_img);

        % Threshold the image to obtain binary represtation
        binary = threshold(dip_img);
        msr = measure(binary, dip_img, ({'Size','Radius', 'Inertia', 'Mu', 'ConvexArea', 'Center', 'Gravity','CartesianBox','Perimeter','Feret','P2A','Convexity','Sum','Mean','CCBendingEnergy','MajorAxes'}));

        % Default we set it to the first object
        j = 1;

        obj_count = size(msr, 1);

        % Some threshold measures have multiple entries since there
        % are multiple objects. However the first one is also the largest
        if (obj_count ~= 1)
            % More than one object in the image
            % We select on the largest object
            [val, j] = max(msr.Size);
            %fprintf('Multiple objects found in object %i with label %i...\n', i, numlab);

            % Selected with the max size has index j
        end
        
        M(i,1)  = obj_count;         % number of objects
        M(i,2)  = sum(msr.Size);     % size of all objects
        M(i,3)  = msr.Radius(2,j);   % avg. radius of largest object
        M(i,4)  = msr.Center(1,j);   % CenterX of largest object
        M(i,5)  = msr.Center(2,j);   % CenterY of largest object
        M(i,6)  = msr.Gravity(1,j);  % GravityX of largest object
        M(i,7)  = msr.Gravity(2,j);  % GravityY of largest object
        M(i,8)  = msr.Inertia(1,j);  % InertiaM1 of largest object
        M(i,9)  = msr.Inertia(2,j);  % InertiaM2 of largest object
        M(i,10) = msr.Mu(1,j);       % Mu_xx of largest object
        M(i,11) = msr.Mu(2,j);       % Mu_xy of largest object
        M(i,12) = msr.Mu(3,j);       % Mu_yy of largest object
        M(i,13) = msr.ConvexArea(j); % ConvexArea of largest object
        M(i,14) = msr.CartesianBox(1,j); % CartBoxX around largest object
        M(i,15) = msr.CartesianBox(2,j); % CartBoxY around largest object
        M(i,16) = msr.CCBendingEnergy(1,j); % CCBendingEnergy of largest object
        M(i,17) = msr.Convexity(j);  % Convexity of largest object
        M(i,18) = msr.Feret(1,j);    % FeretMax of largest object
        M(i,19) = msr.Feret(2,j);    % FeretMin of largest object
        M(i,20) = msr.Feret(4,j);    % FeretAngleMax of largest object
        M(i,21) = msr.Feret(5,j);    % FeretAngleMin of largest object
        M(i,22) = msr.Mean(j);       % Mean object intensity
        M(i,23) = msr.P2A(j);        % Circularity
        M(i,24) = msr.Perimeter(j);  % Perimeter (chain-code method)
        M(i,25) = msr.Sum(j);        % Sum of all pixels
        
        % Resize the image
        if (d > 0)
            scaleFactor = d/size(dip_img,1);
            dip_img = smooth(dip_img, 2);
            dip_img = resample(dip_img, scaleFactor);

            % d*d matrix containing pixel values (0-255)
            mat_img = im2mat(dip_img);

            for j = 0:(d-1)
                istart = 25+(d*j)+1;
                iend = istart+(d-1);

                %fprintf('[%i, %i]\n', istart, iend); 
                M(i, istart:iend) = mat_img(j+1, 1:d);
            end
        end

    end

    %   row 1-25 =    features
    %   row 25-x =    pixel values
    a = prdataset(M, labels);

end

