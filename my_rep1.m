function a = my_rep1( m )
    
    % Preprocess the digits
    preproc = im_box([],0,1)*im_resize([],[(10) (13)],'bicubic')*im_box([],1,0);
    a = m*preproc;
    
    a = prdataset(a, getlabels(m));
    %a = prdataset(a);
    
end

