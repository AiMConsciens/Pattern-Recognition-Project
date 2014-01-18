function cmom = im_cmoment(im,p,q)
    
    % m_00
    rawm00 = im_moment(im, 0, 0);
    
    % [x_avg y_avg]
    centroids = [im_moment(im,1,0)/rawm00 , im_moment(im,0,1)/rawm00];
    
    % Central moment (mu_pq)
    cmom = sum(sum((([1:size(im,1)]-centroids(2))'.^q * ([1:size(im,2)]-centroids(1)).^p) .* im));
                 
end

