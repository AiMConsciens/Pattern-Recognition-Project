function M = im_moment(im, p, q)
    
    % m_pq in the paper
    M = sum(sum(((1:size(im,1))'.^p * (1:size(im,2)).^q) .* im));

end

