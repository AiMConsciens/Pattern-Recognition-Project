
a = prnist([0:9],[1:1:1000]);
preproc = im_box([],0,1)*im_resize([],[(10) (13)],'bicubic')*im_box([],1,0);
a = a*preproc;
pr_a = prdataset(a);

mapping = scalem(pr_a, 'c-mean');

%scale the original pixel-datasets
scaledPixels = pr_a*mapping;

% Perform Principal Component Analysis
% Extraction features gives ~95% of the variance

[mapping, frac] = pcam(scaledPixels, 51);
pcaData = scaledPixels*mapping;
[Train,Test] = gendat(pcaData,0.8);

W = Train * libsvc([],(proxm([],'r',2.9)),1);
E = Test*W*testc;