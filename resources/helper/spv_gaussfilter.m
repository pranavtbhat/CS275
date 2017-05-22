function adj=spv_gaussfilter(edges,points, sigma)
%========================================================================%
%Constants
EPSILON = 1e-5;
if nargin < 3
    sigma = 1;
end

%Compute geomDistances, if desired
geomDistances=sqrt(abs(sum((points(edges(:,1),:)- ...
        points(edges(:,2),:)).^2,2)));

%Compute Gaussian weights
weights=exp(-(geomDistances.^2)/(2*sigma*sigma)) + EPSILON;
adj = adjacency(edges, weights);
[row col] = size(adj);
for i=1:row
    adj(i,:) = normalize(adj(i,:));
end