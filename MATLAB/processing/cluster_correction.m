function [h, crit_p, adj_p]=cluster_correction(p,alpha)
% p: 2D connected matrix of p-values
% alpha: significance level for multiple comparisons correction
% h: binary matrix indicating significant clusters
% crit_p: cluster-corrected p-value threshold
% adj_p: adjusted p-values for each data point

% Find size of matrix
[n, m] = size(p);

% Transform p-values to Z-scores
z = -norminv(p);

% Find clusters of significant Z-scores
clusters = bwconncomp(z > norminv(1-alpha/2));

% Initialize output
h = zeros(n, m);

% Compute cluster-level statistics
cluster_pvals = zeros(clusters.NumObjects,1);
for c = 1:clusters.NumObjects
    cluster_pvals(c) = min(p(clusters.PixelIdxList{c}));
end

% Correct for multiple comparisons using the minimum cluster-level p-value
crit_p = max(alpha / (n*m), min(cluster_pvals));
adj_p = min(1, m * p ./ crit_p);

% Indicate significant clusters
for c = 1:clusters.NumObjects
    if min(p(clusters.PixelIdxList{c})) <= crit_p
        h(clusters.PixelIdxList{c}) = 1;
    end
end
end



