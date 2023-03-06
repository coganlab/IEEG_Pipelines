function [h, crit_p, adj_p]=cluster_correction_method(p, alpha, method)
% p: 2D connected matrix of p-values
% alpha: significance level for multiple comparisons correction
% method: established correction method (e.g. 'fdr', 'bonferroni')
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

% Correct for multiple comparisons using the established method
if strcmp(method, 'bonferroni')
crit_p = alpha / (n*m);
elseif strcmp(method, 'fdr')
crit_p = m * alpha / (2 * sum(p < alpha/2));
else
error('Unrecognized correction method.');
end
adj_p = min(1, m * p ./ crit_p);

% Indicate significant clusters
for c = 1:clusters.NumObjects
if min(p(clusters.PixelIdxList{c})) <= crit_p
h(clusters.PixelIdxList{c}) = 1;
end
end
end