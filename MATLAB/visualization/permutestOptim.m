function [clusters, p_values, t_sums, permutation_distribution] = permutestOptim(trial_group_1, trial_group_2, dependent_samples, ...
    p_threshold, num_permutations, two_sided, num_clusters)
% Permutation test for dependent or independent measures of 1-D or 2-D data.
% Based on Maris & Oostenveld 2007 for 1-D and 2-D vectors. The test
% statistic is T-Sum - the total of t-values within a cluster of contingent
% above-threshold data points.
% See: Maris, E., & Oostenveld, R. (2007). Nonparametric statistical
% testing of EEG-and MEG-data. Journal of Neuroscience Methods, 164(1),
% 177–190. https://doi.org/10.1016/j.jneumeth.2007.03.024

% Set optional arguments:
if nargin < 7 || isempty(num_clusters)
    num_clusters = inf;
end
if nargin < 6 || isempty(two_sided)
    two_sided = false;
end
if nargin < 5 || isempty(num_permutations)
    num_permutations = 10^4;
end
if nargin < 4 || isempty(p_threshold)
    p_threshold = 0.05;
end
if nargin < 3 || isempty(dependent_samples)
    dependent_samples = true;
end
if nargin < 2
    error('Not enough input arguments');
end

% Check input dimensions:
[sample_size_1, num_trials_1] = size(trial_group_1);
[sample_size_2, num_trials_2] = size(trial_group_2);

if dependent_samples
    num_trials = num_trials_1;
    if sample_size_1 ~= sample_size_2 || num_trials_1 ~= num_trials_2
        error('Size of all dimensions should be identical for two dependent samples');
    end
else
    if sample_size_1 ~= sample_size_2
        error('Size of all dimensions but the last one (corresponding to the number of trials) should be identical for two independent samples');
    end
end

% Check that number of requested permutations is possible:
if dependent_samples
    max_num_permutations = 2^num_trials;
else
    max_num_permutations = nchoosek(num_trials_1 + num_trials_2, num_trials_1);
end

if num_permutations > max_num_permutations
    warning('With %d trials, only %d permutations are possible. Using this value instead of %d.',...
        num_trials, max_num_permutations, num_permutations);
    num_permutations = max_num_permutations;
end

% Initialize output variables
max_num_clusters = min(num_clusters, sample_size_1 * sample_size_2);
clusters = cell(1, max_num_clusters);
p_values = ones(1, max_num_clusters);
t_sums = zeros(1, max_num_clusters);
permutation_distribution = zeros(num_permutations, 1);

% Compute t-value threshold from p-value threshold
if dependent_samples
    tThreshold = abs(tinv(p_threshold, num_trials - 1));
else
    tThreshold = abs(tinv(p_threshold, num_trials_1 + num_trials_2 - 1));
end

% PRODUCE PERMUTATION VECTORS
if num_permutations < max_num_permutations / 1000
    if dependent_samples
        permutation_vectors = round(rand(num_trials, num_permutations)) * 2 - 1;
    else
        permutation_vectors = ones(num_trials_1 + num_trials_2, num_permutations);
        for p = 1:num_permutations
            idx = randperm(num_trials_1 + num_trials_2, num_trials_2);
            permutation_vectors(idx, p) = 2;
        end
    end
else
    if dependent_samples
        permutation_vectors = NaN(num_trials, num_permutations);
        rndB = dec2bin(randperm(2^num_trials, num_permutations) - 1);
        nBits = size(rndB, 2);
        if nBits < num_trials
            rndB(:, (num_trials - nBits + 1):num_trials) = rndB;
            rndB(:, 1:num_trials - nBits) = '0';
        end
        for ii = 1:numel(permutation_vectors)
            permutation_vectors(ii) = str2double(rndB(ii)) * 2 - 1;
        end
    else
        permutation_vectors = ones(num_trials_1 + num_trials_2, num_permutations);
        idx_matrix = nchoosek(1:(num_trials_1 + num_trials_2), num_trials_2);
        idx_matrix = idx_matrix(randperm(size(idx_matrix, 1), num_permutations), :);
        for p = 1:num_permutations
            permutation_vectors(idx_matrix(p, :), p) = 2;
        end
    end
end

% RUN PRIMARY PERMUTATION
t_value_vector = zeros(sample_size_1, sample_size_2);
if dependent_samples
    for ii = 1:sample_size_2
        t_value_vector(:, ii) = simpleTTest(squeeze(trial_group_1(:, ii, :)) - squeeze(trial_group_2(:, ii, :)), 0);
    end
else
    all_trials = cat(3, trial_group_1,trial_group_2);
    for ii = 1:sample_size_2
        t_value_vector(:, ii) = simpleTTest2(squeeze(all_trials(:, ii, permutation_vectors == 1))', squeeze(all_trials(:, ii, permutation_vectors == 2))');
    end
end

% Find the above-threshold clusters:
CC = bwconncomp(t_value_vector > tThreshold, 4);
cMapPrimary = zeros(size(t_value_vector));
tSumPrimary = zeros(CC.NumObjects, 1);
for i = 1:CC.NumObjects
    cMapPrimary(CC.PixelIdxList{i}) = i;
    tSumPrimary(i) = sum(t_value_vector(CC.PixelIdxList{i}));
end
if two_sided % Also look for negative clusters
    n = CC.NumObjects;
    CC = bwconncomp(t_value_vector < -tThreshold, 4);
    for i = 1:CC.NumObjects
        cMapPrimary(CC.PixelIdxList{i}) = n + i;
        tSumPrimary(n + i) = sum(t_value_vector(CC.PixelIdxList{i}));
    end
end

% Sort clusters:
[~, tSumIdx] = sort(abs(tSumPrimary), 'descend');
tSumPrimary = tSumPrimary(tSumIdx);

% RUN PERMUTATIONS
if dependent_samples
    trial_group_diff = trial_group_1 - trial_group_2;
else
    trial_group_diff = all_trials(:, :, permutation_vectors == 1) - all_trials(:, :, permutation_vectors == 2);
end

for p = 1:num_permutations
    if dependent_samples
        for ii = 1:sample_size_2
            t_value_vector(:, ii) = simpleTTest(squeeze(trial_group_diff(:, ii, :)) .* permutation_vectors(:, p), 0);
        end
    else
        p_trial_group_1 = all_trials(:, :, permutation_vectors(:, p) == 1);
        p_trial_group_2 = all_trials(:, :, permutation_vectors(:, p) == 2);
        for ii = 1:sample_size_2
            t_value_vector(:, ii) = simpleTTest2(squeeze(p_trial_group_1(:, ii, :))', squeeze(p_trial_group_2(:, ii, :))');
        end
    end
    
    % Find clusters:
    CC = bwconncomp(t_value_vector > tThreshold, 4);
    tSum = zeros(CC.NumObjects, 1);
    for i = 1:CC.NumObjects
        tSum(i) = sum(t_value_vector(CC.PixelIdxList{i}));
    end
    if two_sided % Also look for negative clusters
        n = CC.NumObjects;
        CC = bwconncomp(t_value_vector < -tThreshold, 4);
        for i = 1:CC.NumObjects
            tSum(n + i) = sum(t_value_vector(CC.PixelIdxList{i}));
        end
    end
    if isempty(tSum)
        permutation_distribution(p) = 0;
    else
        [~, idx] = max(abs(tSum));
        permutation_distribution(p) = tSum(idx);
    end
end

% DETERMINE SIGNIFICANCE
for clustIdx = 1:min(num_clusters, length(tSumPrimary))
    if two_sided
        ii = sum(abs(permutation_distribution) >= abs(tSumPrimary(clustIdx)));
    else
        ii = sum(permutation_distribution >= tSumPrimary(clustIdx));
    end
    clusters{clustIdx} = cMapPrimary == tSumIdx(clustIdx);
    p_values(clustIdx) = (ii + 1) / (num_permutations + 1);
    t_sums(clustIdx) = tSumPrimary(clustIdx);
end

% Return regular arrays if only one cluster is requested
if num_clusters == 1
    clusters = clusters{1};
    p_values = p_values(1);
    t_sums = t_sums(1);
end
end

function t = simpleTTest(data, mu)
% Simple one-sample t-test
n = size(data, 1);
mean_diff = mean(data, 1) - mu;
std_diff = std(data, [], 1) / sqrt(n);
t = mean_diff ./ std_diff;
end

function t = simpleTTest2(data1, data2)
% Simple independent-samples t-test
n1 = size(data1, 1);
n2 = size(data2, 1);
mean_diff = mean(data1, 1) - mean(data2, 1);
std_diff = sqrt(var(data1, [], 1) / n1 + var(data2, [], 1) / n2);
t = mean_diff ./ std_diff;
end
