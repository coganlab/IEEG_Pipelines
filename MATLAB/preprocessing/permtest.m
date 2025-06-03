function p = permtest(sample1, sample2, numperm)
% permtest - Perform one-sided permutation test to compare the means of two samples.
%
% Syntax: p = permtest(sample1, sample2, numperm)
%
% Inputs:
%   sample1     - First sample data (1 x n1) array (signal of interest)
%   sample2     - Second sample data (1 x n2) array (baseline signal)
%   numperm     - Number of permutations to perform
%
% Outputs:
%   p           - p-value indicating the significance of the difference between the means
%
% Example:
%   sample1 = [1, 2, 3, 4, 5]; % Example first sample
%   sample2 = [6, 7, 8, 9, 10]; % Example second sample
%   p = permtest(sample1, sample2, 1000); % Perform one-sided permutation test with 1000 permutations

samples = [sample1 sample2]; % Combine the two samples
samplediff = mean(sample1) - mean(sample2); % Calculate the difference between the means of the samples
sampdiffshuff = zeros(1, numperm); % Initialize an array to store shuffled sample differences

for n = 1:numperm
    sampshuff = samples(randperm(length(samples))); % Shuffle the combined samples
    sampdiffshuff(n) = mean(sampshuff(1:length(sampshuff)/2)) - mean(sampshuff(length(sampshuff)/2+1:end)); % Calculate the difference between means for the shuffled samples
end

p = length(find(sampdiffshuff > samplediff)) / numperm; % Calculate the p-value as the proportion of shuffled sample differences greater than the observed difference

end
