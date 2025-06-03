function [NumTrials, goodtrials] = remove_bad_trials_old(data, threshold)
% remove_bad_trials - Removes bad trials based on a threshold detection.
%
% Syntax: [NumTrials, goodtrials] = remove_bad_trials(data, threshold)
%
% Inputs:
%   data        - Electrodes x Trials x Samples matrix
%   threshold   - Threshold of standard deviation to remove noisy trials
%
% Outputs:
%   NumTrials   - Number of good trials in each channel
%   goodtrials  - Trial indices after removing bad trials
%
% Example:
%   data = randn(10, 100, 500); % Example data with 10 electrodes, 100 trials, and 500 samples
%   threshold = 3; % Threshold of 3 standard deviations
%   [NumTrials, goodtrials] = remove_bad_trials(data, threshold); % Remove bad trials


thresh = threshold;

for iCh = 1:size(data, 1) % Iterate over channels
    tmp = squeeze(data(iCh, :, :)); % Extract trials for the current channel
    tmp = detrend(tmp); % Detrend the trials
    sd = std(tmp(:)); % Calculate the standard deviation of the detrended trials
    e = max(abs(tmp')); % Find the maximum absolute value for each trial (single point)
    if thresh < 100 % Artifact threshold is in terms of standard deviations
        th = thresh * sd; % Calculate the threshold as a multiple of the standard deviation
    else
        th = 10 * thresh; % Artifact threshold is in terms of uV, accounting for preamp gain
    end
    NumTrials(iCh) = length(find(e < th)); % Count the number of trials below the threshold
    goodtrials{iCh} = find(e < th); % Get the indices of the good trials (below the threshold)
end

end
