function [NumTrials, goodtrials] = remove_bad_trials(data, options)
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

arguments
    data double
    options.threshold = 10;
    options.method = 1;
end

thresh = options.threshold;

for iCh = 1:size(data, 1) % Iterate over channels
    tmp = squeeze(data(iCh, :, :)); % Extract trials for the current channel
    switch(options.method)
        case 1
            th = thresh*std(abs(tmp(:)))+mean(abs(tmp(:)));
            e = max(abs(tmp')); % Finds the maximum SINGLE point
            NumTrials(iCh) = length(find(e < th | e ~=0)); % Count the number of trials below the threshold
            goodtrials(iCh,:) = (e < th| e ~=0); % Get the indices of the good trials (below the threshold)
        case 2
            difftmp = (diff(tmp'));           
            NumTrials(iCh) = length(find(max(difftmp)<thresh)); % Count the number of trials below the threshold
            goodtrials(iCh,:) = max(difftmp)<thresh; % Get the indices of the good trials (below the threshold)
    end
end

end
