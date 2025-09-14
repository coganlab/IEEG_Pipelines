function [ieegFilt, meanIeeg] = carFilterImpedance(ieeg, badChan)
% carFilterImpedance - Common Average Reference (CAR) filter with impedance handling for intracranial EEG data.
%
% Syntax:  [ieegFilt, meanIeeg] = carFilterImpedance(ieeg, badChan)
%
% Inputs:
%    ieeg - Intracranial EEG data (channels x trials x time)
%    badChan - Indices of bad channels to be excluded from the CAR filter (1 x nBadChan)
%
% Outputs:
%    ieegFilt - Intracranial EEG data after applying the CAR filter
%    meanIeeg - Mean of the filtered EEG data across good channels
%
% Example:
%    ieeg = randn(16, 10, 1000); % Example intracranial EEG data
%    badChan = [3, 7]; % Indices of bad channels
%    [ieegFilt, meanIeeg] = carFilterImpedance(ieeg, badChan); % Apply the CAR filter with impedance handling
%


% Identify good channels by excluding the indices of bad channels
goodChan = setdiff(1:size(ieeg, 1), badChan);

% Subtract the mean of each channel across trials from the EEG data (excluding bad channels)
ieegFilt = ieeg - mean(ieeg(goodChan, :, :), 1,"omitnan");

% Calculate the mean of the filtered EEG data across good channels
meanIeeg = squeeze(mean(ieeg(goodChan, :, :), 1,"omitnan"));

end
