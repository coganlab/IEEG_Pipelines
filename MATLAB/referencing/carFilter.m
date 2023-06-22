function ieegFilt = carFilter(ieeg)
% carFilter - Common Average Reference (CAR) filter for intracranial EEG data.
%
% Syntax:  ieegFilt = carFilter(ieeg)
%
% Inputs:
%    ieeg - Intracranial EEG data (channels x trials x time)
%
% Outputs:
%    ieegFilt - Intracranial EEG data after applying the CAR filter
%
% Example:
%    ieeg = randn(16, 10, 1000); % Example intracranial EEG data
%    ieegFilt = carFilter(ieeg); % Apply the CAR filter
%


% Subtract the mean of each channel across trials from the EEG data
ieegFilt = ieeg - mean(ieeg, 1);

end
