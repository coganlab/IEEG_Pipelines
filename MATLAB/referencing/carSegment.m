function car = carSegment(ieeg)
% carSegment - Applies the Common Average Reference (CAR) segmentation to intracranial EEG data.
%
% Syntax:  car = carSegment(ieeg)
%
% Inputs:
%    ieeg - Intracranial EEG data (channels x trials x time)
%
% Outputs:
%    car - CAR segmented EEG data (channels x trials x time)
%
% Example:
%    ieeg = randn(16, 10, 1000); % Example intracranial EEG data
%    car = carSegment(ieeg); % Apply the CAR segmentation
%

n = size(ieeg, 1); % Number of channels
c = 1:n;

for tr = 1:size(ieeg, 2)
    tr
    
    ieegtrial = squeeze(ieeg(:, tr, :)); % Extract EEG data for the current trial
    ieegmean = mean(ieegtrial, 1); % Compute the mean across time for each channel
    corrIeegMean = ieegmean * ieegmean'; % Compute the correlation of the mean signal
    
    for i = 1:n % Iterate through channels
        car(i, tr, :) = ieegtrial(i, :) - (ieegtrial(i, :) * ieegmean') .* ieegmean ./ corrIeegMean;
        % Apply the CAR segmentation by subtracting the weighted average of the mean signal from each channel's data
    end
end

end
