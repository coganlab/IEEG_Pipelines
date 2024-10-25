function wCarIeeg = weightedCar(ieeg)
% weightedCar - Applies the weighted Common Average Reference (CAR) segmentation to intracranial EEG data.
%
% Syntax:  car = weightedCar(ieeg)
%
% Inputs:
%    ieeg - Intracranial EEG data (channels x  time)
%
% Outputs:
%    wCarIeeg - weighted CAR EEG data (channels x  time)
%
% Example:
%    ieeg = randn(16, 1000); % Example intracranial EEG data
%    wCarIeeg = weightedCar(ieeg); % Apply the CAR segmentation
%

n = size(ieeg, 1); % Number of channels


ieegmean = mean(ieeg, 1); % Compute the mean across time for each channel
corrIeegMean = ieegmean * ieegmean'; % Compute the correlation of the mean signal
    
for i = 1:n % Iterate through channels
    wCarIeeg(i,  :) = ieeg(i, :) - (ieeg(i, :) * ieegmean') .* ieegmean ./ corrIeegMean;
    % Apply the CAR segmentation by subtracting the weighted average of the mean signal from each channel's data
end


end
