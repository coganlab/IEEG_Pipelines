function meanFreqChanOut = extractSpecNorm(spec, tw, etw)
% extractSpecNorm - Extracts the normalized mean frequency from spectrograms.
%
% Inputs:
%    spec - Spectrograms (cell array of size [1 x nChannels])
%    tw - Time window of interest [start_time, end_time]
%    etw - Extraction time window [start_time, end_time]
%
% Output:
%    meanFreqChanOut - Normalized mean frequency (nChannels x nFrequencies)
%
% Example:
%    spec = cell(10, 1); % Example spectrograms (cell array)
%    tw = [0, 10]; % Time window of interest
%    etw = [2, 8]; % Extraction time window
%    meanFreqChanOut = extractSpecNorm(spec, tw, etw); % Extract normalized mean frequency
%


tspec = linspace(tw(1), tw(2), size(spec{1}, 2)); % Time vector for the time window of interest
meanFreqChanOut = [];

for iChan = 1:length(spec)
    spec2Analyze = spec{iChan}; % Spectrogram of the current channel
    meanFreq = zeros(1, size(spec2Analyze, 3)); % Preallocate array for mean frequencies
    
    for iFreq = 1:size(spec2Analyze, 3)
        % Extract the spectrogram values within the extraction time window and calculate the mean
        meanFreq(iFreq) = mean2(squeeze(spec2Analyze(:, tspec >= etw(1) & tspec <= etw(2), iFreq)));
    end
    
    meanFreqChanOut(iChan, :) = meanFreq; % Store the mean frequencies for the current channel
end

end
