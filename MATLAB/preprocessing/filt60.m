function ieegfilt = filt60(ieeg, fs)
% filt60 - Apply a notch filter at 60 Hz to remove power line interference.
%
% Syntax: ieegfilt = filt60(ieeg, fs)
%
% Inputs:
%   ieeg    - Input EEG signal (channels x samples)
%   fs      - Sampling frequency of the IEEG signal (in Hz)
%
% Output:
%   ieegfilt- Filtered EEG signal with power line interference removed
%
% Example:
%   ieegSignal = randn(8, 1000); % IEEG signal with 8 channels and 1000 samples
%   fs = 1000; % Sampling frequency of 1000 Hz
%   filteredSignal = filt60(ieegSignal, fs);
%
    f60 = 60; % Power line frequency
    q = 10; % Quality factor
    bw = round((f60 / (fs/2)) / q); % bandwidth
    
    % Design the notch filter
    [filtb, filta] = iircomb(round(fs / f60), bw, 'notch');
    
    % Apply the notch filter to each channel of the EEG signal
    ieegfilt = zeros(size(ieeg));
    for i = 1:size(ieeg, 1)
        ieegfilt(i, :) = filtfilt(filtb, filta, ieeg(i, :));
    end
end
