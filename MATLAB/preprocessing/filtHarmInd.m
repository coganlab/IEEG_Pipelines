% filtHarmInd - Apply a bandstop filter to remove a specific frequency component from an IEEG signal.
%
% Syntax: ieegfilt = filtHarmInd(ieeg, fs, fFilt)
%
% Inputs:
%   ieeg    - Input IEEG signal (channels x samples)
%   fs      - Sampling frequency of the EEG signal (in Hz)
%   fFilt   - Frequency component to filter (in Hz)
%
% Output:
%   ieegfilt- Filtered EEG signal with the specified frequency component removed
%
% Example:
%   ieegSignal = randn(8, 1000); % IEEG signal with 8 channels and 1000 samples
%   fs = 1000; % Sampling frequency of 1000 Hz
%   fFilt = 60; % Frequency component to filter (in Hz)
%   filteredSignal = filtHarmInd(ieegSignal, fs, fFilt);


function ieegfilt = filtHarmInd(ieeg, fs, fFilt)
    % Create a bandstop filter to remove the specified frequency component
    d = designfilt('bandstopiir', 'FilterOrder', 20, ...
                   'HalfPowerFrequency1', fFilt-1, 'HalfPowerFrequency2', fFilt+1, ...
                   'DesignMethod', 'butter', 'SampleRate', fs);
    
    % Apply the bandstop filter to each channel of the EEG signal
    ieegfilt = zeros(size(ieeg));
    for i = 1:size(ieeg, 1)
        ieegfilt(i, :) = filtfilt(d, ieeg(i, :));
    end
end
