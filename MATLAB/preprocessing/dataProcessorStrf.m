function [XMatrix, YMatrix] = dataProcessorStrf(SMel, ieeg, fsIeeg, lag)

% dataProcessorStrf - Process data for spectrotemporal receptive field (STRF) analysis.
%
% Syntax: [XMatrix, YMatrix] = dataProcessorStrf(SMel, ieeg, fsIeeg, lag)
%
% Inputs:
%   SMel    - Spectrogram matrix (trial x frequency x time)
%   ieeg    - Intracranial EEG data (channels x trial x time)
%   fsIeeg  - Sampling frequency of the intracranial EEG data
%   lag     - Time lag in seconds
%
% Outputs:
%   XMatrix - Input matrix for STRF analysis (samples x frequency x lag)
%   YMatrix - Output matrix for STRF analysis (samples x channels)

    % Initialize variables for storing input and output matrices
    XMatrix = [];
    YMatrix = [];
    
    % Loop over each trial
    for iTrial = 1:size(ieeg, 2)
        SMelTemp = squeeze(SMel(iTrial, :, :));
        ieegTemp = squeeze(ieeg(:, iTrial, :))';
        
        % Check the dimensions of SMelTemp and ieegTemp
        size(SMelTemp)
        size(ieegTemp)
        
        % Loop over each time point within the trial
        for iTime = 1:(size(ieegTemp, 2) - (lag * fsIeeg - 1))
            % Check if the values in SMelTemp are finite
            if isfinite(SMelTemp(:, iTime + lag * fsIeeg - 1))
                YMatrix = [YMatrix, ieegTemp(:, iTime)];
                XMatrix = cat(3, XMatrix, SMelTemp(:, iTime:iTime + lag * fsIeeg - 1));
            else
                continue;
            end
        end
    end
end
