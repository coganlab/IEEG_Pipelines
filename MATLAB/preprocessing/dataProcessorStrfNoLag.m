% dataProcessorStrfNoLag - Process data for STRF analysis without time lag.
%
% Syntax: [XMatrix, YMatrix, XMatCell, YMatCell, YMatChanCell] = dataProcessorStrfNoLag(WMel, SMellog, ieeg)
%
% Inputs:
%   WMel        - Modulation spectrogram matrix (trial x frequency x time)
%   SMellog     - Log-scale spectrogram matrix (trial x frequency x time)
%   ieeg        - Intracranial EEG data (channels x trial x time)
%
% Outputs:
%   XMatrix     - Input matrix for STRF analysis (samples x frequency)
%   YMatrix     - Output matrix for STRF analysis (samples x channels)
%   XMatCell    - Cell array of input matrices for each trial (1 x #trials)
%   YMatCell    - Cell array of output matrices for each trial (1 x #trials)
%   YMatChanCell- Cell array of output matrices for each channel within each trial (#trials x #channels)
%


function [XMatrix, YMatrix, XMatCell, YMatCell, YMatChanCell] = dataProcessorStrfNoLag(WMel, SMellog, ieeg)
    % Initialize variables for storing matrices
    XMatrix = [];
    YMatrix = [];
    XMatCell = {};
    YMatCell = {};
    YMatChanCell = {};
    
    % Loop over each trial
    for iTrial = 1:size(ieeg, 2)
        WMelTemp = squeeze(WMel(iTrial, :, :));
        SMelLogTemp = squeeze(SMellog(iTrial, :, :));
        ieegTemp = squeeze(ieeg(:, iTrial, :));
        
        xmattemp = [];
        ymattemp = [];
        
        % Loop over each time point within the trial
        for iTime = 1:size(ieegTemp, 2)
            % Check if the values in SMelLogTemp are finite
            if isfinite(SMelLogTemp(:, iTime))
                YMatrix = cat(2, YMatrix, ieegTemp(:, iTime));
                XMatrix = cat(2, XMatrix, WMelTemp(:, iTime));
                ymattemp = cat(2, ymattemp, ieegTemp(:, iTime));
                xmattemp = cat(2, xmattemp, WMelTemp(:, iTime));
            else
                continue;
            end
        end
        
        % Handle infinite values in xmattemp by replacing them with the mean
        infVal = isinf(xmattemp);
        meanX = mean(xmattemp(infVal == 0));
        xmattemp(infVal) = meanX;
        
        % Store the input and output matrices for each trial
        XMatCell{iTrial} = xmattemp';
        YMatCell{iTrial} = ymattemp';
        
        % Store the output matrices for each channel within each trial
        for iChan = 1:size(ymattemp, 1)
            YMatChanCell{iTrial, iChan} = ymattemp(iChan, :)';
        end
    end
    
    % Handle infinite values in XMatrix by replacing them with the mean
    infVal = isinf(XMatrix);
    meanX = mean(XMatrix(infVal == 0));
    XMatrix(infVal) = meanX;
    
    % Transpose the matrices for proper orientation
    XMatrix = XMatrix';
    YMatrix = YMatrix';
end
