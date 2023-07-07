function [trigOnsUpdate, micSplitNew] = correctTriggerOnsetExtractor(trigOns, timitPath, audioPathUsed, micSplit, fsMic)
% correctTriggerOnsetExtractor - Correct trigger onset extraction using audio alignment.
%
% Syntax: [trigOnsUpdate, micSplitNew] = correctTriggerOnsetExtractor(trigOns, timitPath, audioPathUsed, micSplit, fsMic)
%
% Inputs:
%   trigOns       - Extracted trigger onsets (1 x #triggers) in seconds
%   timitPath     - File path to TIMIT database
%   audioPathUsed - Cell matrix {1 x #triggers} - File path to individual trials
%   micSplit      - Microphone split signals (matrix with size #triggers x samples)
%   fsMic         - Sampling frequency of the microphone split signals
%
% Outputs:
%   trigOnsUpdate - Updated trigger onsets after correction
%   micSplitNew   - Corrected microphone split signals after alignment
%
% Example:
%   trigOns = [0.5, 1.2, 2.8];
%   timitPath = 'path/to/timit';
%   audioPathUsed = {'trial1.wav', 'trial2.wav', 'trial3.wav'};
%   micSplit = [mic1; mic2; mic3]; % matrix of microphone split signals
%   fsMic = 44100;
%   [trigOnsUpdate, micSplitNew] = correctTriggerOnsetExtractor(trigOns, timitPath, audioPathUsed, micSplit, fsMic);
%   

% Initialize variables for storing microphone delays and updated trigger onsets
    micDel = [];
    trigOnsUpdate = zeros(1, length(trigOns));
    micSplitNew = [];
    
    % Loop over each trigger sound
    for iSound = 1:length(audioPathUsed)
        % Read audio from TIMIT database
        [audioTimit, fsAudio] = audioread([timitPath audioPathUsed{iSound}]);
        
        % Resample and normalize the audio signal
        audioTimit = zscore(resample(audioTimit, fsMic, fsAudio));
        
        % Check if the audio signal has multiple channels and keep only the first channel
        if size(audioTimit, 2) > 1
            audioTimit = audioTimit(:, 1);
        end
        
        % Perform signal alignment between audio and microphone split signal
        [mcal, mdal, micDelTemp] = alignsignals(audioTimit, zscore(micSplit(iSound, :)));
        
        % Store the microphone delay
        micDel(iSound) = micDelTemp;
        
        % Resample the aligned audio signal
        micTemp = resample(audioTimit, fsMic, fsAudio);
        micTemp = micTemp';
        
        % Adjust the length of the microphone split signal to match the aligned audio signal
        if size(micSplit, 2) > length(micTemp)
            micSplitNew(iSound, :) = [micTemp zeros(1, size(micSplit, 2) - length(micTemp))];
        else
            micSplitNew(iSound, :) = micTemp(1:size(micSplit, 2));
        end
        
        % Update the trigger onset with the microphone delay
        trigOnsUpdate(iSound) = trigOns(iSound) + micDel(iSound) / fsMic;
    end
end
