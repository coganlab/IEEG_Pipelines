function [micSplitNew, micSplitCell] = timitAudioExtract(timitPath, audioPathUsed, tw, fsMic)
% timitAudioExtract - Extracts audio data from TIMIT dataset and splits it into specified time windows.
%
% Syntax: [micSplitNew, micSplitCell] = timitAudioExtract(timitPath, audioPathUsed, tw, fsMic)
%
% Inputs:
%   timitPath       - Path to the TIMIT dataset
%   audioPathUsed   - Cell array of audio file paths within the TIMIT dataset
%   tw              - Time window for each audio segment (in seconds)
%   fsMic           - Sampling frequency of the microphone (in Hz)
%
% Outputs:
%   micSplitNew     - Extracted and split audio data (Trials x Samples)
%   micSplitCell    - Cell array of the extracted audio data for each segment
%
% Example:
%   timitPath = 'C:\TIMIT\'; % Example path to the TIMIT dataset
%   audioPathUsed = {'DR1\FAKS0\SA1.WAV', 'DR2\FCJF0\SA2.WAV'}; % Example audio file paths
%   tw = [-0.5 1.5]; % Example time window of -0.5 to 1.5 seconds for each segment
%   fsMic = 20000; % Example sampling frequency of the microphone
%   [micSplitNew, micSplitCell] = timitAudioExtract(timitPath, audioPathUsed, tw, fsMic); % Extract and split the audio data
%


micSplitNew = []; % Initialize the extracted and split audio data
micSplitCell = {}; % Initialize the cell array to store the extracted audio data for each segment
timeMic = linspace(tw(1), tw(2), (tw(2) - tw(1)) * fsMic); % Generate the time vector for the microphone data
for iSound = 1:length(audioPathUsed)
    %iSound
    %[timitPath audioPathUsed{iSound}]
    [audioTimit, fsAudio] = audioread([timitPath audioPathUsed{iSound}]); % Read the audio file from the TIMIT dataset
    wav2convert = (audioTimit(:, 1)'); % Extract the audio waveform from the first channel
    
    micTemp = resample(wav2convert', fsMic, fsAudio); % Resample the audio waveform to the microphone's sampling frequency
    micTemp = micTemp'; % Transpose the resampled audio waveform
    
    
    micSplitCell{iSound} = micTemp; % Store the extracted audio data for the current segment in the cell array
    
    if (length(timeMic) > length(micTemp))
        micSplitNew(iSound, :) = [micTemp, zeros(1, length(timeMic) - length(micTemp))]; % Pad with zeros if the audio length is shorter than the time window
    else        
        micSplitNew(iSound, :) = micTemp(1:length(timeMic)); % Truncate if the audio length is longer than the time window
    end
end

end
