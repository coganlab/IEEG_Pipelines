function [micClean, fsMic] = microphoneMergeTask(micAudPath, micEphysRecord)
% microphoneMergeTask - Merge microphone audio with audio channel from ephys recording.
%
% Syntax: [micClean, fsMic] = microphoneMergeTask(micAudPath, micEphysRecord)
%
% Inputs:
%   micAudPath      - Path to the directory containing microphone audio files
%   micEphysRecord  - microphone from the ephys recording
%
% Outputs:
%   micClean        - Merged audio signal of the microphone and ephys recording
%   fsMic           - Sampling frequency of the microphone audio
%

path = fullfile(micAudPath); % Create the full path to the microphone audio directory
files = dir(path); % Get a list of files in the directory
fsMic = []; % Initialize the microphone audio sampling frequency
micClean = zeros(size(micEphysRecord)); % Initialize the merged audio signal

for fileIndex = 1:length(files) % Iterate over the files in the directory
    if (files(fileIndex).isdir == 0) % Check if the current item is a file, not a directory
        if (~isempty(strfind(files(fileIndex).name, 'wav'))) % Check if the file is a WAV file
            fullfile(path, files(fileIndex).name); % Create the full path to the WAV file
            [micTemp, fsMic] = audioread(fullfile(path, files(fileIndex).name)); % Read the microphone audio file and obtain the audio data and sampling frequency
            
            [micAudAlign, micIntanAlign] = alignsignals(micTemp, micEphysRecord); % Align the microphone audio with the ephys recording
            micAudAlign = micAudAlign'; % Transpose the aligned microphone audio
            
            micAudAlign = [micAudAlign, zeros(1, length(micEphysRecord) - length(micAudAlign))]; % Extend or truncate the aligned microphone audio to match the length of the ephys recording
            
            if (length(micAudAlign) > length(micClean)) % Check if the aligned audio is longer than the merged audio signal
                micAudAlign = micAudAlign(1:length(micClean)); % Truncate the aligned audio to match the length of the merged audio signal
            end
            
            micClean = micClean + micAudAlign; % Add the aligned microphone audio to the merged audio signal
        end
    end
end

end
