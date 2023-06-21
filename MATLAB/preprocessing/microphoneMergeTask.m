function [micOut] = microphoneMergeTask(micAudPath,micEphysRecord,fsEphys)
path = fullfile(micAudPath);
files = dir(path);
fsMic = []
micOut = zeros(size(micEphysRecord));
% Initialize microphone output
for fileIndex=1:length(files)
    % Check if the file index is not directory
    if (files(fileIndex).isdir == 0)
        % Check if it is a wav file
        if (~isempty(strfind(files(fileIndex).name,'wav')))
            % display the file name
            fullfile(path,files(fileIndex).name)
            % read the wav file
            [micTemp,fsMic] = audioread(fullfile(path,files(fileIndex).name));  
            % change the sampling frequency to ephys sampling frequency
            micTempResamp = resample(micTemp,fsEphys,fsMic);
            % aling the signals
            [micAudAlign,micEphysAlign] = alignsignals(micTempResamp,micEphysRecord);
            micAudAlign = micAudAlign';            
            micAudAlign = [micAudAlign, zeros(1, length(micEphysRecord) - length(micAudAlign))];
            length(micAudAlign)
            length(micOut)
            if(length(micAudAlign)>length(micOut))
                micAudAlign = micAudAlign(1:length(micOut));
            end
            micOut = micOut+micAudAlign;
        end
    end
end
    
end