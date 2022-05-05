function [micClean,fsMic] = microphoneMergeTask(micAudPath,micEphysRecord)
path = fullfile(micAudPath);
files = dir(path);
fsMic = []
micClean = zeros(size(micEphysRecord));
for fileIndex=1:length(files)
    if (files(fileIndex).isdir == 0)
        if (~isempty(strfind(files(fileIndex).name,'wav')))
            fullfile(path,files(fileIndex).name)
            [micTemp,fsMic] = audioread(fullfile(path,files(fileIndex).name));            
            [micAudAlign,micIntanAlign] = alignsignals(micTemp,micEphysRecord);
            micAudAlign = micAudAlign';            
            micAudAlign = [micAudAlign, zeros(1, length(micEphysRecord) - length(micAudAlign))];
            if(length(micAudAlign)>length(micClean))
                micAudAlign = micAudAlign(1:length(micClean));
            end
            micClean = micClean+micAudAlign;
        end
    end
end
    
end