function micCleanAll = audioWrap(audioFilePath,microphoneOld)

files = dir(fullfile(audioFilePath,'*.wav'));
micCleanAll = zeros(1,length(microphoneOld));
for i = 1:length(files)
    i
    [micClean,fsAudio] = audioread(strcat(files(i).folder,'/',files(i).name));
    [micOldNew,micCleanNew] = alignsignals(microphoneOld,micClean);
    micCleanNew = padarray(micCleanNew,[length(microphoneOld)-length(micCleanNew) 0],0,'post');
    micCleanAll = micCleanAll+micCleanNew';
end
end