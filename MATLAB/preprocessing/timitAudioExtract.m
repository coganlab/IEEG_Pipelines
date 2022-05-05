function [micSplitNew, micSplitCell] = timitAudioExtract(timitPath,audioPathUsed,tw,fsMic)
micSplitNew = []; micSplitCell = [];
for iSound = 1:length(audioPathUsed)
    %iSound
    %[timitPath audioPathUsed{iSound}]
    [audioTimit,fsAudio] = audioread([timitPath audioPathUsed{iSound}]);
    wav2convert = (audioTimit(:,1)');
    micTemp = resample(wav2convert',fsMic,fsAudio);
    micTemp = micTemp';
    timeMic = linspace(tw(1),tw(2),(tw(2)-tw(1)).*fsMic);
    micSplitCell{iSound} = micTemp;
    
    if(length(timeMic)> length(micTemp))
        micSplitNew(iSound,:) = [micTemp zeros(1, length(timeMic) - length(micTemp))];
    else        
        micSplitNew(iSound,:) = micTemp(1:length(timeMic));
    end
end
end