function [trigOnsUpdate,micSplitNew] = correctTriggerOnsetExtractor(trigOns,timitPath,audioPathUsed,micSplit,fsMic)
% trigOns - extracted Trigger Onsets (1 x #triggers) in seconds
% timitPath - file Path to timit database
% audioPath - cell matrix {1 x #triggers} - file path to individual trials
micDel =[]; trigOnsUpdate = zeros(1,length(trigOns));
micSplitNew = [];
for iSound = 1:length(audioPathUsed)
    %iSound
    %[timitPath audioPathUsed{iSound}]
    [audioTimit,fsAudio] = audioread([timitPath audioPathUsed{iSound}]);
    size(zscore(resample(audioTimit,fsMic,fsAudio)))
    if(size(audioTimit,2)>1)
        audioTimit = audioTimit(:,1);
    end
    
    [mcal,mdal,micDelTemp] = alignsignals(zscore(resample(audioTimit,fsMic,fsAudio)),zscore(micSplit(iSound,:)));
    micDelTemp
    micDel(iSound) = micDelTemp;
    micTemp = resample(audioTimit,fsMic,fsAudio);
    micTemp = micTemp';
    
    if(size(micSplit,2)> length(micTemp))
        micSplitNew(iSound,:) = [micTemp zeros(1, size(micSplit,2) - length(micTemp))];
    else        
        micSplitNew(iSound,:) = micTemp(1:size(micSplit,2));
    end
%     figure;
%     plot(zscore(resample(audioTimit{iSound},fsMic,fsAudio)));
%     hold on;
%     plot(zscore(micSplit(iSound,:))); 
    trigOnsUpdate(iSound) = (trigOns(iSound) + micDel(iSound)/fsMic);
    
end
% for iSound = 1:20
% figure;
%     plot(zscore(resample(audioTimit{iSound},fsMic,fsAudio)));
%     hold on;
%     plot(zscore(micSplitNew(iSound,:))); 
% end

end