function [ieegSplit,trig_labels,chanMapNew] =  loadRatDataTDMS(dataPath)
h5List = dir(fullfile(dataPath,'*.h5'));
fs = 2000;
trig_labels = []; ieegSplit = [];
 for i = 1:length(h5List)
     ieegRaw = (double(h5read(strcat(dataPath,h5List(i).name),'/data')))';
     trigger = double(h5read(strcat(dataPath,h5List(i).name),'/trig')');
     dataTrig = double(h5read(strcat(dataPath,h5List(i).name),'/tones'));
     fsRecord = double(h5read(strcat(dataPath,h5List(i).name),'/Fs'));
     ieegGround = double(h5read(strcat(dataPath,h5List(i).name),'/ground_chans')');
     chanMapNew = double(h5read(strcat(dataPath,h5List(i).name),'/array_map')');
     [trigOns] = extractTriggerOnsetTone(trigger,fsRecord);
     tRec = (size(ieegRaw,2)-1)./fsRecord;
     timeRecord = linspace(0,tRec,size(ieegRaw,2));
     timeReq = linspace(0,tRec,tRec*fs);
     ieegRawNew = [];
     parfor t = 1:60
         ieegRawNew(t,:) = interp1(timeRecord,ieegRaw(t,:),timeReq,'spline');
         %ieegRawNew(t,:) = decimate(ieegRaw(t,:),fs,fsRecord);
     end
     trigger = round(trigger*fs/fsRecord);
     ieegSplit = [ieegSplit splitIeeg(ieegRawNew,round(trigOns.*fs),[-0.5 1],fs)];
     trig_labels = [trig_labels; dataTrig]; 
 end
 parfor t = 1:size(ieegSplit,2)
     ieegSplit(:,t,:) = detrend(squeeze(ieegSplit(:,t,:))')';
 end

end