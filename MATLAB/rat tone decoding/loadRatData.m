function [ieegSplit,trig_labels] =  loadRatData(dataPath,chanMap,ground)
    h5List = dir(fullfile(dataPath,'*.h5'));
    textList = dir(fullfile(dataPath,'*.txt'));
    trig_labels = []; ieegSplit = [];
     fs = 2000;
    for i = 1:length(h5List)
        dataRaw = h5read(strcat(dataPath,h5List(i).name),'/chdata')';
        selectedChannels = setdiff(unique(chanMap),ground);
        adc = h5read(strcat(dataPath,h5List(i).name),'/adc');
        dataTrig = load(strcat(dataPath,textList(i).name));              
       
        trigger = adc(:,1)';
        mic = adc(:,2)';
        [trigOns,timeAll] = extractTriggerOnsetTone(trigger,fs);
        hold on;
        plot(timeAll,mic);
        dataTrig = dataTrig(1:length(trigOns));
        ieegRaw = dataRaw(selectedChannels,:);
        ieegSplit = [ieegSplit splitIeeg(ieegRaw,round(trigOns.*fs),[-0.5 1],fs)];
        trig_labels = [trig_labels; dataTrig]; 
    end
end