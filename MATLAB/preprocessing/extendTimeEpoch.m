function ieegStructNew = extendTimeEpoch(ieegStruct,newTimeEpoch)
%EXTENDTIMEEPOCH 
warning off;
padTimeArray = [0.1 0.2 0.3 0.4 0.5];
ieegData = ieegStruct.data;
timeEpoch = ieegStruct.tw;
fs = ieegStruct.fs;
padNumArray = padTimeArray*fs;
%time2pad = (newTimeEpoch(2)-newTimeEpoch(1))/(timeEpoch(2)-timeEpoch(1));
%time2pad = (newTimeEpoch(2)-newTimeEpoch(1))/padTime;
sigLen = (newTimeEpoch(2)-newTimeEpoch(1))*fs;
for iChan = 1:size(ieegData,1)
    ieegChan = squeeze(ieegData(iChan, :, :));
    ieegChanPad = [];
    for iTrial = 1:size(ieegChan, 1)
        padChoose = randsample(padTimeArray,1);
        time2pad = ((newTimeEpoch(2)-newTimeEpoch(1))/padChoose);
        selectTrials = setdiff(1:size(ieegChan, 1),iTrial);
        randTrials = datasample(selectTrials, ceil(time2pad) - 1, 'Replace', false);

        trials2join = ieegChan(randTrials, 1:padChoose*fs)';
        %sigGenLen = length(trials2join(:)) +padChoose;
        if(ceil(time2pad)==time2pad)
            ieegChanPad(iTrial, :) = [ieegChan(iTrial, 1:padChoose*fs) trials2join(:)'];
        else
            time2remove = (ceil(time2pad)-(time2pad))*fs;
            joinTrials = trials2join(:)';
            %randTrial = ieegChan(randsample(selectTrials,1),1:extraTimeNeed*fs-1);
            ieegChanPad(iTrial, :) = [ieegChan(iTrial, 1:padChoose*fs) joinTrials(1:sigLen-padChoose*fs)];
        end
    end
%     for iTrial = 1:size(ieegChan, 1)
%         %ieegChanPad(iTrial, :) = ieegChan(iTrial, :);
%         randStart = randi(20);
%         ieegflip = ieegChan(iTrial, randStart:randStart+padNum-1);
%         ieegPadTrial= ieegChan(iTrial, randStart:randStart+padNum-1);
%         for iPad = 1:time2pad-1 
%             ieegflip = fliplr(ieegflip);
%             ieegPadTrial = [ieegPadTrial; ieegflip];
%         end
%         ieegChanPad(iTrial,:) = ieegPadTrial;
%     end
    ieegPad(iChan,:,:) = ieegChanPad;
end

ieegStructNew = ieegStruct;
ieegStructNew.data = ieegPad;
ieegStructNew.tw = newTimeEpoch;


