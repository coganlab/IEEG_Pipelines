function ieegStructAll = extractRawDataWithROI(Subject,options)
arguments
    Subject struct % subject output of populated task
    options.Epoch string = 'Start' % Epoch information; e.g., 'Auditory', 'Go', 'ResponseStart'
    options.Time double{mustBeVector} = [-1 1]; % Epoch time window
    options.roi = '' % anatomical extraction; e.g., 'precentral', 'superiortemporal'
    options.subsetElec cell = '' % subset of electrodes to select from stats 
    options.isCAR logical = true; % true to perform CAR subtraction
    options.remNoiseTrials logical = true; % true to remove all noisy trials
    options.remNoResponseTrials logical = true; % true to remove all no-response trials    
    options.remFastResponseTimeTrials double = -1; % response time threshold to remove all faster response trials    
    options.remWMchannels logical = true;
end

ieegStructAll = [];

for iSubject=1:length(Subject)

    Subject(iSubject).Name
    Trials=Subject(iSubject).Trials;
    numTrials = length(Trials);
    chanIdx=Subject(iSubject).goodChannels;
    counterN=0;
    counterNR=0;
    noiseIdx=0;
    noResponseIdx=0;
    negResponseIdx = 0;

    anatName = {Subject(iSubject).ChannelInfo.Location};
    anatName(cellfun(@isempty,anatName)) = {'dummy'};
    
    
    channelName = {Subject(iSubject).ChannelInfo.Name};
    channelName(cellfun(@isempty,channelName)) = {'dummy'};
    assert(length(channelName)==length(anatName),'Channel Dimension mismatch')
    
    whiteMatterIds = false(size(channelName));
    whiteMatterIds(Subject(iSubject).WM) = true;
    
    anatName = anatName(chanIdx);
    channelName = channelName(chanIdx);
    whiteMatterIds= whiteMatterIds(chanIdx);

    if(~isempty(options.roi))
        
        anatChanId = contains(anatName,options.roi);  
        disp(['Selecting desired anatomy : ' num2str(sum(anatChanId))])
    else
        disp('No specified anatomy; Extracting all channels')
        anatChanId = true(size(chanIdx));
    end

    if(~isempty(options.subsetElec))
        selectChanId = ismember(channelName,options.subsetElec);  
        disp(['Selecting desired input channel : ' num2str(sum(selectChanId))])
    else
        disp('No specified input channels; Extracting all channels')
        selectChanId = true(size(chanIdx));
    end

    if(options.remWMchannels)
        disp(['Removing white matter channels : ' num2str(sum(whiteMatterIds))])
        nonwhiteMatterId = ~whiteMatterIds;
    else
        nonwhiteMatterId = true(size(chanIdx));
    end
    
    chan2select = selectChanId & anatChanId & nonwhiteMatterId;

    if(isempty(find(chan2select)))
        disp('No requested channels found; Iterating next subject');
        ieegStructAll(iSubject).ieegStruct = [];
        ieegStructAll(iSubject).channelName = [];
        continue;
        % Forces the iteration for next subject;
    else
        disp(['Total number of selected channels : ' num2str(sum(chan2select))]);
    end

    
    channelNameAnat = channelName(chan2select);
    if(options.remNoiseTrials)
        
        for iTrials=1:length(Trials)
            if Trials(iTrials).Noisy==1
                noiseIdx(counterN+1)=iTrials;
                counterN=counterN+1;
            end        
        end
        disp(['Removing Noisy trials : ' num2str(length(noiseIdx))])
    end
    if(options.remNoResponseTrials)
        
        for iTrials = 1:length(Trials)
            if Trials(iTrials).NoResponse==1
                noResponseIdx(counterNR+1)=iTrials;
                counterNR=counterNR+1;
            end
        end
        disp(['Removing Trials with no-response : ' num2str(length(noResponseIdx))] )
    end

    if(options.remFastResponseTimeTrials>=0)
        
        RespTime=[];
       for iTrials=1:length(Trials)
           if ~isempty(Trials(iTrials).ResponseStart)
               RespTime(iTrials)=(Trials(iTrials).ResponseStart-Trials(iTrials).Go)./30000;
           else
               RespTime(iTrials)=0;
           end
       end
       negResponseIdx=find(RespTime<options.remFastResponseTimeTrials);
       disp(['Removing Trials with negative response time ' num2str(length(negResponseIdx))])
    end
    trials2remove = unique(cat(2,noiseIdx,noResponseIdx, negResponseIdx));
    
    TrialSelect=Trials(setdiff(1:numTrials,trials2remove));
    ieegEpoch=trialIEEG(TrialSelect,chanIdx,options.Epoch,options.Time.*1000);
    ieegEpoch = permute(ieegEpoch,[2,1,3]);
    fs = Subject(iSubject).Experiment.processing.ieeg.sample_rate;   
    ieegStruct = ieegStructClass(ieegEpoch, fs, options.Time, [1 fs/2], options.Epoch);   

    if(options.isCAR)
        ieegStruct = extractCar(ieegStruct);
    end
    ieegStruct.data = ieegStruct.data(chan2select,:,:);
%     length(channelNameAnat)
%     size(ieegStruct.data)
    assert(length(channelNameAnat)==size(ieegStruct.data,1),'Channel mismatch');
    assert(length(TrialSelect)==size(ieegStruct.data,2),'Trial mismatch');
    ieegStructAll(iSubject).ieegStruct = ieegStruct;
    ieegStructAll(iSubject).channelName = channelNameAnat;
end


end