function trialInfos = extractTrialInfo(Subject,options)
arguments    
    Subject struct % subject output of populated task    
    options.remNoiseTrials logical = true; % true to remove all noisy trials
    options.remNoResponseTrials logical = true; % true to remove all no-response trials 
    options.remFastResponseTimeTrials double = -1; % threshold to remove all no-response trials 
end
%EXTRACTTRIALINFO Summary of this function goes here
%   Detailed explanation goes here
trialInfos = {};
for iSubject=1:length(Subject)
    Subject(iSubject).Name
    Trials = Subject(iSubject).Trials;
    numTrials = length(Trials);
    counterN=0;
    counterNR=0;
    negResponseIdx = 0;
    noiseIdx=0;
    noResponseIdx=0;
    if(options.remNoiseTrials)
        disp('Removing Noisy trials')
        for iTrials=1:numTrials
            if Trials(iTrials).Noisy==1
                noiseIdx(counterN+1)=iTrials;
                counterN=counterN+1;
            end        
        end
    end
    if(options.remNoResponseTrials)
        disp('Removing Trials with no-response')
        for iTrials = 1:length(Trials)
            if Trials(iTrials).NoResponse==1
                noResponseIdx(counterNR+1)=iTrials;
                counterNR=counterNR+1;
            end
        end
    end
   
        respTime=[];
       for iTrials=1:length(Trials)
           if ~isempty(Trials(iTrials).ResponseStart)
               respTime(iTrials)=(Trials(iTrials).ResponseStart-Trials(iTrials).Go)./30000;
           else
               respTime(iTrials)=0;
           end
       end
    if(options.remFastResponseTimeTrials>=0)
       disp('Removing Trials with negative response time')
       negResponseIdx=find(respTime<options.remFastResponseTimeTrials);
    end
    trials2select=setdiff(1:numTrials,cat(2,noiseIdx,noResponseIdx,negResponseIdx));
    trialInfos{iSubject} = [Subject(iSubject).trialInfo(trials2select)];
    % Phoneme Sequence trial parsing
%     phonemeTrial = phonemeSequenceTrialParser(trialInfo);   
%     trialInfoStruct(iSubject).subjectId =  Subject(iSubject).Name;
%     trialInfoStruct(iSubject).phonemeTrial = phonemeTrial;
%     trialInfoStruct(iSubject).responseTime = respTime(trials2select);
end
end

