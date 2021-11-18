function [condIdx noiseIdx noResponseIdx]=SentenceRepConds(Trials)

% condIdx = index of conditions
% 1 = LS words
% 2 = LS sentences
% 3 = JL words
% 4 = JL sentences
% 5 = LM words
condIdx=zeros(length(Trials),1);
for iTrials=1:length(Trials);
    if Trials(iTrials).StartCode<=4
        condIdx(iTrials)=1;
    elseif Trials(iTrials).StartCode>4 && Trials(iTrials).StartCode<=7
        condIdx(iTrials)=2;
    elseif Trials(iTrials).StartCode>7 && Trials(iTrials).StartCode<=11
        condIdx(iTrials)=3;
    elseif Trials(iTrials).StartCode>11 && Trials(iTrials).StartCode<=14
        condIdx(iTrials)=4;
    elseif Trials(iTrials).StartCode>14
        condIdx(iTrials)=5;
    end
end

counterN=0;
counterNR=0;
noiseIdx=0;
noResponseIdx=0;
for iTrials=1:length(Trials)
    if Trials(iTrials).Noisy==1
        noiseIdx(counterN+1)=iTrials;
        counterN=counterN+1;
    end
    if Trials(iTrials).NoResponse==1
        noResponseIdx(counterNR+1)=iTrials;
        counterNR=counterNR+1;
    end
end