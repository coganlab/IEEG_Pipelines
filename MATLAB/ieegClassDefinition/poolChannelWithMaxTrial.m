function [ieegStructAll,phonemeTrialAll,channelNameAll] = poolChannelWithMaxTrial(ieegHGStruct,trialInfoStruct)
%POOLCHANNELWITHMINTRIAL Summary of this function goes here
%   Detailed explanation goes here

assert(length(ieegHGStruct)==length(trialInfoStruct),'Number of Subjects mismatch');

ieegHGStructDummy = ieegHGStruct(1).ieegHGNorm;
uniqueToken = 0;

for iSubject = 1:length(trialInfoStruct)
    uniqueTokenTemp = unique(trialInfoStruct(iSubject).phonemeTrial.tokenName);
    if(length(uniqueTokenTemp) > length(uniqueToken))
        uniqueToken = uniqueTokenTemp;
        subject2choose = iSubject;
    end
end
disp("Number of tokens: " + num2str(length(uniqueToken)));

% append all tokens
phonemeTrialAll.syllableUnit = [];
phonemeTrialAll.phonemeUnit = [];
phonemeTrialAll.phonemeClass = [];
phonemeTrialAll.phonotactic = [];
ieegStructAll = ieegStructClass([],ieegHGStructDummy.fs,ieegHGStructDummy.tw,ieegHGStructDummy.fBand,ieegHGStructDummy.name);

for iToken = 1:length(uniqueToken)
    disp(['Combining channels for token: ' uniqueToken{iToken}])
    repeatIds = [];
    numRepeats = [];
    for iSubject = 1:length(trialInfoStruct)
        repeatIds{iSubject} = find(ismember(trialInfoStruct(iSubject).phonemeTrial.tokenName,uniqueToken(iToken)));
        numRepeats(iSubject) = length(repeatIds{iSubject});
    end
    [maxRepeat,maxRepeatId] = max(numRepeats);
    disp(['Number of maximum repeats : ' num2str(maxRepeat)])
    dataToken = [];
    syllableToken = trialInfoStruct(maxRepeatId).phonemeTrial.syllableUnit(repeatIds{maxRepeatId}(1:maxRepeat),:);
    phonemeUnitToken = trialInfoStruct(maxRepeatId).phonemeTrial.phonemeUnit(repeatIds{maxRepeatId}(1:maxRepeat),:);
    phonemeClassToken = trialInfoStruct(maxRepeatId).phonemeTrial.phonemeClass(repeatIds{maxRepeatId}(1:maxRepeat),:);
    phonoTacticToken = trialInfoStruct(maxRepeatId).phonemeTrial.phonoTactic(repeatIds{maxRepeatId}(1:maxRepeat),:);
    totalzeropad = 0;
    for iSubject = 1:length(trialInfoStruct)
        repeatIdsShuffle = shuffle(repeatIds{iSubject});
        %minRepeatIdsSubject = repeatIdsShuffle(1:maxRepeat);
        dataTemp = ieegHGStruct(iSubject).ieegHGNorm.data(:,repeatIdsShuffle,:);
        if(maxRepeat - length(repeatIdsShuffle)>0)
            numtrials2zeropad = maxRepeat - length(repeatIdsShuffle);
            dataTemp = cat(2,dataTemp,zeros(size(dataTemp,1),numtrials2zeropad,size(dataTemp,3)));
            totalzeropad = totalzeropad + numtrials2zeropad;
        end
            dataToken = cat(1,dataToken,dataTemp);
    end
     disp(['Number of zero padded trials : ' num2str(totalzeropad)])
    phonemeTrialAll.syllableUnit = cat(1,phonemeTrialAll.syllableUnit,syllableToken);
    phonemeTrialAll.phonemeUnit = cat(1,phonemeTrialAll.phonemeUnit,phonemeUnitToken);
    phonemeTrialAll.phonemeClass = cat(1,phonemeTrialAll.phonemeClass,phonemeClassToken);
    phonemeTrialAll.phonotactic = cat(1,phonemeTrialAll.phonotactic,phonoTacticToken);
    ieegStructAll.data = cat(2,ieegStructAll.data,dataToken);
end
assert(size(ieegStructAll.data,2)==size(phonemeTrialAll.phonemeUnit,1),'Trial dimension mismatch');
channelNameAll = [];
shuffleId = randperm(size(ieegStructAll.data,2));
ieegStructAll.data = ieegStructAll.data(:,shuffleId,:);
phonemeTrialAll.syllableUnit = phonemeTrialAll.syllableUnit(shuffleId,:);
phonemeTrialAll.phonemeUnit = phonemeTrialAll.phonemeUnit(shuffleId,:);
phonemeTrialAll.phonemeClass = phonemeTrialAll.phonemeClass(shuffleId,:);
phonemeTrialAll.phonotactic = phonemeTrialAll.phonotactic(shuffleId,:);

for iSubject = 1:length(ieegHGStruct)
    channelNameAll = [channelNameAll ieegHGStruct(iSubject).channelName];
end

assert(size(ieegStructAll.data,1)==length(channelNameAll),'Channel dimension mismatch');

end

