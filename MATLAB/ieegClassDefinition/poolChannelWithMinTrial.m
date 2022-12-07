function [ieegStructAll,phonemeTrialAll,channelNameAll] = poolChannelWithMinTrial(ieegHGStruct,trialInfoStruct)
%POOLCHANNELWITHMINTRIAL Summary of this function goes here
%   Detailed explanation goes here

assert(length(ieegHGStruct)==length(trialInfoStruct),'Number of Subjects mismatch');

ieegHGStructDummy = ieegHGStruct(1).ieegHGNorm;
uniqueToken = 0;

for iSubject = 1:length(trialInfoStruct)
    uniqueTokenTemp = unique(trialInfoStruct(iSubject).phonemeTrial.tokenName);
    if(length(uniqueTokenTemp) > length(uniqueToken))
        uniqueToken = uniqueTokenTemp;
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
        repeatIds{iSubject} = find(strcmpi(trialInfoStruct(iSubject).phonemeTrial.tokenName,uniqueToken(iToken)));
        numRepeats(iSubject) = length(repeatIds{iSubject});
    end
    minRepeat = min(numRepeats);
    disp(['Number of minimum repeats : ' num2str(minRepeat)])
    dataToken = [];
    syllableToken = trialInfoStruct(1).phonemeTrial.syllableUnit(repeatIds{1}(1:minRepeat),:);
    phonemeUnitToken = trialInfoStruct(1).phonemeTrial.phonemeUnit(repeatIds{1}(1:minRepeat),:);
    phonemeClassToken = trialInfoStruct(1).phonemeTrial.phonemeClass(repeatIds{1}(1:minRepeat),:);
    phonoTacticToken = trialInfoStruct(1).phonemeTrial.phonoTactic(repeatIds{1}(1:minRepeat),:);
    for iSubject = 1:length(trialInfoStruct)
        repeatIdsShuffle = shuffle(repeatIds{iSubject});
        minRepeatIdsSubject = repeatIdsShuffle(1:minRepeat);
        dataTemp = ieegHGStruct(iSubject).ieegHGNorm.data(:,minRepeatIdsSubject,:);
        dataToken = cat(1,dataToken,dataTemp);     

    end
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

