function [ieegStructAll, phonemeTrialAll, channelNameAll] = poolChannelWithMaxTrial(ieegHGStruct, trialInfoStruct)
    % Combines channels with maximum trials for a specific token across subjects.
    %
    % Arguments:
    % - ieegHGStruct: Array of structures containing ephys data for each subject.
    % - trialInfoStruct: Array of structures containing trial information for each subject.
    %
    % Returns:
    % - ieegStructAll: Combined ephys data structure with pooled channels.
    % - phonemeTrialAll: Combined trial information for all tokens.
    % - channelNameAll: Cell array of channel names from all subjects.
    
    % Get the ephys data of the first subject
    ieegHGStructDummy = ieegHGStruct(1).ieegHGNorm;
    
    % Initialize variables
    uniqueToken = 0;
    subject2choose = 1;
    phonemeTrialAll.syllableUnit = [];
    phonemeTrialAll.phonemeUnit = [];
    phonemeTrialAll.phonemeClass = [];
    phonemeTrialAll.phonotactic = [];
    ieegStructAll = ieegStructClass([], ieegHGStructDummy.fs, ieegHGStructDummy.tw, ieegHGStructDummy.fBand, ieegHGStructDummy.name);
    
    % Find the token with the maximum number of unique occurrences across subjects
    for iSubject = 1:length(trialInfoStruct)
        uniqueTokenTemp = unique(trialInfoStruct(iSubject).phonemeTrial.tokenName);
        if length(uniqueTokenTemp) > length(uniqueToken)
            uniqueToken = uniqueTokenTemp;
            subject2choose = iSubject;
        end
    end
    
    disp("Number of tokens: " + num2str(length(uniqueToken)));
    
    for iToken = 1:length(uniqueToken)
        disp(['Combining channels for token: ' uniqueToken{iToken}])
        repeatIds = [];
        numRepeats = [];
        
        % Find repeat indices for the current token across subjects
        for iSubject = 1:length(trialInfoStruct)
            repeatIds{iSubject} = find(ismember(trialInfoStruct(iSubject).phonemeTrial.tokenName, uniqueToken(iToken)));
            numRepeats(iSubject) = length(repeatIds{iSubject});
        end
        
        % Find the subject with the maximum number of repeats for the current token
        [maxRepeat, maxRepeatId] = max(numRepeats);
        disp(['Number of maximum repeats: ' num2str(maxRepeat)])
        
        % Extract trial information for the current token
        syllableToken = trialInfoStruct(maxRepeatId).phonemeTrial.syllableUnit(repeatIds{maxRepeatId}(1:maxRepeat), :);
        phonemeUnitToken = trialInfoStruct(maxRepeatId).phonemeTrial.phonemeUnit(repeatIds{maxRepeatId}(1:maxRepeat), :);
        phonemeClassToken = trialInfoStruct(maxRepeatId).phonemeTrial.phonemeClass(repeatIds{maxRepeatId}(1:maxRepeat), :);
        phonoTacticToken = trialInfoStruct(maxRepeatId).phonemeTrial.phonoTactic(repeatIds{maxRepeatId}(1:maxRepeat), :);
        
        dataToken = [];
        totalzeropad = 0;
        
        % Combine the ephys data for the current token across subjects
        for iSubject = 1:length(trialInfoStruct)
            repeatIdsShuffle = shuffle(repeatIds{iSubject});
            dataTemp = ieegHGStruct(iSubject).ieegHGNorm.data(:, repeatIdsShuffle, :);
            
            if maxRepeat - length(repeatIdsShuffle) > 0
                numtrials2zeropad = maxRepeat - length(repeatIdsShuffle);
                dataTemp = cat(2, dataTemp, zeros(size(dataTemp, 1), numtrials2zeropad, size(dataTemp, 3)));
                totalzeropad = totalzeropad + numtrials2zeropad;
            end
            
            dataToken = cat(1, dataToken, dataTemp);
        end
        
        disp(['Number of zero padded trials: ' num2str(totalzeropad)])
        
        % Append the trial information and ephys data for the current token
        phonemeTrialAll.syllableUnit = cat(1, phonemeTrialAll.syllableUnit, syllableToken);
        phonemeTrialAll.phonemeUnit = cat(1, phonemeTrialAll.phonemeUnit, phonemeUnitToken);
        phonemeTrialAll.phonemeClass = cat(1, phonemeTrialAll.phonemeClass, phonemeClassToken);
        phonemeTrialAll.phonotactic = cat(1, phonemeTrialAll.phonotactic, phonoTacticToken);
        ieegStructAll.data = cat(2, ieegStructAll.data, dataToken);
    end
    
    % Ensure the dimensions of the combined data match the trial information
    assert(size(ieegStructAll.data, 2) == size(phonemeTrialAll.phonemeUnit, 1), 'Trial dimension mismatch');
    
    % Shuffle the data and trial information
    channelNameAll = [];
    shuffleId = randperm(size(ieegStructAll.data, 2));
    ieegStructAll.data = ieegStructAll.data(:, shuffleId, :);
    phonemeTrialAll.syllableUnit = phonemeTrialAll.syllableUnit(shuffleId, :);
    phonemeTrialAll.phonemeUnit = phonemeTrialAll.phonemeUnit(shuffleId, :);
    phonemeTrialAll.phonemeClass = phonemeTrialAll.phonemeClass(shuffleId, :);
    phonemeTrialAll.phonotactic = phonemeTrialAll.phonotactic(shuffleId, :);
    
    % Collect channel names from all subjects
    for iSubject = 1:length(ieegHGStruct)
        channelNameAll = [channelNameAll ieegHGStruct(iSubject).channelName];
    end
    
    assert(size(ieegStructAll.data, 1) == length(channelNameAll), 'Channel dimension mismatch');
end
