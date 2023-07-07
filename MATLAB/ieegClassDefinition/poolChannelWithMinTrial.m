function [ieegStructAll, phonemeTrialAll, channelNameAll] = poolChannelWithMinTrial(ieegHGStruct, trialInfoStruct)
    % Combines channels with minimum trials for a specific token across subjects.
    %
    % Arguments:
    % - ieegHGStruct: Array of structures containing ephys data for each subject.
    % - trialInfoStruct: Array of structures containing trial information for each subject.
    %
    % Returns:
    % - ieegStructAll: Combined ephys data structure with pooled channels.
    % - phonemeTrialAll: Combined trial information for all tokens.
    % - channelNameAll: Cell array of channel names from all subjects.

    % Check if the number of subjects in both input structures match
    assert(length(ieegHGStruct) == length(trialInfoStruct), 'Number of Subjects mismatch');

    % Get a dummy structure to initialize ieegStructAll
    ieegHGStructDummy = ieegHGStruct(1).ieegHGNorm;
    uniqueToken = 0;

    % Get unique tokens
    for iSubject = 1:length(trialInfoStruct)
        uniqueTokenTemp = unique(trialInfoStruct(iSubject).phonemeTrial.tokenName);
        if length(uniqueTokenTemp) > length(uniqueToken)
            uniqueToken = uniqueTokenTemp;
        end
    end

    % Display the number of tokens
    disp("Number of tokens: " + num2str(length(uniqueToken)));

    % Initialize the phonemeTrialAll structure
    phonemeTrialAll.syllableUnit = [];
    phonemeTrialAll.phonemeUnit = [];
    phonemeTrialAll.phonemeClass = [];
    phonemeTrialAll.phonotactic = [];

    % Initialize the ieegStructAll structure
    ieegStructAll = ieegStructClass([], ieegHGStructDummy.fs, ieegHGStructDummy.tw, ieegHGStructDummy.fBand, ieegHGStructDummy.name);

    % Loop over each token
    for iToken = 1:length(uniqueToken)
        disp(['Combining channels for token: ' uniqueToken{iToken}])

        % Get repeatIds and numRepeats for each subject for the current token
        repeatIds = [];
        numRepeats = [];
        for iSubject = 1:length(trialInfoStruct)
            repeatIds{iSubject} = find(strcmpi(trialInfoStruct(iSubject).phonemeTrial.tokenName, uniqueToken(iToken)));
            numRepeats(iSubject) = length(repeatIds{iSubject});
        end

        % Get the minimum number of repeats across all subjects for the current token
        minRepeat = min(numRepeats);
        disp(['Number of minimum repeats: ' num2str(minRepeat)])

        % Initialize dataToken
        dataToken = [];

        % Get the syllables, phonemes, phoneme classes, and phono tactics for the minimum number of repeats for the current token
        syllableToken = trialInfoStruct(1).phonemeTrial.syllableUnit(repeatIds{1}(1:minRepeat), :);
        phonemeUnitToken = trialInfoStruct(1).phonemeTrial.phonemeUnit(repeatIds{1}(1:minRepeat), :);
        phonemeClassToken = trialInfoStruct(1).phonemeTrial.phonemeClass(repeatIds{1}(1:minRepeat), :);
        phonoTacticToken = trialInfoStruct(1).phonemeTrial.phonoTactic(repeatIds{1}(1:minRepeat), :);

        for iSubject = 1:length(trialInfoStruct)
            repeatIdsShuffle = shuffle(repeatIds{iSubject});
            minRepeatIdsSubject = repeatIdsShuffle(1:minRepeat);
            dataTemp = ieegHGStruct(iSubject).ieegHGNorm.data(:, minRepeatIdsSubject, :);
            dataToken = cat(1, dataToken, dataTemp);
        end

        phonemeTrialAll.syllableUnit = cat(1, phonemeTrialAll.syllableUnit, syllableToken);
        phonemeTrialAll.phonemeUnit = cat(1, phonemeTrialAll.phonemeUnit, phonemeUnitToken);
        phonemeTrialAll.phonemeClass = cat(1, phonemeTrialAll.phonemeClass, phonemeClassToken);
        phonemeTrialAll.phonotactic = cat(1, phonemeTrialAll.phonotactic, phonoTacticToken);
        ieegStructAll.data = cat(2, ieegStructAll.data, dataToken);
    end

    assert(size(ieegStructAll.data, 2) == size(phonemeTrialAll.phonemeUnit, 1), 'Trial dimension mismatch');

    channelNameAll = [];

    shuffleId = randperm(size(ieegStructAll.data, 2));
    ieegStructAll.data = ieegStructAll.data(:, shuffleId, :);
    phonemeTrialAll.syllableUnit = phonemeTrialAll.syllableUnit(shuffleId, :);
    phonemeTrialAll.phonemeUnit = phonemeTrialAll.phonemeUnit(shuffleId, :);
    phonemeTrialAll.phonemeClass = phonemeTrialAll.phonemeClass(shuffleId, :);
    phonemeTrialAll.phonotactic = phonemeTrialAll.phonotactic(shuffleId, :);

    for iSubject = 1:length(ieegHGStruct)
        channelNameAll = [channelNameAll ieegHGStruct(iSubject).channelName];
    end

    assert(size(ieegStructAll.data, 1) == length(channelNameAll), 'Channel dimension mismatch');
end
