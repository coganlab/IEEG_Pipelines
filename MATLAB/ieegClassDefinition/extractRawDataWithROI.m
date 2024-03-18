function ieegStructAll = extractRawDataWithROI(Subject, options)
% Extracts raw data based on specified options for each subject in the Subject structure
% and returns the extracted data in the ieegStructAll structure.
%
% Arguments:
%   Subject: struct
%       Subject structure containing data for each subject
%   options: struct (optional)
%       Optional arguments for data extraction
%       - Epoch: string (default: 'Start')
%           Epoch information; e.g., 'Auditory', 'Go', 'ResponseStart'
%       - Time: double array (default: [-1 1])
%           Epoch time window
%       - roi: string (default: '')
%           Anatomical extraction; e.g., {'precentral', 'superiortemporal'}
%       - subsetElec: cell (default: '')
%           Subset of electrodes to select from stats
%       - isCAR: logical (default: true)
%           True to perform common average referencing (CAR)
%       - remNoiseTrials: logical (default: true)
%           True to remove all noisy trials
%       - remNoResponseTrials: logical (default: true)
%           True to remove all no-response trials
%       - remFastResponseTimeTrials: double (default: -1)
%           Response time threshold to remove faster response trials
%       - remWMchannels: logical (default: true)
%           True to remove white matter channels
%       - remNoiseThreshold: double (default: 10)
%           True to remove white matter channels
%
%
% Returns:
%   ieegStructAll: struct
%       Extracted raw data for each subject

arguments
    Subject struct % subject output of populated task
    options.Epoch string = 'Start' % Epoch information; e.g., 'Auditory', 'Go', 'ResponseStart'
    options.Time double{mustBeVector} = [-1 1]; % Epoch time window
    options.roi = '' % anatomical extraction; e.g., {'precentral', 'superiortemporal'}
    options.subsetElec cell = '' % subset of electrodes to select from stats 
    options.isCAR logical = true; % true to perform CAR subtraction
    options.remNoiseTrials logical = true; % true to remove all noisy trials
    options.remNoResponseTrials logical = true; % true to remove all no-response trials    
    options.remFastResponseTimeTrials double = 0; % response time threshold to remove all faster response trials   
    options.remlongDurationTrials double = 0 % response duration threshold to remove longer duration trials   
    options.remNoiseThreshold double = -1; % noise threshold to remove noisy trials
    options.remWMchannels logical = true; % remove white matter channels
end

% Precompute subject-specific data that does not depend on the options
numSubjects = length(Subject);
ieegStructAll = struct('ieegStruct', cell(1, numSubjects), ...
                       'channelName', cell(1, numSubjects), ...
                       'trialInfo', cell(1, numSubjects));
                   
for iSubject = 1:numSubjects
    currentSubject = Subject(iSubject);
    disp(currentSubject.Name);
    
    % Extract relevant information for the current subject
    Trials = currentSubject.Trials;
    numTrials = length(Trials);
    goodChan = currentSubject.goodChannels;
    
    % Extract anatomical and channel names from the subject information
    anatName = {currentSubject.ChannelInfo.Location};    
    anatName(cellfun(@isempty, anatName)) = {'dummy'};
    
    channelName = {currentSubject.ChannelInfo.Name};
    chanIdx = ismember(1:length(channelName),goodChan);
%     channelName(cellfun(@isempty, channelName)) = {'dummy'};
    chanIdx(cellfun(@isempty, channelName)) = 0;
    assert(length(channelName) == length(anatName), 'Channel Dimension mismatch')
    
    % Identify white matter channels
    whiteMatterIds = false(size(channelName));
    whiteMatterIds(currentSubject.WM) = true;
    
    % Filter anatomical and channel names based on the selected channels
    anatName = anatName(chanIdx);
    channelName = channelName(chanIdx);
    whiteMatterIds = whiteMatterIds(chanIdx);
    
    % Select channels based on the specified region of interest (ROI)
    if (~isempty(options.roi))
        roirequested = options.roi;
        anatChanId = contains(anatName, roirequested{1});
        for iRoi = 2:length(roirequested)
            anatChanId = anatChanId | contains(anatName, roirequested{iRoi});
        end
        disp(['Selecting desired anatomy: ' num2str(sum(anatChanId))])
    else
        disp('No specified anatomy; Extracting all channels')
        anatChanId = true(size(anatName));
    end
    
    % Select channels based on the specified subset of electrodes
    if (~isempty(options.subsetElec))
        selectChanId = ismember(channelName, options.subsetElec);
        disp(['Selecting desired input channel: ' num2str(sum(selectChanId))])
    else
        disp('No specified input channels; Extracting all channels')
        selectChanId = true(size(channelName));
    end
    
    % Remove white matter channels if specified
    if (options.remWMchannels)
        disp(['Removing white matter channels: ' num2str(sum(whiteMatterIds))])
        nonwhiteMatterId = ~whiteMatterIds;
    else
        nonwhiteMatterId = true(size(whiteMatterIds));
    end
    
    % Combine channel selection criteria
    chan2select = selectChanId & anatChanId & nonwhiteMatterId;
    
    % Check if any channels are selected for extraction
    if (isempty(find(chan2select, 1)))
        disp('No requested channels found; Iterating to the next subject');
        continue;  % Forces the iteration for the next subject
    else
        disp(['Total number of selected channels: ' num2str(sum(chan2select))]);
    end
    
    % Extract relevant information for selected channels
    channelNameAnat = channelName(chan2select);
    responseTime = nan(1,length(Trials));
    responseDuration = nan(1,length(Trials));
    % Remove noisy, no-response, and fast response time trials if specified
    if (options.remNoiseTrials || options.remNoResponseTrials || options.remFastResponseTimeTrials >= 0|| options.remlongDurationTrials >= 0)
        noisyTrials = [Trials.Noisy] == 1;
        noResponseTrials = [Trials.NoResponse] == 1;
        negResponseTrials = false(1, numTrials);
        longResponseTrials = false(1, numTrials);
        if (options.remFastResponseTimeTrials >= 0)
            for iTrials = 1:numTrials
                if (~isempty(Trials(iTrials).ResponseStart))
                    RespTime = (Trials(iTrials).ResponseStart - Trials(iTrials).Go) ./ 30000;
                    RespDuration = (Trials(iTrials).ResponseEnd - Trials(iTrials).ResponseStart) ./ 30000;
                else
                    RespTime = nan;
                    RespDuration = nan;
                end
                if (RespTime < options.remFastResponseTimeTrials)
                    negResponseTrials(iTrials) = true;
                end
                if (RespDuration >= options.remlongDurationTrials)
                    longResponseTrials(iTrials) = true;
                end
                responseTime(iTrials) = RespTime;
                responseDuration(iTrials) = RespDuration;
            end
        end
        
        disp(['Removing Noisy trials: ' num2str(sum(noisyTrials))])
        disp(['Removing Trials with no-response: ' num2str(sum(noResponseTrials))])
        disp(['Removing Trials with negative response time: ' num2str(sum(negResponseTrials))])
        disp(['Removing Trials with longer Duration: ' num2str(sum(longResponseTrials))])
        % Combine trial indices to remove into a single array
        trials2remove = unique([find(noisyTrials), find(noResponseTrials), find(negResponseTrials), find(longResponseTrials)]);
        trials2select = setdiff(1:numTrials, trials2remove);
    else
        % If no trials are to be removed, select all trials
        trials2select = 1:numTrials;
    end
    
    % Extract epoch data for selected trials and channels
    ieegEpoch = trialIEEG(Trials(trials2select), find(chanIdx), options.Epoch, options.Time .* 1000);
    ieegEpoch = permute(ieegEpoch, [2, 1, 3]);

    fs = currentSubject.Experiment.processing.ieeg.sample_rate;
    ieegStruct = ieegStructClass(ieegEpoch, fs, options.Time, [1, fs/2], options.Epoch);
    
    % Perform common average referencing (CAR) if specified
    if (options.isCAR)
        ieegStruct = extractCar(ieegStruct);
        %ieegStruct = extractCableCar(ieegStruct,channelName);
    end
    
    % Removing noisy trials
    if options.remNoiseThreshold > 0
        [~, goodtrialIds] = remove_bad_trials(ieegStruct.data, options.remNoiseThreshold);
        %goodTrialsCommon = extractCommonTrials(goodtrials);
    else
        goodtrialIds = ones(size(ieegEpoch,1),size(ieegEpoch,2));
    end

    for iChan = 1:size(ieegEpoch,1)
        % Assigning noisy trials to values of 0
        ieegStruct.data(iChan,~goodtrialIds(iChan,:),:) = nan;
    end
    
    
    % Filter data based on selected channels
    ieegStruct.data = ieegStruct.data(chan2select, :, :);
    assert(length(channelNameAnat) == size(ieegStruct.data, 1), 'Channel mismatch');
    
    % Store the extracted data in the output structure
    ieegStructAll(iSubject).ieegStruct = ieegStruct;
    ieegStructAll(iSubject).channelName = channelNameAnat;
    ieegStructAll(iSubject).trialInfo = currentSubject.trialInfo(trials2select);
    
    ieegStructAll(iSubject).responseTime = responseTime(trials2select);
    ieegStructAll(iSubject).responseDuration = responseDuration(trials2select);
end
end
