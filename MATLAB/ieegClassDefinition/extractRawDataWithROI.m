function ieegStructAll = extractRawDataWithROI(Subject,options)
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
    options.remFastResponseTimeTrials double = -1; % response time threshold to remove all faster response trials    
    options.remWMchannels logical = true; % remove white matter channels
end

% Iterate over each subject
for iSubject = 1:length(Subject)
    Subject(iSubject).Name
    
    % Extract relevant information for the current subject
    Trials = Subject(iSubject).Trials;
    numTrials = length(Trials);
    chanIdx = Subject(iSubject).goodChannels;
    
    % Initialize counters and index arrays
    counterN = 0;
    counterNR = 0;
    noiseIdx = zeros(1, numTrials);
    noResponseIdx = zeros(1, numTrials);
    negResponseIdx = zeros(1, numTrials);
    
    % Extract anatomical and channel names from the subject information
    anatName = {Subject(iSubject).ChannelInfo.Location};
    anatName(cellfun(@isempty, anatName)) = {'dummy'};
    
    channelName = {Subject(iSubject).ChannelInfo.Name};
    channelName(cellfun(@isempty, channelName)) = {'dummy'};
    assert(length(channelName) == length(anatName), 'Channel Dimension mismatch')
    
    % Identify white matter channels
    whiteMatterIds = false(size(channelName));
    whiteMatterIds(Subject(iSubject).WM) = true;
    
    % Filter anatomical and channel names based on the selected channels
    anatName = anatName(chanIdx);
    channelName = channelName(chanIdx);
    whiteMatterIds = whiteMatterIds(chanIdx);
    
    % Select channels based on the specified region of interest (ROI)
    if (~isempty(options.roi))
        anatChanId = false(size(chanIdx));
        roirequested = options.roi;
        for iRoi = 1:length(roirequested)
            anatChanId = anatChanId | contains(anatName, roirequested{iRoi});
        end
        disp(['Selecting desired anatomy: ' num2str(sum(anatChanId))])
    else
        disp('No specified anatomy; Extracting all channels')
        anatChanId = true(size(chanIdx));
    end
    
    % Select channels based on the specified subset of electrodes
    if (~isempty(options.subsetElec))
        selectChanId = ismember(channelName, options.subsetElec);
        disp(['Selecting desired input channel: ' num2str(sum(selectChanId))])
    else
        disp('No specified input channels; Extracting all channels')
        selectChanId = true(size(chanIdx));
    end
    
    % Remove white matter channels if specified
    if (options.remWMchannels)
        disp(['Removing white matter channels: ' num2str(sum(whiteMatterIds))])
        nonwhiteMatterId = ~whiteMatterIds;
    else
        nonwhiteMatterId = true(size(chanIdx));
    end
    
    % Combine channel selection criteria
    chan2select = selectChanId & anatChanId & nonwhiteMatterId;
    
    % Check if any channels are selected for extraction
    if (isempty(find(chan2select, 1)))
        disp('No requested channels found; Iterating to the next subject');
        ieegStructAll(iSubject).ieegStruct = [];
        ieegStructAll(iSubject).channelName = [];
        continue;  % Forces the iteration for the next subject
    else
        disp(['Total number of selected channels: ' num2str(sum(chan2select))]);
    end
    
    % Extract relevant information for selected channels
    channelNameAnat = channelName(chan2select);
    
    % Remove noisy, no-response, and fast response time trials if specified
    if (options.remNoiseTrials || options.remNoResponseTrials || options.remFastResponseTimeTrials >= 0)
        % Iterate over trials and identify noisy, no-response, and negative response time trials
        for iTrials = 1:length(Trials)
            if (options.remNoiseTrials && Trials(iTrials).Noisy == 1)
                counterN = counterN + 1;
                noiseIdx(counterN) = iTrials;
            end
            
            if (options.remNoResponseTrials && Trials(iTrials).NoResponse == 1)
                counterNR = counterNR + 1;
                noResponseIdx(counterNR) = iTrials;
            end
            
            if (options.remFastResponseTimeTrials >= 0)
                if (~isempty(Trials(iTrials).ResponseStart))
                    RespTime = (Trials(iTrials).ResponseStart - Trials(iTrials).Go) ./ 30000;
                else
                    RespTime = 0;
                end
                if (RespTime < options.remFastResponseTimeTrials)
                    negResponseIdx(iTrials) = 1;
                end
            end
        end
        
        disp(['Removing Noisy trials: ' num2str(counterN)])
        disp(['Removing Trials with no-response: ' num2str(counterNR)])
        disp(['Removing Trials with negative response time: ' num2str(sum(negResponseIdx))])
    end
    
    % Combine trial indices to remove into a single array
    trials2remove = unique([noiseIdx, noResponseIdx, find(negResponseIdx)]);
    
    % Use logical indexing to select desired trials
    trials2select = setdiff(1:numTrials, trials2remove);
    TrialSelect = Trials(trials2select);
    
    % Extract epoch data for selected trials and channels
    ieegEpoch = trialIEEG(TrialSelect, chanIdx, options.Epoch, options.Time .* 1000);
    ieegEpoch = permute(ieegEpoch, [2, 1, 3]);
    fs = Subject(iSubject).Experiment.processing.ieeg.sample_rate```matlab
    ieegStruct = ieegStructClass(ieegEpoch, fs, options.Time, [1, fs/2], options.Epoch);
    
    % Perform common average referencing (CAR) if specified
    if (options.isCAR)
        ieegStruct = extractCar(ieegStruct);
    end
    
    % Filter data based on selected channels
    ieegStruct.data = ieegStruct.data(chan2select, :, :);
    assert(length(channelNameAnat) == size(ieegStruct.data, 1), 'Channel mismatch');
    assert(length(TrialSelect) == size(ieegStruct.data, 2), 'Trial mismatch');
    
    % Store the extracted data in the output structure
    ieegStructAll(iSubject).ieegStruct = ieegStruct;
    ieegStructAll(iSubject).channelName = channelNameAnat;
    ieegStructAll(iSubject).trialInfo = Subject(iSubject).trialInfo(trials2select);
end
end


