function ieegHGAll = extractHGDataWithROI(Subject,options)
% extractHGDataWithROI - Extracts high-gamma (HG) band data with region of interest (ROI) selection
%
% Usage:
%   ieegHGAll = extractHGDataWithROI(Subject, options)
%
% Inputs:
%   - Subject: struct
%     subject output of populated task
%   - options: struct, optional (default values in brackets)
%     - Epoch: string ['ResponseStart']
%       Epoch information; e.g., 'Auditory', 'Go', 'ResponseStart'
%     - Time: double array (1x2) [-1 1]
%       Epoch time window
%     - roi: string ['']
%       Anatomical extraction; e.g., 'precentral', 'superiortemporal'
%     - normFactor: []
%       Normalization factors for HG power normalization
%     - normType: integer [1]
%       Normalization type: 1 - z-score, 2 - mean normalization
%     - fDown: double [200]
%       Down-sampling frequency
%     - baseTimeRange: double array (1x2) [-0.5 0]
%       Time range for baseline extraction
%     - baseName: string ['Start']
%       Epoch name for baseline extraction
%     - respTimeThresh: double [-1]
%       Response time threshold to remove trials with faster responses
%     - subsetElec: cell ['']
%       Subset of electrodes to select from statistics
%     - remNoiseTrials: logical [true]
%       Remove all noisy trials
%     - remNoResponseTrials: logical [true]
%       Remove all no-response trials
%     - remWMchannels: logical [true]
%       Remove white matter channels
%
% Outputs:
%   - ieegHGAll: struct array
%     Extracted high-gamma data for each subject
%     - ieegHGNorm: struct
%       Normalized HG band data
%     - channelName: cell
%       Channel names for selected channels
%     - normFactor: double array
%       Normalization factors for each channel
%     - trialInfo: struct
%       Information about selected trials
%
arguments
    Subject struct % subject output of populated task
    options.Epoch string = 'ResponseStart' % Epoch information; e.g., 'Auditory', 'Go', 'ResponseStart'
    options.Time (1,2) double = [-1 1]; % Epoch time window
    options.roi = '' % anatomical extraction; e.g., 'precentral', 'superiortemporal'
    options.normFactor = [];
    options.normType = 1; % 1 - z-score, 2 - mean normalization
    options.fDown double = 200;
    options.baseTimeRange = [-0.6 -0.1]; 
    options.baseName = 'Start'
    options.respTimeThresh = 0.1;
    options.respDurThresh = 2;
    options.subsetElec cell = '' % subset of electrodes to select from stats 
    options.remNoiseTrials logical = true; % true to remove all noisy trials
    options.remNoResponseTrials logical = true; % true to remove all no-response trials 
    options.remWMchannels logical = true;
    options.remNoiseThreshold double = 10;
end

% Extract normalization type and time padding
normType = options.normType;
timePad = 0.5;

% Check if normalization factors are provided
if(isempty(options.normFactor))
    disp('No normalization factors provided');
    
    % Selecting all trials with no noise for baseline
    ieegBaseStruct = extractRawDataWithROI(Subject, 'Epoch', options.baseName, ...
        'Time', [options.baseTimeRange(1)-timePad options.baseTimeRange(2)+timePad], ...
        'roi', options.roi, 'remFastResponseTimeTrials', -1, ...
        'remNoiseTrials', false, 'remNoResponseTrials', false,  ...
        'subsetElec', options.subsetElec, 'remWMchannels', options.remWMchannels);
    
    % Extracting normalization parameters for each subject
    normFactorSubject = cell(length(Subject), 1);
    parfor iSubject = 1:length(Subject)
        if(~isempty(ieegBaseStruct(iSubject).ieegStruct))
            ieegBaseHG = extractHiGamma(ieegBaseStruct(iSubject).ieegStruct, ...
                options.fDown, options.baseTimeRange);
            normFactorBase = extractHGnormFactor(ieegBaseHG);
            normFactorSubject{iSubject} = normFactorBase;
        end
    end
else
    normFactorSubject = options.normFactor;
end

clear ieegBaseStruct;

% Extracting field epochs for the fixed parameters
ieegFieldStruct = extractRawDataWithROI(Subject, 'Epoch', options.Epoch, ...
    'Time', [options.Time(1)-timePad options.Time(2)+timePad], ...
    'roi', options.roi, 'remFastResponseTimeTrials', options.respTimeThresh, ...
    'remNoiseTrials', options.remNoiseTrials, 'remNoResponseTrials', options.remNoResponseTrials, ...
    'subsetElec', options.subsetElec, 'remWMchannels', options.remWMchannels,'remlongDurationTrials',options.respDurThresh);

ieegHGAll = repmat(struct('ieegHGNorm', [], 'channelName', [], 'normFactor', [], 'trialInfo', []), length(Subject), 1);

% Filtering signal in the high-gamma band for each subject
parfor iSubject = 1:length(Subject)
    if(~isempty(ieegFieldStruct(iSubject).ieegStruct))
        ieegFieldHG = extractHiGamma(ieegFieldStruct(iSubject).ieegStruct, ...
            options.fDown, options.Time, normFactorSubject{iSubject}, normType);
         % Removing noisy trials
%         if options.remNoiseThreshold > 0
%             [~, goodtrialIds] = remove_bad_trials(ieegFieldHG.data, threshold = options.remNoiseThreshold, method=2);
%             %goodTrialsCommon = extractCommonTrials(goodtrials);
%         else
%             goodtrialIds = ones(size(ieegEpoch,1),size(ieegEpoch,2));
%         end
% 
%         for iChan = 1:size(ieegFieldHG.data,1)
%             % Assigning noisy trials to values of 0
%             ieegFieldHG.data(iChan,~goodtrialIds(iChan,:),:) = nan;
%         end
        ieegHGAll(iSubject).ieegHGNorm = ieegFieldHG;
        ieegHGAll(iSubject).channelName = ieegFieldStruct(iSubject).channelName;
        ieegHGAll(iSubject).trialInfo = ieegFieldStruct(iSubject).trialInfo;
        ieegHGAll(iSubject).normFactor = normFactorSubject{iSubject};
        ieegHGAll(iSubject).responseTime = ieegFieldStruct(iSubject).responseTime;
        ieegHGAll(iSubject).responseDuration = ieegFieldStruct(iSubject).responseDuration;
    end
end

end
