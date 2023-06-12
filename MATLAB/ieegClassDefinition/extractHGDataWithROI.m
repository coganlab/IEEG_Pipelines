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

function ieegHGAll = extractHGDataWithROI(Subject,options)
arguments
    Subject struct % subject output of populated task
    options.Epoch string = 'ResponseStart' % Epoch information; e.g., 'Auditory', 'Go', 'ResponseStart'
    options.Time (1,2) double = [-1 1]; % Epoch time window
    options.roi = '' % anatomical extraction; e.g., 'precentral', 'superiortemporal'
    options.normFactor = [];
    options.normType = 1; % 1 - z-score, 2 - mean normalization
    options.fDown double = 200;
    options.baseTimeRange = [-0.5 0]; 
    options.baseName = 'Start'
    options.respTimeThresh = -1;
    options.subsetElec cell = '' % subset of electrodes to select from stats 
    options.remNoiseTrials logical = true; % true to remove all noisy trials
    options.remNoResponseTrials logical = true; % true to remove all no-response trials 
    options.remWMchannels logical = true;
end

% Extract normalization type and time padding
normType = options.normType;
timePad = 0.5;

% Check if normalization factors are provided
if(isempty(options.normFactor))
    disp('No normalization factors provided');
    
    % Selecting all trials with no noise for baseline
    ieegBaseStruct = extractRawDataWithROI(Subject, Epoch = options.baseName, ...
        Time = [options.baseTimeRange(1)-timePad options.baseTimeRange(2)+timePad], ...
        roi = options.roi, remFastResponseTimeTrials = -1, ...
        remNoiseTrials = options.remNoiseTrials, remNoResponseTrials = false, ...
        subsetElec = options.subsetElec, remWMchannels = options.remWMchannels);
    
    % Extracting normalization parameters for each subject
    parfor iSubject = 1:length(Subject)
        if(isempty(ieegBaseStruct(iSubject).ieegStruct))
            normFactorSubject{iSubject} = [];
            continue;
        end
        ieegBaseHG = extractHiGamma(ieegBaseStruct(iSubject).ieegStruct, ...
            options.fDown, options.baseTimeRange);
        normFactorBase = extractHGnormFactor(ieegBaseHG);
        normFactorSubject{iSubject} = normFactorBase;
    end
else
    normFactorSubject = options.normFactor;
end

clear ieegBaseStruct;

% Extracting field epochs for the fixed parameters
ieegFieldStruct = extractRawDataWithROI(Subject, Epoch = options.Epoch, ...
    Time = [options.Time(1)-timePad options.Time(2)+timePad], ...
    roi = options.roi, remFastResponseTimeTrials = options.respTimeThresh, ...
    remNoiseTrials = options.remNoiseTrials, remNoResponseTrials = options.remNoResponseTrials, ...
    subsetElec = options.subsetElec, remWMchannels = options.remWMchannels);

ieegHGAll = [];

% Filtering signal in the high-gamma band for each subject
parfor iSubject = 1:length(Subject)
    if(isempty(ieegFieldStruct(iSubject).ieegStruct))
        ieegHGAll(iSubject).ieegHGNorm = [];
        ieegHGAll(iSubject).channelName = [];
        ieegHGAll(iSubject).normFactor = [];
        ieegHGAll(iSubject).trialInfo = [];
        continue;
    end
    ieegFieldHG = extractHiGamma(ieegFieldStruct(iSubject).ieegStruct, ...
        options.fDown, options.Time, normFactorSubject{iSubject}, normType);
    ieegHGAll(iSubject).ieegHGNorm = ieegFieldHG;
    ieegHGAll(iSubject).channelName = ieegFieldStruct(iSubject).channelName;
    ieegHGAll(iSubject).trialInfo = ieegFieldStruct(iSubject).trialInfo;
    ieegHGAll(iSubject).normFactor = normFactorSubject{iSubject};
end

end
