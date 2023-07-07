function ieegBandPass = extractBandPassDataWithROI(Subject,options)
% Extracts band-pass filtered data with specified region of interest (ROI)
% and other optional parameters.
%
% Arguments:
%   Subject: struct
%       Subject output of populated task
%   options: struct (optional)
%       Optional arguments for extraction
%       - Epoch: string (default: 'ResponseStart')
%           Epoch information; e.g., 'Auditory', 'Go', 'ResponseStart'
%       - Time: (1x2) double array (default: [-1 1])
%           Epoch time window
%       - roi: string (default: '')
%           Anatomical extraction; e.g., 'precentral', 'superiortemporal'
%       - fBand: double array (default: [15 30])
%           Frequency band for band-pass filtering
%       - fDown: double (default: 200)
%           Downsampling frequency
%       - respTimeThresh: double (default: -1)
%           Threshold for removing trials with fast response times
%       - subsetElec: cell (default: '')
%           Subset of electrodes to select from stats
%       - remNoiseTrials: logical (default: true)
%           True to remove all noisy trials
%       - remNoResponseTrials: logical (default: true)
%           True to remove all no-response trials
%       - remWMchannels: logical (default: true)
%           True to remove working memory channels
%
% Returns:
%   ieegBandPass: struct
%       Band-pass filtered data with extracted features

arguments
    Subject struct % subject output of populated task
    options.Epoch string = 'ResponseStart' % Epoch information; e.g., 'Auditory', 'Go', 'ResponseStart'
    options.Time (1,2) double = [-1 1]; % Epoch time window
    options.roi = '' % anatomical extraction; e.g., 'precentral', 'superiortemporal'
    options.fBand double = [15 30];
    options.fDown double = 200;   
    options.respTimeThresh = -1;
    options.subsetElec cell = '' % subset of electrodes to select from stats 
    options.remNoiseTrials logical = true; % true to remove all noisy trials
    options.remNoResponseTrials logical = true; % true to remove all no-response trials 
    options.remWMchannels logical = true;
end
timePad = 0.5;

% Extract raw data with ROI and other specified options
ieegFieldStruct = extractRawDataWithROI(Subject, Epoch = options.Epoch, ...
    Time = [options.Time(1)-timePad options.Time(2)+timePad], ...
    roi = options.roi, remFastResponseTimeTrials = options.respTimeThresh, ...
    remNoiseTrials = options.remNoiseTrials, remNoResponseTrials = options.remNoResponseTrials, ...
    subsetElec = options.subsetElec, remWMchannels = options.remWMchannels);

ieegBandPass = [];

% Filtering signal in the high-gamma band for each subject
for iSubject = 1:length(Subject)
    Subject(iSubject).Name
    
    % Check if ieegStruct is empty for the current subject
    if (isempty(ieegFieldStruct(iSubject).ieegStruct))
        ieegBandPass(iSubject).ieegFilter = [];
        ieegBandPass(iSubject).channelName = [];
        ieegBandPass(iSubject).trialInfo = [];
        continue;
    end
    
    % Extract band-pass filter and power for the current subject
    [ieegFilter, ieegPower] = extractBandPassFilter(ieegFieldStruct(iSubject).ieegStruct, ...
        options.fBand, options.fDown, options.Time);
    
    % Store the filtered data, channel names, and trial information
    ieegBandPass(iSubject).ieegFilter = ieegFilter;
    ieegBandPass(iSubject).ieegPower = ieegPower;
    ieegBandPass(iSubject).channelName = ieegFieldStruct(iSubject).channelName;
    ieegBandPass(iSubject).trialInfo = ieegFieldStruct(iSubject).trialInfo;
end
end



