function ieegHGAll = extractHGDataWithROI(Subject,options)
arguments
    Subject struct % subject output of populated task
    options.Epoch string = 'ResponseStart' % Epoch information; e.g., 'Auditory', 'Go', 'ResponseStart'
    options.Time (1,2) double = [-1 1]; % Epoch time window
    options.roi = '' % anatomical extraction; e.g., 'precentral', 'superiortemporal'
    options.normFactor double = [];
    options.fDown double = 200;
    options.baseTimeRange = [-0.5 0]; 
    options.baseName = 'Start'
    options.respTimeThresh = -1;
    options.subsetElec cell = '' % subset of electrodes to select from stats 
    options.remNoiseTrials logical = true; % true to remove all noisy trials
    options.remNoResponseTrials logical = true; % true to remove all no-response trials 
    options.remWMchannels logical = true;
end

timePad = 0.5;
if(isempty(options.normFactor))
    disp('No normalization factors provided');
    % Selecting all trials with no noise for baseline
    ieegBaseStruct = extractRawDataWithROI(Subject,Epoch = options.baseName,...
        Time = [options.baseTimeRange(1)-timePad options.baseTimeRange(2)+timePad],...
        roi = options.roi, remFastResponseTimeTrials=-1, ...
        remNoiseTrials=options.remNoiseTrials,remNoResponseTrials=false,...
        subsetElec=options.subsetElec, remWMchannels=options.remWMchannels);
    
    % Extracting normalization parameters for each subject
    for iSubject = 1:length(Subject)
        if(isempty(ieegBaseStruct(iSubject).ieegStruct))
            continue;
        end
        ieegBaseHG = extractHiGamma(ieegBaseStruct(iSubject).ieegStruct,...
            options.fDown,options.baseTimeRange);
        normFactorBase = extractHGnormFactor(ieegBaseHG);
        normFactorSubject(iSubject).normFactor = normFactorBase;
    end
else
    normFactorSubject = options.normFactor;
end
clear ieegBaseStruct;
% Extracting field epochs for the fixed parameters
ieegFieldStruct = extractRawDataWithROI(Subject,Epoch = options.Epoch,...
    Time = [options.Time(1)-timePad options.Time(2)+timePad],...
    roi = options.roi,remFastResponseTimeTrials=options.respTimeThresh,...
    remNoiseTrials=options.remNoiseTrials,remNoResponseTrials=options.remNoResponseTrials,...
    subsetElec=options.subsetElec, remWMchannels=options.remWMchannels);
ieegHGAll = [];
% Filtering signal in the high-gamma band for each subject
for iSubject = 1:length(Subject)
    if(isempty(ieegFieldStruct(iSubject).ieegStruct))
        ieegHGAll(iSubject).ieegHGNorm = [];
        ieegHGAll(iSubject).channelName = [];
        ieegHGAll(iSubject).normFactor = [];
            continue;
    end
    ieegFieldHG = extractHiGamma(ieegFieldStruct(iSubject).ieegStruct,...
        options.fDown, options.Time,normFactorSubject(iSubject).normFactor,2);
    ieegHGAll(iSubject).ieegHGNorm = ieegFieldHG;
    ieegHGAll(iSubject).channelName = ieegFieldStruct(iSubject).channelName;
    ieegHGAll(iSubject).normFactor = normFactorSubject(iSubject).normFactor;
end


end

