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
end

timePad = 0.5;
if(isempty(options.normFactor))
    disp('No normalization factors provided');
    ieegBaseStruct = extractRawDataWithROI(Subject,Epoch = options.baseName,...
        Time = [options.baseTimeRange(1)-timePad options.baseTimeRange(2)+timePad],...
        roi = options.roi, remFastResponseTimeTrials=options.respTimeThresh, ...
        subsetElec=options.subsetElec);
    
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
ieegFieldStruct = extractRawDataWithROI(Subject,Epoch = options.Epoch,...
    Time = [options.Time(1)-timePad options.Time(2)+timePad],...
    roi = options.roi,remFastResponseTimeTrials=options.respTimeThresh,...
    subsetElec=options.subsetElec);
ieegHGAll = [];
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

