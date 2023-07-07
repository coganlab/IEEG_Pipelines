function [trialInfos,trialInfoStruct] = extractTrialInfo(Subject,options)
% Extracts trial information based on specified options for each subject in the Subject structure.
% Returns the trial information in two outputs: trialInfos and trialInfoStruct.
%
% Arguments:
%   Subject: struct
%       Subject structure containing data for each subject
%   options: struct (optional)
%       Optional arguments for trial information extraction
%       - remNoiseTrials: logical (default: true)
%           True to remove all noisy trials
%       - remNoResponseTrials: logical (default: true)
%           True to remove all no-response trials
%       - remFastResponseTimeTrials: double (default: -1)
%           Threshold to remove trials with negative response time
%
% Returns:
%   trialInfos: cell array
%       Extracted trial information for each subject
%   trialInfoStruct: struct
%       Structured trial information for each subject
arguments    
    Subject struct % subject output of populated task    
    options.remNoiseTrials logical = true; % true to remove all noisy trials
    options.remNoResponseTrials logical = true; % true to remove all no-response trials 
    options.remFastResponseTimeTrials double = -1; % threshold to remove all no-response trials 
end
trialInfos = {};  % Cell array to store trial information for each subject
trialInfoStruct = [];  % Structured trial information for each subject

% Iterate over each subject
for iSubject = 1:length(Subject)
    Subject(iSubject).Name
    
    % Extract relevant information for the current subject
    Trials = Subject(iSubject).Trials;
    numTrials = length(Trials);
    counterN = 0;
    counterNR = 0;
    negResponseIdx = 0;
    noiseIdx = 0;
    noResponseIdx = 0;
    
    % Remove noisy trials if specified
    if (options.remNoiseTrials)
        disp('Removing Noisy trials')
        for iTrials = 1:numTrials
            if Trials(iTrials).Noisy == 1
                noiseIdx(counterN + 1) = iTrials;
                counterN = counterN + 1;
            end        
        end
    end
    
    % Remove no-response trials if specified
    if (options.remNoResponseTrials)
        disp('Removing Trials with no-response')
        for iTrials = 1:length(Trials)
            if Trials(iTrials).NoResponse == 1
                noResponseIdx(counterNR + 1) = iTrials;
                counterNR = counterNR + 1;
            end
        end
    end
   
    respTime = [];
    for iTrials = 1:length(Trials)
        if ~isempty(Trials(iTrials).ResponseStart)
            respTime(iTrials) = (Trials(iTrials).ResponseStart - Trials(iTrials).Go) ./ 30000;
        else
            respTime(iTrials) = 0;
        end
    end
    
    % Remove trials with negative response time if specified
    if (options.remFastResponseTimeTrials >= 0)
        disp('Removing Trials with negative response time')
        negResponseIdx = find(respTime < options.remFastResponseTimeTrials);
    end
    
    % Select desired trials based on the indices to keep
    trials2select = setdiff(1:numTrials, cat(2, noiseIdx, noResponseIdx, negResponseIdx));
    
    % Store the trial information in the cell array for the current subject
    trialInfos{iSubject} = [Subject(iSubject).trialInfo(trials2select)];
    
    % Perform Phoneme Sequence trial parsing
    phonemeTrial = phonemeSequenceTrialParser(trialInfos{iSubject});
    
    % Store the structured trial information in the output struct for the current subject
    trialInfoStruct(iSubject).subjectId = Subject(iSubject).Name;
    trialInfoStruct(iSubject).phonemeTrial = phonemeTrial;
    trialInfoStruct(iSubject).responseTime = respTime(trials2select);
end
end


