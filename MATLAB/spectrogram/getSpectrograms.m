function [spec, pPerc] = getSpectrograms(ieeg, goodtrials, tw, etw, efw, prtw, pertw, intF, fs, ispermTest)
% getSpectrograms - Extracts spectrograms and performs statistical tests on ECoG data.
%
% Inputs:
%    ieeg - ECoG data (channels x trials x time)
%    goodtrials - Good trials for each channel (cell array of size [1 x channels])
%    tw - Time window of interest [start_time, end_time] in seconds
%    etw - Spectrogram time window [start_time, end_time] in seconds
%    efw - Spectrogram frequency window [start_frequency, stop_frequency] in Hz
%    prtw - Pre-onset time window [start_time, end_time] to get significant channels
%    pertw - Post-onset time window [start_time, end_time] to get significant channels
%    intF - Frequency range of interest for statistical tests [start_frequency, stop_frequency] in Hz
%    fs - Sampling frequency in Hz
%    ispermTest - Flag (0/1) indicating whether to perform a permutation test to determine channel significance
%
% Outputs:
%    spec - Spectrograms of each trial for each channel (cell array of size [1 x channels])
%    pPerc - P-values from the permutation test to check channel significance (1 x channels)
%
% Example:
%    ieeg = rand(10, 100, 1000); % Example ECoG data
%    goodtrials = cell(1, 10); % Example good trials (cell array)
%    tw = [0, 10]; % Time window of interest
%    etw = [2, 8]; % Spectrogram time window
%    efw = [30, 80]; % Spectrogram frequency window
%    prtw = [0, 2]; % Pre-onset time window
%    pertw = [2, 4]; % Post-onset time window
%    intF = [30, 50]; % Frequency range of interest for statistical tests
%    fs = 1000; % Sampling frequency
%    ispermTest = 1; % Perform permutation test
%    [spec, pPerc] = getSpectrograms(ieeg, goodtrials, tw, etw, efw, prtw, pertw, intF, fs, ispermTest); % Extract spectrograms and perform statistical tests
%

AnaParams.dn = 0.05;
AnaParams.Tapers = [0.5, 10];
AnaParams.fk = [efw(1), efw(2)];
AnaParams.Fs = fs;

channelOfInterest = 1:size(ieeg, 1);
numPerm = 10000;

for iChan = 1:length(channelOfInterest)
    iChan
    
    if isempty(goodtrials)
        trials_g = 1:size(ieeg, 2);
    elseif iscell(goodtrials)
        trials_g = goodtrials{iChan};
    else
        trials_g = goodtrials;
    end
    
    [spec{iChan}, F] = extract_spectrograms_channel(squeeze(ieeg(iChan, trials_g, :)), AnaParams);
    gammaFreq = F >= intF(1) & F <= intF(2);
    tspec = linspace(tw(1), tw(2), size(spec{iChan}, 2));
    prtspec = tspec >= prtw(1) & tspec <= prtw(2);
    perctspec = tspec >= pertw(1) & tspec <= pertw(2);
    
    if ispermTest == 1
        meanBase = [];
        meanOnsetPercept = [];
        
        for t = 1:length(trials_g)
            meanBase(t) = mean2(squeeze(spec{iChan}(t, prtspec, gammaFreq)));
            meanOnsetPercept(t) = mean2(squeeze(spec{iChan}(t, perctspec, gammaFreq)));
        end
        
        pPerc(iChan) = permtest(meanOnsetPercept, meanBase, numPerm);
    else
        pPerc(iChan) = 0;
    end
    
    etspec = tspec >= etw(1) & tspec <= etw(2);
    spec{iChan} = spec{iChan}(:, etspec, :);
end

end
