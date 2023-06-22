function [phonMet, ieegAll, pLabel] = timitPhonemeTimeExtractor(audioPathUsed, micAudio, ieeg, etw, pSpan)
% timitPhonemeTimeExtractor - Extracts phoneme-related features and time windows from the TIMIT dataset.
%
% Syntax: [phonMet, ieegAll, pLabel] = timitPhonemeTimeExtractor(audioPathUsed, micAudio, ieeg, etw, pSpan)
%
% Inputs:
%   audioPathUsed   - Cell array of audio file paths within the TIMIT dataset
%   micAudio        - Microphone audio data (Channels x Samples)
%   ieeg            - Intracranial EEG data (Channels x Trials x Samples)
%   etw             - Time window for the EEG data (in seconds)
%   pSpan           - Time span around each phoneme for feature extraction (in seconds)
%
% Outputs:
%   phonMet         - Structure containing phoneme-related features and time windows
%   ieegAll        - Concatenated gamma band EEG data across all phonemes (Channels x Trials x Samples)
%   pLabel          - Cell array of phoneme labels
%
% Example:
%   audioPathUsed = {'DR1\FAKS0\SA1.WAV', 'DR2\FCJF0\SA2.WAV'}; % Example audio file paths
%   micAudio = randn(2, 1000); % Example microphone audio data
%   ieeg = randn(16, 10, 1000); % Example intracranial EEG data
%   etw = [0.5 1.5]; % Example time window for EEG data
%   pSpan = [0.1 0.3]; % Example time span around each phoneme for feature extraction
%   [phonMet, ieegAll, pLabel] = timitPhonemeTimeExtractor(audioPathUsed, micAudio, ieeg, etw, pSpan); % Extract phoneme-related features
%
% Author: OpenAI
% Date: 2023-06-21

timitPath = 'E:\timit\TIMIT\'; % Path to the TIMIT dataset

timeGamma = linspace(etw(1), etw(2), size(ieeg, 3)); % Time vector for the EEG data
phonMet = []; % Initialize the structure to store phoneme-related features
ieegAll = []; % Initialize the concatenated gamma band EEG data
pLabel = []; % Initialize the cell array to store phoneme labels


for iAudio = 1:length(audioPathUsed)
    phonMetTemp = []; % Temporary structure for phoneme-related features
    
    sentPath = [timitPath audioPathUsed{iAudio}]; % Full path to the audio file
    [audFile, fsAudio] = audioread([timitPath audioPathUsed{iAudio}]); % Read the audio file from the TIMIT dataset
    
    phonPath = [sentPath(1:end-3) 'PHN']; % Path to the phoneme file
    [pb, pe, pname] = textread(phonPath, '%n %n %s'); % Read the phoneme file
    
    phonMetTemp.pb = pb ./ fsAudio; % Phoneme boundary start times (normalized)
    phonMetTemp.pe = pe ./ fsAudio; % Phoneme boundary end times (normalized)
    phonMetTemp.pname = pname; % Phoneme labels
    
    if (iAudio == 1)
        timeAudio = [0:size(micAudio, 2)-1] ./ fsMic; % Time vector for the microphone audio data
        timeAudioRead = [0:length(audFile)-1] ./ fsAudio; % Time vector for the audio file read from TIMIT
        
        figure; % Plotting example
        plot(timeAudio, micAudio(iAudio, :));
        hold on;
        plot(timeAudioRead, audFile);
        scatter(pb' ./ fsAudio, 0.2 * ones(size(pb')));
    end
    
    for iPhon = 1:length(pb)
        phonMetTemp.gamma(:, :, iPhon) = ieeg(:, iAudio, timeGamma >= (round(pb(iPhon) ./ fsAudio + pSpan(1), 2)) & timeGamma <= (round(pb(iPhon) ./ fsAudio + pSpan(2), 2))); % Extract gamma band EEG data for each phoneme
    end
    
    ieegAll = cat(3, ieegAll, phonMetTemp.gamma); % Concatenate gamma band EEG data across all phonemes
    pLabel = [pLabel; phonMetTemp.pname]; % Append phoneme labels
    
end

end
