function [ieegBetaBand, ieegBetaP] = extractBeta(data, fs, fDown, tw, gtw, name, normFactor, normType)
    % Extracts beta band (10-50 Hz) of ieeg (active) with normalization factors
    % if provided
    %
    % Input:
    % data - ieeg data channels x trials x timepoints
    % fDown - Downsampled frequency (Optional; if not present use
    %   same sampling frequency)
    % tw - time window of the data
    % gtw - output time-window
    % name - label for data (auditory, response)
    % normFactor - normalization values (channels x 2; if not
    %   present, no normalization
    % normType - Normalization type (1=zscore, 2=mean subtracted)
    % 
    % Output:
    % ieegBetaBand - Extracted spike band structure;
    % ieegBetaP - Power of extracted spike band

switch nargin
    case 6
        normFactor = [];
        normType = 1;
    case 7
        normType = 1;
end

disp(['Extracting Beta ' name])
fSB = [15 25]; % Hz, prev=[10 50], [15 40]
ieegBetaTemp = [];

if size(data, 1) == 1
    [~, ieegBetaTemp(1, :, :)] = ...
        EcogExtractHighGammaTrial(double(squeeze(data)), fs, fDown, fSB, tw,...
                                  gtw, normFactor, normType);

else
    for iTrial = 1:size(data, 2)
        [~, ieegBetaTemp(:, iTrial, :)] = ...
            EcogExtractHighGammaTrial(double(squeeze(data(:, iTrial, :))),...
                                      fs, fDown, fSB, tw, gtw, normFactor,...
                                      normType);
    end
end

ieegBetaBand = ieegStructClass(ieegBetaTemp, fDown, gtw, fSB, name);
ieegBetaP = (squeeze(mean(log10(ieegBetaTemp.^2), 3)));

end