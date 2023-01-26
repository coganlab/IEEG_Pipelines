function [ieegHighGamma, ieegHighGammaPower] = extractHG_chestek(data, fs, fDown, tw, gtw, name, normFactor, normType)
    % Extracts High Gamma Power (HG) of ieeg (active) with normalization factors
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
    % ieegSpikeBand - Extracted spike band structure;
    % ieegSPB - Power of extracted spike band

switch nargin
    case 6
        normFactor = [];
        normType = 1;
    case 7
        normType = 1;
end

disp(['Extracting High Gamma (Chestek) ' name])
fHG = [70 150]; % Hz
nFilterPoints = 200; % Filter order
decimateFactor = 10; % Decimate to fs/10 (2 kHz -> 200 Hz)
fDown = fs/decimateFactor;
ieegHGtemp = [];

if size(data, 1) == 1
    ieegHGtemp(1, :, :) = ...
        EcogExtractSpikeBandPowerTrial(double(squeeze(data)), fs, fHG, tw,...
                                       gtw, nFilterPoints, decimateFactor,...
                                       normFactor, normType);
        % EcogExtractHighGammaTrial(double(squeeze(data)), fs, fDown, fHG, tw,...
        %                           gtw, normFactor, normType);

else
    for iTrial = 1:size(data, 2)
        ieegHGtemp(:, iTrial, :) = ...
        EcogExtractSpikeBandPowerTrial(double(squeeze(data(:, iTrial,:))),...
                                       fs, fHG, tw, gtw, nFilterPoints,...
                                       decimateFactor, normFactor, normType);
            % EcogExtractHighGammaTrial(double(squeeze(data(:, iTrial, :))),...
            %                           fs, fDown, fHG, tw, gtw, normFactor,...
            %                           normType);
    end
end

ieegHighGamma = ieegStructClass(ieegHGtemp, fDown, gtw, fHG, name);
ieegHighGammaPower = (squeeze(mean(log10(ieegHGtemp.^2), 3)));

end