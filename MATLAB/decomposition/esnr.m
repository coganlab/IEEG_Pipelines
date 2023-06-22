% esnr - Calculate the Evoke Signal-to-Noise Ratio (eSNR) between a channel signal and channel noise.
%
% Syntax: evokeSnr = esnr(chanSignal, chanNoise)
%
% Inputs:
%   chanSignal - Matrix of channel signal samples (rows represent observations, columns represent features)
%   chanNoise  - Matrix of channel noise samples (rows represent observations, columns represent features)
%
% Output:
%   evokeSnr   - Scalar value representing the eSNR in decibels (dB)
%
% Example:
%   chanSignal = [1 2 3; 4 5 6; 7 8 9];
%   chanNoise = [0.5 1 1.5; 2 2.5 3; 3.5 4 4.5];
%   evokeSnr = esnr(chanSignal, chanNoise);
%


function evokeSnr = esnr(chanSignal, chanNoise)
    % Calculate the covariance matrix of the noise
    sigmaNoise = cov1para(chanNoise);
    
    % Calculate the Mahalanobis distances of the noise samples
    mDistNoise = mahalUpdate(chanNoise, chanNoise, sigmaNoise);
    
    % Calculate the variance of the noise
    varNoise = exp(mean(log(mDistNoise.^2)));
    
    % Calculate the Mahalanobis distances of the signal samples
    mDistSignal = mahalUpdate(chanSignal, chanNoise, sigmaNoise);
    
    % Calculate the eSNR in decibels (dB)
    evokeSnr = 10 .* log10(mean(mDistSignal.^2 / varNoise));
end
