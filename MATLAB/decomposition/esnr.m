function evokeSnr = esnr(chanSignal, chanNoise)
% esnr - Calculate the Evoke Signal-to-Noise Ratio (eSNR) between a channel signal and channel noise.
%
% Syntax: evokeSnr = esnr(chanSignal, chanNoise)
%
% Inputs:
%   chanSignal - Vector of channel signal samples (rows represent observations, columns represent features)
%   chanNoise  - Vector of channel noise samples (rows represent observations, columns represent features)
%
% Output:
%   evokeSnr   - Scalar value representing the eSNR in decibels (dB)
%
% Example:
%   chanSignal = [1 2 3 4 5 6 7 8 9];
%   chanNoise = [0.5 1 1.5 2 2.5 3 3.5 4 4.5];
%   evokeSnr = esnr(chanSignal, chanNoise);
%

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



function dist = mahalUpdate(noise, signal, sigma)
% mahalUpdate - Calculate Mahalanobis distance between signal and noise.
%
% Syntax: dist = mahalUpdate(noise, signal, sigma)
%
% Inputs:
%   noise  - Matrix of noise samples (rows represent trials, columns represent features)
%   signal - Matrix of signal samples (rows represent trials, columns represent features)
%   sigma  - Covariance matrix (assumed to be positive definite)
%
% Output:
%   dist   - Row vector of Mahalanobis distances for each trial
%
% Example:
%   noise = [1 2; 3 4; 5 6];
%   signal = [2 3; 4 5; 6 7];
%   sigma = [1 0.5; 0.5 1];
%   dist = mahalUpdate(noise, signal, sigma);
%

    % Calculate the mean of the noise samples
    meanN = mean(noise, 1)';
    
    % Calculate the number of trials
    nTrials = size(noise, 1);
    
    % Initialize the array to store Mahalanobis distances
    dist = zeros(1, nTrials);
    
    % Calculate the Mahalanobis distance for each trial
    for tr = 1:nTrials
        % Calculate the difference between the signal and mean of noise
        diff = signal(tr, :)' - meanN;
        
        % Calculate the Mahalanobis distance
        dist(tr) = sqrt(diff' / sigma * diff);
    end
end