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
