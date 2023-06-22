function [S,w] = lrrSegment(ieeg)
% lrrSegment - Performs linear regression referencing (LRR) on ECoG data with multiple channels.
%
% Reference:
% Young, D., et al. "Signal processing methods for reducing artifacts in microelectrode brain recordings caused by functional electrical stimulation."
% Journal of neural engineering 15.2 (2018): 026014.
%
% Created by Kumar Duraivel for Viventi and Cogan lab.
%
% Inputs:
%    ieeg - Input signal with 'n' channels and 't' timepoints (n x trials x time)
%
% Outputs:
%    S - Linear regression referenced signal (n x t)
%    w - Least square weights (n x trial x n-1)
%
% Example:
%    ieeg = randn(16, 10, 1000); % Example ECoG data
%    [S, w] = lrrSegment(ieeg); % Apply linear regression referencing
%


n = size(ieeg, 1); % Number of channels
c = 1:n;

for tr = 1:size(ieeg, 2)
    tr
    
    ieegtrial = squeeze(ieeg(:, tr, :)); % Extract EEG data for the current trial
    
    for i = 1:n
        [cw(i, :), lw(i, :)] = wavedec(ieegtrial(i, :), 10, 'db2'); % Perform wavelet decomposition
    end
    
    Strial = zeros(n, size(ieeg, 3));
    
    for wd = 1:11
        parfor i = 1:n
            if (wd ~= 11)
                x(i, :) = wrcoef('d', cw(i, :), lw(i, :), 'db2', wd); % Extract detail coefficients
            else
                x(i, :) = wrcoef('a', cw(i, :), lw(i, :), 'db2', wd-1); % Extract approximation coefficients
            end
        end
        
        parfor i = 1:n % Iterate through channels
            R = x(i, :)'; % Reference channel
            X = x(setdiff(c, i), :)'; % Channels to regress
            wtrial = (X' * X) \ (X' * R); % Least square weights estimation
            Strialwd(i, :) = wtrial * X'; % LRR referencing
            wtrialu(wd, i, :) = wtrial;
        end
        
        Strial = Strial + Strialwd;
    end
    
    S(:, tr, :) = Strial;
    w(:, :, tr, :) = wtrialu;
end

end
