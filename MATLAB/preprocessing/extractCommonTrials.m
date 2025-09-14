function goodTrialsCommon = extractCommonTrials(goodTrials)
% extractCommonTrials - Extracts the common trials from multiple sets of good trials.
%
% Syntax: goodTrialsCommon = extractCommonTrials(goodTrials)
%
% Inputs:
%   goodTrials      - Cell array containing multiple sets of good trials
%
% Output:
%   goodTrialsCommon- Array of common trials present in all sets
%
% Example:
%   goodTrials1 = [1, 2, 3, 4, 5];
%   goodTrials2 = [3, 4, 5, 6, 7];
%   goodTrials3 = [4, 5, 6, 7, 8];
%   goodTrials = {goodTrials1, goodTrials2, goodTrials3};
%   goodTrialsCommon = extractCommonTrials(goodTrials);
%
    % Initialize the common trials with the first set of good trials
    goodTrialsCommon = find(goodTrials(1,:));
    
    % Iterate through the remaining sets of good trials
    for na = 2:size(goodTrials,1)
        % Find the intersection of the current set with the common trials
        goodTrialsCommon = intersect(goodTrialsCommon, find(goodTrials(na,:)));
    end
end
