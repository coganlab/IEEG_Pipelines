function ieegSpaceArrange = spatialMap(ieeg, chanMap)
% spatialMap - Rearranges the iEEG data based on a channel map.
%
% Syntax: ieegSpaceArrange = spatialMap(ieeg, chanMap)
%
% Inputs:
%   ieeg        - iEEG data (Channels x Trials x Samples)
%   chanMap     - Channel map indicating the spatial arrangement of channels
%
% Outputs:
%   ieegSpaceArrange - Rearranged iEEG data based on the channel map
%

selectedChannels = sort(chanMap(~isnan(chanMap)))'; % Select channels based on the channel map and sort them
ieegSpaceArrange = nan(size(chanMap, 1), size(chanMap, 2), size(ieeg, 2)); % Initialize the rearranged iEEG data matrix

for c = 1:length(selectedChannels)
    % Find the row and column indices of the selected channel in the channel map
    [cIndR, cIndC] = find(ismember(chanMap, selectedChannels(c))); 
    % Assign the iEEG data to the corresponding location in the rearranged matrix
    ieegSpaceArrange(cIndR, cIndC, :) = ieeg(c, :); 
end

end
