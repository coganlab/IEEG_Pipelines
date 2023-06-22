function ieegSpaceSmooth = spatialSmooth(ieeg, chanMap, window)
% spatialSmooth - Applies spatial smoothing to the iEEG data based on a channel map.
%
% Syntax: ieegSpaceSmooth = spatialSmooth(ieeg, chanMap, window)
%
% Inputs:
%   ieeg        - iEEG data (Channels x Trials x Samples)
%   chanMap     - Channel map indicating the spatial arrangement of channels
%   window      - Window size for spatial smoothing (e.g., [3 3] for a 3x3 window)
%
% Outputs:
%   ieegSpaceSmooth - Spatially smoothed iEEG data
%


ieegSpace = spatialMap(ieeg, chanMap); % Rearrange the iEEG data based on the channel map
h = ones(window) ./ (window(1) .* window(2)); % Create a spatial smoothing filter
ieegSpaceSmooth = imfilter(ieegSpace, h); % Apply spatial smoothing using the filter
ieegSpaceSmooth = reshape(ieegSpaceSmooth, size(ieeg)); % Reshape the smoothed data to match the original size

end
