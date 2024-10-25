function [valChannelSmooth, b] = chanViewGrid(val2disp, chanMap, options)
% function to visualize channel activations as a scatter plot
arguments
    val2disp double {mustBeVector}
    chanMap double
    options.selectedChannels double = sort(chanMap(~isnan(chanMap)))';
    options.nonChan logical = isnan(chanMap);
    options.titl {mustBeTextScalar} = 'Channel Map Activation'
    options.cval double = []
    options.pixRange double = []
    options.isSmooth double = 0
end

% Initialize channel activation matrices
chanMap2use = chanMap;
chanMap2use(~ismember(chanMap(:), options.selectedChannels)) = nan;
valChannel = zeros(size(chanMap, 1), size(chanMap, 2));
valChannelNan = nan(size(chanMap, 1), size(chanMap, 2));

% Fill in activation values
for iChan = 1:length(options.selectedChannels)
    [cIndR, cIndC] = find(ismember(chanMap2use, options.selectedChannels(iChan)));
    valChannel(cIndR, cIndC) = val2disp(iChan);
    valChannelNan(cIndR, cIndC) = val2disp(iChan);
end

% Smooth the activation values if requested
if options.isSmooth
    valChannelSmooth = imgaussfilt(valChannel, options.isSmooth);
else
    valChannelSmooth = valChannelNan;
end
valChannelSmooth(options.nonChan) = nan;

% Prepare for scatter plot
if ~isempty(options.pixRange)
    valChannelSmooth = valChannelSmooth(options.pixRange(1):options.pixRange(2), options.pixRange(3):options.pixRange(4));
end

% Get channel coordinates and activations
[channelRows, channelCols] = find(~isnan(valChannelSmooth));
activations = valChannelSmooth(~isnan(valChannelSmooth));

% Normalize activations for scatter plot
markerSizes = log(abs(activations) + 1) * 500; % Adjust the scaling factor as needed
colors = activations;

% Get empty channel coordinates
[emptyRows, emptyCols] = find((isnan(valChannelSmooth))&(~isnan(chanMap)));

% Create scatter plot
figure;
hold on;
scatter(channelCols, channelRows, 200, colors, 'filled');
scatter(emptyCols, emptyRows, 10,  'MarkerEdgeColor', 'none', 'MarkerFaceColor', [0.75 0.75 0.75]); % Empty channels with black outline
% Format plot
axis equal;
axis tight;
axis off;
if ~isempty(options.cval)
    caxis(options.cval);
end
title(options.titl);
colormap(parula(4096));
colorbar;
end
