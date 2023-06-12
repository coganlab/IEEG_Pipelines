function visTimeGenAcc2D(decodeStruct,options)
% Visualizes the generalized accuracy over time for a decoding analysis.
% 
% Arguments:
%   - decodeStruct: A structure containing decoding results.
%   - options: A structure containing optional parameters for visualization.
%     - perc2cutoff: The percentile value to determine the contour cutoff (default: 90).
%     - pVal2Cutoff: The p-value threshold to determine significant regions (default: 0.01).
%     - chanceVal: The value representing chance performance (default: 1).
%     - timePad: The padding added to the time range (default: 0.1).
%     - clabel: The label for the colorbar (default: "Output Value").
%     - axisLabel: The label for the x and y axes (default: "").
%     - clowLimit: The lower limit for the color axis (default: 0).
%
% Note: The decodeStruct should contain the following fields:
%   - timeRange: A vector specifying the time points for analysis.
%   - accTime: A matrix of generalized accuracy values over time.
%   - pValTime: A matrix of p-values over time.
%
% Example usage:
%   decodeStruct.timeRange = [0:0.1:1];
%   decodeStruct.accTime = rand(11);
%   decodeStruct.pValTime = rand(11);
%   options = struct();
%   visTimeGenAcc2D(decodeStruct, options);

arguments
    decodeStruct struct
    options.perc2cutoff double = 90;
    options.pVal2Cutoff double = 0.01;
    options.chanceVal double = 1;
    options.timePad double = 0.1;
    options.clabel string = "Output Value"
    options.axisLabel string = ""
    options.clowLimit double = 0
end

% Set optional parameter values
perc2cutoff = options.perc2cutoff;
pVal2Cutoff = options.pVal2Cutoff;
chanceVal = options.chanceVal;
timeRange = decodeStruct.timeRange + options.timePad;
accTime = decodeStruct.accTime ./ chanceVal;
pValTime = decodeStruct.pValTime;

% Create a meshgrid for contour plotting
[timeGridX, timeGridY] = meshgrid(timeRange);

% Plot the accuracy heatmap
figure; 
imagesc(timeRange, timeRange, accTime);
caxis([options.clowLimit max(accTime(:)) + 0.1]);
hold on;
cb = colorbar;
ylabel(cb, options.clabel);
set(gca, 'YDir', 'normal');
xlabel(['Testing time at ' options.axisLabel ' (s)']);
ylabel(['Training time at ' options.axisLabel ' (s)']);
set(gca, 'FontSize', 15);
title(['Contour at ' num2str(perc2cutoff) ' percentile']);

% Add contour lines at the specified percentile
cutoff = prctile(accTime(:), perc2cutoff);
[~, cont1] = contour(timeGridX, timeGridY, accTime, [cutoff, cutoff]);
cont1.LineWidth = 2;
cont1.LineColor = 'r';
axis square;
axis equal;
formatTicks(gca)

% Plot the p-value mask heatmap
figure; 
imagesc(timeRange, timeRange, accTime);
caxis([options.clowLimit max(accTime(:)) + 0.1]);
hold on;
cb = colorbar;
ylabel(cb, options.clabel);
set(gca, 'YDir', 'normal');
xlabel(['Testing time at ' options.axisLabel ' (s)']);
ylabel(['Training time at ' options.axisLabel ' (s)']);
set(gca, 'FontSize', 15);
title(['Contour at p < ' num2str(pVal2Cutoff)]);

% Generate the p-value mask using cluster correction
pmask = zeros(size(pValTime));
[pmask, corrected_p] = cluster_correction(pValTime, pVal2Cutoff);

% Add contour lines at the significant regions
[~, cont2] = contour(timeGridX, timeGridY, pmask, [1, 1]);
cont2.LineWidth = 2;
cont2.LineColor = 'r';
axis square;
axis equal;
formatTicks(gca)
end
