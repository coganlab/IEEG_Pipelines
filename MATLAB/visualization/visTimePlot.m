function visTimePlot(timeEpoch,signal2plot,options)
%VIS_TIME_PLOT Visualizes time-series data with optional error shading.
%
%   This function plots time-series data with options to include error shading
%   representing the standard error of the mean. The plot can be customized with
%   various options including color, sampling frequency, and labels.
%
%   Arguments:
%       timeEpoch (double): A vector specifying the time points for the x-axis. If
%           timeEpoch has two elements, it is treated as the start and end times,
%           and the time axis is linearly spaced.
%
%       signal2plot (double): Matrix of the signals to plot. Rows correspond to
%           different trials and columns correspond to time points.
%
%       options (struct): Struct to specify the following plot options:
%           - colval (array): RGB color values for the plot. Default is [1 0 1] (magenta).
%           - fs (double): Sampling frequency of the signal. Default is 100 Hz.
%           - labels (cell array of strings): Labels for the x-axis. Default is {'Auditory'}.
%           - tileLayout (optional): Specifies the tile layout for subplotting. Default is empty.
%           - showStd (logical): If true, plots the standard error around the mean signal. Default is true (1).
%
%   Example Usage:
%       % Define time and signals
%       timeEpoch = [0 100];  % Time from 0 to 100 seconds
%       signal2plot = randn(50, 1000);  % Random data for 50 trials and 1000 time points
%       options = struct('colval', [0 0 1], 'fs', 250, 'labels', {'Response'});
%       
%       % Call the function
%       visTimePlot(timeEpoch, signal2plot, options);
%
%   Notes:
%       - Ensure the number of columns in signal2plot matches the number of
%         points defined in timeEpoch when it is a full vector.


arguments
    timeEpoch double
    signal2plot double 
    options.colval = [1 0 1]; 
    options.fs = 100; 
    options.labels = {'Auditory'}
    options.tileLayout = []
    options.showStd = 1;
end

colval = options.colval;

% Initialize plot
hold on;

% Compute mean and standard error of the signal
sig1M = nanmean(signal2plot, 1);  % extract mean
sig1S = nanstd(signal2plot) ./ sqrt(size(signal2plot, 1));  % extract standard error

% Determine time values based on timeEpoch input
if length(timeEpoch) == 2
    tw = timeEpoch;
    timeValues = linspace(tw(1), tw(2), size(signal2plot, 2));
else
    timeValues = timeEpoch;
end

% Plot the mean signal
sig1M2plot = sig1M(1:round(length(timeValues)));
sig1S2plot = sig1S(1:round(length(timeValues)));
h = plot(timeValues, sig1M2plot, 'LineWidth', 2, 'Color', colval);

% Conditionally plot standard error shading
if options.showStd
    h = patch([timeValues, timeValues(end:-1:1)], ...
              [sig1M2plot + sig1S2plot, sig1M2plot(end:-1:1) - sig1S2plot(end:-1:1)], ...
              0.5 * colval);
    set(h, 'FaceAlpha', 0.5, 'EdgeAlpha', 0, 'Linestyle', 'none');
end

% Finalize plot settings
xlim([timeValues(1) timeValues(end)]);
xlabel(options.labels);
formatTicks(gca);

end