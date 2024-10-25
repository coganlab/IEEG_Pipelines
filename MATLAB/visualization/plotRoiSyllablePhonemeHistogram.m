function plotRoiSyllablePhonemeHistogram(val2plot, labels, colors)
% PLOTROIHISTOGRAMS Plots overlapping histograms for each pair of data sets in val2plot.
% Each row in val2plot should contain two cells, each with a dataset.
% 'labels' are the labels for each subplot.
% 'colors' is a cell array of colors, each element containing two colors for the corresponding datasets.

timeMaxRoi2plot = val2plot; 
labels2plot = labels;
% Generate figure.
fh = figure();

% Compute axes positions with contiguous edges
n = size(timeMaxRoi2plot,1); 
margins = [.13 .13 .12 .15]; % left, right, bottom, top
height = (1 - sum(margins(3:4))) / n; % height of each subplot
width = 1 - sum(margins(1:2)); % width of each sp
vPos = linspace(margins(3), 1 - margins(4) - height, n); % vert pos of each sp

subHand = gobjects(1, n);
% histHand = gobjects(2, n);

% Loop through each pair of datasets
for i = 1:n
    subHand(i) = axes('position', [margins(1), vPos(n - i + 1), width, height]);
    hold on;
    for j = 1:2
        if (~isempty(timeMaxRoi2plot{i, j})) &(~isnan(timeMaxRoi2plot{i, j}))
            histHand{j,i} = histfit(timeMaxRoi2plot{i, j}, 50, 'kernel');
            set(histHand{j,i}(1), 'FaceColor', colors(i,j,:), 'EdgeColor', 'none', 'FaceAlpha', 0.3); % Histogram bars
            set(histHand{j,i}(2), 'Color', colors(i,j,:)); % Fitted line
        end
    end
end

% Link the subplot x-axes
linkaxes(subHand, 'x')

% Cosmetic adjustments
arrayfun(@(i)set(subHand(i).XAxis, 'Visible', 'off'), 1:n - 1);
set(subHand(n), 'XAxis', 'Visible', 'on'); % Show x-axis only on the bottom subplot
set(subHand, 'YTick', []);

% Set labels
for i = 1:n
    ylabel(subHand(i), labels2plot{i}, 'Rotation', 0, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
end

hold off;
end
