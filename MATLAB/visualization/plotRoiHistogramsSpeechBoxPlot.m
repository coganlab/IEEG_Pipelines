function plotRoiHistogramsSpeechBoxPlot(val2plot, labels, colors)
% PLOTROIHISTOGRAMS Plots overlapping box plots for each pair of data sets in val2plot.
% Each row in val2plot should contain two cells, each with a dataset.
% 'labels' are the labels for each subplot.
% 'colors' is a cell array of colors, each element containing two colors for the corresponding datasets.

labels2plot = labels;
% Generate figure.
fh = figure();

% Compute axes positions with contiguous edges
n = size(val2plot, 1); 
feat = size(val2plot, 2); 
margins = [.13 .13 .12 .15]; % left, right, bottom, top
height = (1 - sum(margins(3:4))) / n; % height of each subplot
width = 1 - sum(margins(1:2)); % width of each sp
vPos = linspace(margins(3), 1 - margins(4) - height, n); % vert pos of each sp

subHand = gobjects(1, n);
for i = 1:n
    subHand(i) = axes('position', [margins(1), vPos(n - i + 1), width, height]);
    hold on;
    for j = 1:feat
        if ~isempty(val2plot{i, j}) && sum(~isnan(val2plot{i, j})) > 0
            val = val2plot{i, j};
            boxchart(subHand(i),0.5.*j.*ones(size(val)), val, 'Orientation', 'horizontal',...
                'BoxWidth', 0.5, 'MarkerStyle', 'none', 'BoxFaceColor', colors(i,j,:));
            
        end
    end
    ylim([0 0.75.*feat])
end

% Link the subplot x-axes
linkaxes(subHand, 'x')
xl = xlim(subHand(end));
% Set labels
for i = 1:n
     arrayfun(@(t)xline(subHand(i),t),subHand(1).XTick); %req. >=r2018b
     ylh = ylabel(subHand(i),labels2plot{i}); 
    set(ylh,'Rotation',0,'HorizontalAlignment','right','VerticalAlignment','middle')
end
subHand(end)
% Cosmetic adjustments
set(subHand(1),'Box','off')
  arrayfun(@(i)set(subHand(i),'XTickLabel',[]),1:n-1)
  arrayfun(@(i)set(subHand(i),'XTick',[]),1:n-1)
set(subHand,'YTick',[])



hold off;
end
