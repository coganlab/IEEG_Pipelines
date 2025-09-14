function plotRoiHistogramsSpeech(val2plot,labels,colors)
%PLOTROIHISTOGRAMS Summary of this function goes here
%   Detailed explanation goes here


labels2plot = labels;
% Generate figure.
fh = figure();

% Compute axes positions with contigunous edges
n = size(val2plot,1); 
feat = size(val2plot,2); 
margins = [.13 .13 .12 .15]; %left, right, bottom, top
height = (1-sum(margins(3:4)))/n; % height of each subplot
width = 1-sum(margins(1:2)); %width of each sp
vPos = linspace(margins(3),1-margins(4)-height,n); %vert pos of each sp

% Plot the histogram fits (normal density function)
% You can optionally specify the number of bins
% as well as the distribution to fit (not shown,
% see https://www.mathworks.com/help/stats/histfit.html)
% Note that histfit() does not allow the user to specify
% the axes (as of r2019b) which is why we need to create 
% the axes within a loop.
% (more info: https://www.mathworks.com/matlabcentral/answers/279951-how-can-i-assign-a-histfit-graph-to-a-parent-axis-in-a-gui#answer_218699)
% Otherwise we could use tiledlayout() (>=r2019b)
% https://www.mathworks.com/help/matlab/ref/tiledlayout.html
subHand = gobjects(1,n);
histHand = gobjects(feat,2,n);
for i = 1:n
    subHand(i) = axes('position', [margins(1), vPos(n - i + 1), width, height]);
    hold on;
    for j = 1:feat
        sum(isnan(val2plot{i, j}))
        if (~isempty(val2plot{i, j})) & (sum(isnan(val2plot{i, j})<length(val2plot{i, j})))
            valhist = val2plot{i, j};
            if(isnan(valhist))
                continue;
            else
            histHand(j,:,i) = histfit(valhist(~isnan(valhist)),5,'kernel');
            end
%             set(histHand{j,i}(1), 'FaceColor', colors(i,j,:), 'EdgeColor', 'none', 'FaceAlpha', 0.3); % Histogram bars
%             set(histHand{j,i}(2), 'Color', colors(i,j,:)); % Fitted line
        end
    end
end

% Link the subplot x-axes
linkaxes(subHand,'x')

% Extend density curves to edges of xlim and fill.
% This is easier, more readable (and maybe faster) to do in a loop. 
xl = xlim(subHand(end));

for i = 1:n
    % histHand(:,i) = histfit(timeMaxRoi2plot{i},50, 'kernel' );
    for j = 1:feat
         if(~isempty(val2plot{i, j})) & (sum(isnan(val2plot{i, j})<length(val2plot{i, j})))
            if(isnan(val2plot{i, j}))
                continue;
            else
             x = [xl(1),histHand(j,2,i).XData,xl([2,1])]; 
            ybar = histHand(j,2,i).YData;

            y = [0,ybar/sum(ybar),0,0];        
            fillHand = fill(subHand(i),x,y,colors(i,j,:),'FaceAlpha',0.4,'EdgeColor','k','LineWidth',1);   
            hold on;
            end
         end
    end
    % Add vertical ref lines at xtick of bottom axis
   % arrayfun(@(t)xline(subHand(i),t),subHand(1).XTick); %req. >=r2018b
    % Add y axis labels
    ylh = ylabel(subHand(i),labels2plot{i}); 
    set(ylh,'Rotation',0,'HorizontalAlignment','right','VerticalAlignment','middle')
end

% Cosmetics

% Delete histogram bars & original density curves 
delete(histHand)
subHand(1)
% remove axes (all but bottom) and 
% add vertical ref lines at x ticks of bottom axis
set(subHand(1),'Box','off')
 arrayfun(@(i)set(subHand(i),'XTick',[]),1:n-1)
 arrayfun(@(i)set(subHand(i),'YTick',[]),1:n)
set(subHand,'YTick',[])
% set(subHand,'XLim',xl)
end

