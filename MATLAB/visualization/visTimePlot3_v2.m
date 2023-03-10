function t = visTimePlot3_v2(timeEpoch,signal2plot,options)
%VISTIMEPLOT Visualize channel averaged plot of the entire time series
%   Detailed explanation goes here
% timeEpoch - 3 dimensions of time points 
arguments
    timeEpoch double % 2D time values in seconds [-0.5000    2.0000;   -0.5000    1.0000;   -1.0000    1.5000]
    signal2plot double % 2D -  channels x timeseries
   
    options.fs = 200; % sampling frequency
    options.labels = {'Auditory','Go','ResponseOnset'}
    options.tileLayout = []
end
fs = options.fs;


figure;
hold on;
if(isempty(options.tileLayout))
    t = tiledlayout(1,3,'TileSpacing','compact');
else
    t = options.tileLayout;
end
sig1M=(signal2plot); % extract mean
%sig1S=std(signal2plot)./sqrt(size(signal2plot,1)); % extract standard error
   timeGamma1 = linspace(timeEpoch(1,1),timeEpoch(1,2),(timeEpoch(1,2)-timeEpoch(1,1))*fs );
   timeGamma2 = linspace(timeEpoch(2,1),timeEpoch(2,2),(timeEpoch(2,2)-timeEpoch(2,1))*fs);
   timeGamma3 = linspace(timeEpoch(3,1),timeEpoch(3,2),(timeEpoch(3,2)-timeEpoch(3,1))*fs );

    ax1 =  axes(t);
    ax1.Layout.Tile = 1;

    sig1M2plot = sig1M(:,1:round(length(timeGamma1)));
    %sig1S2plot = sig1S(1:round(length(timeGamma1)));
    h = plot(ax1,timeGamma1,sig1M2plot,'LineWidth',2);
    hold on;
%     h = patch(ax1,[timeGamma1,timeGamma1(end:-1:1)],[sig1M2plot + sig1S2plot, ...
%     sig1M2plot(end:-1:1) - sig1S2plot(end:-1:1)],0.5*colval);
%     set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
    hold on;
    xline(ax1,timeGamma1(end),':');
    ax1.Box = 'off';
    xlim(ax1,[timeGamma1(1) timeGamma1(end)])
    
    xlabel(ax1,options.labels{1})
    
     formatTicks(ax1)
     axis square
    

    ax2 =  axes(t);
    ax2.Layout.Tile = 2;
    startTimePoint = round(length(timeGamma1))+1;
    sig1M2plot = sig1M(:,startTimePoint:startTimePoint+length(timeGamma2)-1);
    %sig1S2plot = sig1S(startTimePoint:startTimePoint+length(timeGamma2)-1);
    h = plot(ax2,timeGamma2,sig1M2plot,'LineWidth',2);
%     h = patch(ax2,[timeGamma2,timeGamma2(end:-1:1)],[sig1M2plot + sig1S2plot, ...
%     sig1M2plot(end:-1:1) - sig1S2plot(end:-1:1)],0.5*colval);
%     set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
    xline(ax2,timeGamma2(1),':');
    xline(ax2,timeGamma2(end),':');
    ax2.YAxis.Visible = 'off';
    ax2.Box = 'off';
    xlim(ax2,[timeGamma2(1) timeGamma2(end)])
  
    
    xlabel(ax2,options.labels{2})
    
    
   formatTicks(ax2)
     axis square
    ax3 =  axes(t);
    ax3.Layout.Tile = 3;
    startTimePoint = startTimePoint+round(length(timeGamma2));
    sig1M2plot = sig1M(:,startTimePoint:end);
    %sig1S2plot = sig1S(startTimePoint:end);
    h = plot(ax3,timeGamma3,sig1M2plot,'LineWidth',2);
%      h = patch(ax3,[timeGamma3,timeGamma3(end:-1:1)],[sig1M2plot + sig1S2plot, ...
%     sig1M2plot(end:-1:1) - sig1S2plot(end:-1:1)],0.5*colval);
%      set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
    xline(ax3,timeGamma3(1),':');
    xline(ax3,timeGamma3(end),':');
    ax3.YAxis.Visible = 'off';
    ax3.Box = 'off';
    xlim(ax3,[timeGamma3(1) timeGamma3(end)])

    
    xlabel(ax3,options.labels{3})
    
    formatTicks(ax3)
     axis square
% Link the axes
    linkaxes([ax1 ax2 ax3], 'y')
end

