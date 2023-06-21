function visTimePlot(timeEpoch,signal2plot,options)
%VISTIMEPLOT Visualize channel averaged plot of the entire time series
%   Detailed explanation goes here
% timeEpoch - 3 dimensions of time points 
arguments
    timeEpoch double % 2D time values in seconds [-0.5000    2.0000;   -0.5000    1.0000;   -1.0000    1.5000]
    signal2plot double % 2D -  channels x timeseries
    options.colval = [1 0 1]; % color value
    options.fs = 200; % sampling frequency
    options.labels = {'Auditory'}
    options.tileLayout = []
end
fs = options.fs;
colval = options.colval;
% scrsize = get(0, 'Screensize');
% 
% figure('Position', [scrsize(1) scrsize(2) scrsize(3) scrsize(4)/2]);
hold on;

sig1M=mean(signal2plot); % extract mean
sig1S=std(signal2plot)./sqrt(size(signal2plot,1)); % extract standard error
 tw = [0 sum(diff(timeEpoch'))]+timeEpoch(1,1);
timeGamma1 = linspace(tw(1),tw(2),(tw(2)-tw(1))*fs );
  
    

    sig1M2plot = sig1M(1:round(length(timeGamma1)));
    sig1S2plot = sig1S(1:round(length(timeGamma1)));
    h = plot(timeGamma1,sig1M2plot,'LineWidth',2,'Color',colval);
    hold on;
    h = patch([timeGamma1,timeGamma1(end:-1:1)],[sig1M2plot + sig1S2plot, ...
    sig1M2plot(end:-1:1) - sig1S2plot(end:-1:1)],0.5*colval);
    set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
    hold on;
    
    xlim([timeGamma1(1) timeGamma1(end)])    
    xlabel(options.labels{1})
    formatTicks(gca)
    
     
end

