function ax = visTimeGenAcc1DCluster_v2(decodeStruct,decodeStructShuffle,timeEpoch,options)
% VISTIMEPLOT Visualize channel averaged plot of the entire time series
%   This function generates a channel-averaged plot of the time series data.
%   It visualizes the average accuracy values and significance clusters
%   obtained from decoding analyses.
%
% Arguments:
% - decodeStruct: Cell array containing decoding results for the actual data
% - decodeStructShuffle: Cell array containing decoding results for shuffled data
% - timeEpoch: 2D time values in seconds [start_time end_time] for each epoch
% - options: Structure with optional parameters
%   - pVal2Cutoff: Cutoff value for p-values (default = 0.05)
%   - timePad: Time padding value (default = 0)
%   - clabel: Label for the output value (default = "Output Value")
%   - axisLabel: Label for the y-axis (default = "")
%   - clowLimit: Lower limit for the color (default = 0)
%   - maxVal: Maximum value for scatter plot (default = 1)
%   - chanceVal: Chance value for y-axis line (default = 0.1111)
%   - colval: Color value for plots (default = [0 0 1])
%   - labels: Labels for the three subplots (default = {'Auditory','Go','ResponseOnset'})
%   - tileaxis: Optional pre-defined tile axes (default = [])
%
% Returns:
% - ax: Cell array of axes handles
%
% Example usage:
%   decodeStruct = {...}; % Actual decoding results
%   decodeStructShuffle = {...}; % Shuffled decoding results
%   timeEpoch = [-0.5 2; -0.5 1; -1 1.5]; % Time epochs
%   options = struct('pVal2Cutoff', 0.05, 'timePad', 0, 'clabel', "Output Value", ...);
%   ax = visTimeGenAcc1DCluster_v2(decodeStruct, decodeStructShuffle, timeEpoch, options);


arguments
    decodeStruct   
    decodeStructShuffle
    timeEpoch
    options.pVal2Cutoff double = 0.05;    
    options.timePad double = 0.1;
    options.clabel string = "Output Value"
    options.axisLabel string = ""
    options.clowLimit double = 0
    options.maxVal = 1;
    options.chanceVal = 0.1111;
    options.colval = [0 0 1]
    options.labels = {'Auditory','Go','ResponseOnset'}
    options.tileaxis = [];
end

% Extract options
pVal2Cutoff = options.pVal2Cutoff;
timePad = options.timePad;

timeRange = decodeStruct{1}.timeRange;
timeEpoch = timeEpoch;
dt = diff(timeRange);
fs = 1/dt(1);
acctimeall = [];
acctimeshuffle = [];
pValTime = decodeStruct{1}.pValTime;

% Collect accuracy values for all trials
for iter = 1:length(decodeStruct)
    acctimeall(iter,:) = decodeStruct{iter}.accTime;
    acctimeshuffle(iter,:) = decodeStructShuffle{iter}.accTime;
    ztimeall(iter,:) = norminv(1-decodeStruct{iter}.pValTime);
    ztimeshuffle(iter,:) =  norminv(1-decodeStructShuffle{iter}.pValTime);
end

% Perform permutation test
[clusters, p_values, t_sums, permutation_distribution ] = permutest(acctimeall', acctimeshuffle', false, pVal2Cutoff, 10^5);
sigClusters = find(p_values < pVal2Cutoff);
pmask = zeros(size(timeRange));

% Create a binary mask for significant clusters
for iClust = 1:length(sigClusters)
    pmask(clusters{sigClusters(iClust)}) = 1;
end

hold on;
if(isempty(options.tileaxis))
    t = tiledlayout(1,3,'TileSpacing','compact');
    ax1 =  axes(t);
    ax1.Layout.Tile = 1;
    ax2 =  axes(t);
    ax2.Layout.Tile = 2;
    ax3 =  axes(t);
    ax3.Layout.Tile = 3;
else
    ax1 = options.tileaxis{1};
    ax2 = options.tileaxis{2};
    ax3 = options.tileaxis{3};
end

% Plot the channel-averaged data for each time epoch
sig1M=mean(acctimeall,1); % extract mean
sig1M = smoothdata(sig1M,"gaussian",5);
sig1S=std(acctimeall); % extract standard error

diffTime = diff(timeEpoch');
timeRange1Id = 1:(diffTime(1)-timePad)*fs;
timeRange2Id = round(diffTime(1)*fs+1:(diffTime(1)+diffTime(2)-timePad)*fs);
timeRange3Id = round((diffTime(1)+diffTime(2))*fs+1:(sum(diffTime)-timePad)*fs);

timeGamma1 = linspace(timeEpoch(1,1),timeEpoch(1,2)-timePad,length(timeRange1Id) ) + timePad;
timeGamma2 = linspace(timeEpoch(2,1),timeEpoch(2,2)-timePad,length(timeRange2Id) ) + timePad;
timeGamma3 = linspace(timeEpoch(3,1),timeEpoch(3,2)-timePad,length(timeRange3Id)  ) + timePad;

% Plot the first subplot
hold(ax1,'on')
sig1M2plot = sig1M(:,timeRange1Id);
pmask2plot = pmask(:,timeRange1Id);
sig1S2plot = sig1S(:,timeRange1Id);
h = plot(ax1,timeGamma1,sig1M2plot,'LineWidth',2,Color=options.colval);
hold on;
h = patch(ax1,[timeGamma1,timeGamma1(end:-1:1)],[sig1M2plot + sig1S2plot, ...
sig1M2plot(end:-1:1) - sig1S2plot(end:-1:1)],0.75*options.colval);
set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
scatter(ax1,timeGamma1(find(pmask2plot)),options.maxVal.*ones(1,sum(pmask2plot)),'filled',MarkerEdgeColor=options.colval,MarkerFaceColor=options.colval);
yline(ax1,options.chanceVal, '--','LineWidth',1);
hold on;
xline(ax1,timeGamma1(end),':');
ax1.Box = 'off';
xlim(ax1,[timeGamma1(1) timeGamma1(end)])
xlabel(ax1,options.labels{1})
formatTicks(ax1)
axis square

% Plot the second subplot
hold(ax2,'on')
startTimePoint = round(length(timeGamma1))+1;
sig1M2plot = sig1M(:,timeRange2Id);
pmask2plot = pmask(:,timeRange2Id);
sig1S2plot = sig1S(:,timeRange2Id);
h = plot(ax2,timeGamma2,sig1M2plot,'LineWidth',2,Color=options.colval);
hold on;
scatter(ax2,timeGamma2(find(pmask2plot)),options.maxVal.*ones(1,sum(pmask2plot)),'filled',MarkerEdgeColor=options.colval,MarkerFaceColor=options.colval);
yline(ax2,options.chanceVal, '--','LineWidth',1);
h = patch(ax2,[timeGamma2,timeGamma2(end:-1:1)],[sig1M2plot + sig1S2plot, ...
sig1M2plot(end:-1:1) - sig1S2plot(end:-1:1)],0.75*options.colval);
set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
xline(ax2,timeGamma2(1),':');
xline(ax2,timeGamma2(end),':');
ax2.YAxis.Visible = 'off';
ax2.Box = 'off';
xlim(ax2,[timeGamma2(1) timeGamma2(end)])
xlabel(ax2,options.labels{2})
formatTicks(ax2)
axis square

% Plot the third subplot
hold(ax3,'on')
startTimePoint = startTimePoint+round(length(timeGamma2));
sig1M2plot = sig1M(:,timeRange3Id);
pmask2plot = pmask(:,timeRange3Id);
sig1S2plot = sig1S(:,timeRange3Id);
h = plot(ax3,timeGamma3,sig1M2plot,'LineWidth',2,Color=options.colval);
hold on;
scatter(ax3,timeGamma3(find(pmask2plot)),options.maxVal.*ones(1,sum(pmask2plot)),'filled',MarkerEdgeColor=options.colval,MarkerFaceColor=options.colval);
h = patch(ax3,[timeGamma3,timeGamma3(end:-1:1)],[sig1M2plot + sig1S2plot, ...
sig1M2plot(end:-1:1) - sig1S2plot(end:-1:1)],0.75*options.colval);
set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
xline(ax3,timeGamma3(1),':');
xline(ax3,timeGamma3(end),':');
ax3.YAxis.Visible = 'off';
ax3.Box = 'off';
xlim(ax3,[timeGamma3(1) timeGamma3(end)])
yline(ax3,options.chanceVal, '--','LineWidth',1);
xlabel(ax3,options.labels{3})
formatTicks(ax3)
axis square

ax{1} = ax1;
ax{2} = ax2;
ax{3} = ax3;
% Link the axes
linkaxes([ax1 ax2 ax3], 'y')

end

