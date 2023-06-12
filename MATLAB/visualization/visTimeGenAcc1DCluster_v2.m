function ax = visTimeGenAcc1DCluster_v2(decodeStruct,decodeStructShuffle,timeEpoch,options)
%VISTIMEPLOT Visualize channel averaged plot of the entire time series
%   Detailed explanation goes here
% timeEpoch - 3 dimensions of time points 
arguments
    decodeStruct   
    decodeStructShuffle
    timeEpoch % 2D time values in seconds [-0.5000    2.0000;   -0.5000    1.0000;   -1.0000    1.5000]
    options.pVal2Cutoff double = 0.05;    
    options.timePad double = 0;
    options.clabel string = "Output Value"
    options.axisLabel string = ""
    options.clowLimit double = 0
    options.maxVal = 1;
    options.chanceVal = 0.1111;
    options.colval = [0 0 1]
    options.labels = {'Auditory','Go','ResponseOnset'}
    options.tileaxis = []
end

pVal2Cutoff = options.pVal2Cutoff;
timePad = options.timePad;

timeRange = decodeStruct{1}.timeRange;
timeEpoch = timeEpoch;
dt = diff(timeRange);
fs = 1/dt(1);
acctimeall = [];
acctimeshuffle = [];
pValTime = decodeStruct{1}.pValTime;
%[pvalnew,~] = fdr(pValTime,pVal2Cutoff);
ptimemask = (pValTime<pVal2Cutoff);
for iter = 1:length(decodeStruct)
acctimeall(iter,:) = decodeStruct{iter}.accTime;
acctimeshuffle(iter,:) = decodeStructShuffle{iter}.accTime;

ztimeall(iter,:) = norminv(1-decodeStruct{iter}.pValTime);
ztimeshuffle(iter,:) =  norminv(1-decodeStructShuffle{iter}.pValTime);
%acctimeshuffle(iter,:) = options.chanceVal.*ones(1,length(decodeStructShuffle{iter}.accTime));
end
% running permutation test

% [pcorr] = matlab_tfce('independent',1,acctimeall',acctimeshuffle');
% pcorr
[clusters, p_values, t_sums, permutation_distribution ] = permutest( acctimeall', acctimeshuffle',false,pVal2Cutoff,10^5);
p_values
t_sums

sigClusters = find(p_values<pVal2Cutoff);
pmask = zeros(size(timeRange));
for iClust = 1:length(sigClusters)
    pmask(clusters{sigClusters(iClust)}) = 1;
end

%pmask = ptimemask&pmask;

% zthresh = norminv(1-pVal2Cutoff/2);
% [zValsRawAct, pValsRaw, actClust]=timePermCluster(acctimeall,acctimeshuffle,10000,1,zthresh);
% 
%  pmask = zeros(size(timeRange));
% for iClust=1:length(actClust.Size)
%     if actClust.Size{iClust}>actClust.perm95
%         pmask(actClust.Start{iClust}: ...
%             actClust.Start{iClust}+(actClust.Size{iClust}-1)) ...
%             = 1;
%     end
% end



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
    %t = options.tileLayout;
    ax1 = options.tileaxis{1};
    ax2 = options.tileaxis{2};
    ax3 = options.tileaxis{3};
end

    
sig1M=mean(acctimeall,1); % extract mean
sig1M = smoothdata(sig1M,"gaussian",5);
sig1S=std(acctimeall); % extract standard error
  % timeGamma1 = linspace(timeEpoch(1,1),timeEpoch(1,2),(timeEpoch(1,2)-timeEpoch(1,1))*fs );
   diffTime = diff(timeEpoch');
   timeRange1Id = 1:(diffTime(1)-timePad)*fs;
   timeRange2Id = round(diffTime(1)*fs+1:(diffTime(1)+diffTime(2)-timePad)*fs);
   timeRange3Id = round((diffTime(1)+diffTime(2))*fs+1:(sum(diffTime)-timePad)*fs);


%    timeRange1 = timeRange(timeRange<diffTime(1));
%    timeRange2 = timeRange(timeRange>=diffTime(1)&timeRange<(diffTime(1)+diffTime(2)));
%    timeRange3 = timeRange(timeRange>=(diffTime(1)+diffTime(2)));
   timeGamma1 = linspace(timeEpoch(1,1),timeEpoch(1,2)-timePad,length(timeRange1Id) ) + timePad;
   timeGamma2 = linspace(timeEpoch(2,1),timeEpoch(2,2)-timePad,length(timeRange2Id) ) + timePad;
   timeGamma3 = linspace(timeEpoch(3,1),timeEpoch(3,2)-timePad,length(timeRange3Id)  ) + timePad;

   length(timeRange1Id)
   length(timeRange2Id)
   length(timeRange3Id)

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
    %     
    hold on;
    xline(ax1,timeGamma1(end),':');
    ax1.Box = 'off';
    xlim(ax1,[timeGamma1(1) timeGamma1(end)])
    
    xlabel(ax1,options.labels{1})
    
     formatTicks(ax1)
     axis square
    

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

