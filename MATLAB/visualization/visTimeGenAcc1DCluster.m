function [ax,accResults] = visTimeGenAcc1DCluster(decodeStruct,decodeStructShuffle,options)
% Plots the result from 1D temporal generalization
% Significant time-points are marked by orange markers
% Input
% decodeStuct - result after 1D temporal generalization

arguments
    decodeStruct   
    decodeStructShuffle
    options.pVal2Cutoff double = 0.05;  
    options.perc2cutoff double = 90;
    options.timePad double = 0.1;
    options.clabel string = "Output Value"
    options.axisLabel string = ""
    options.clowLimit double = 0
    options.maxVal = 1;
    options.chanceVal = 0.1111;
    options.colval = [0 0 1];    
    options.tileaxis = [];
    options.showShuffle = 1;
    options.showPeaks = 5;
    options.boxPlotPlace = -0.1
    options.boxPlotStd = 0.1;
    options.showAccperChance = 0
    options.searchRange = [];
end

pVal2Cutoff = options.pVal2Cutoff;
perc2cutoff = options.perc2cutoff;
timeRange = decodeStruct{1}.timeRange + options.timePad;
acctimeall = [];
acctimeshuffle = [];

 if(~isempty(options.searchRange))
    timeSelect = timeRange>=options.searchRange(1) & timeRange<=options.searchRange(2);
    timeRangeSelect = timeRange(timeSelect);
else
    timeSelect = true(1,length(timeRange));
    timeRangeSelect = timeRange;
end
for iter = 1:length(decodeStruct)
    if(isfield(decodeStruct{iter},'accTime'))
        if(options.showAccperChance)
            acctimeall(iter,:) = (decodeStruct{iter}.accTime-options.chanceVal)./(1-options.chanceVal);
        else
            acctimeall(iter,:) = decodeStruct{iter}.accTime;
        end
    else
        acctimeall(iter,:) = decodeStruct{iter}.r2Time;
    end
   
 ztimeall(iter,:) = norminv(1-decodeStruct{iter}.pValTime);
    
   
    %acctimeshuffle(iter,:) = options.chanceVal.*ones(1,length(decodeStructShuffle{iter}.accTime));
end

for iter = 1:length(decodeStructShuffle)
    if(isfield(decodeStructShuffle{iter},'accTime'))
        
        if(options.showAccperChance)
            acctimeshuffleOrig(iter,:) = (decodeStructShuffle{iter}.accTime-options.chanceVal)./(1-options.chanceVal);
        else
            acctimeshuffleOrig(iter,:) = decodeStructShuffle{iter}.accTime;
        end
    else
        acctimeshuffleOrig(iter,:) = decodeStructShuffle{iter}.r2Time;
    end
%     acctimeshuffleOrig(iter,:) = decodeStructShuffle{iter}.accTime;
 acctimeshuffleSmooth(iter,:) = smoothdata(acctimeshuffleOrig(iter,:),"gaussian",50);
    ptimeshuffleOrig(iter,:) = decodeStructShuffle{iter}.pValTime;
%      timeCentroidIter = sum(acctimeshuffleOrig(iter,:).*(1:length(timeRange)))/sum(acctimeshuffleOrig(iter,:))
%     accResults.timeCentroidShuffle(iter) = timeRange(round(timeCentroidIter));
%     accResults.accCentroidShuffle(iter) = acctimeshuffleOrig(iter,round(timeCentroidIter));    
    [maxAcc,maxId] = max(acctimeshuffleOrig(iter,:));
    accResults.timeMaxShuffle(iter) = timeRange(maxId);
    accResults.accMaxShuffle(iter) = maxAcc;
   
     diffTimeAcc = diff(acctimeshuffleSmooth(iter,:));
    [maxRiseAcc,maxRiseId] = max(diffTimeAcc);

    accResults.timeRiseShuffle(iter) = timeRange(maxRiseId+1);
    accResults.accRiseShuffle(iter) = acctimeshuffleSmooth(iter,maxRiseId+1);


    [minValue,closestIndex] = min(abs(acctimeshuffleOrig(iter,:)-(maxAcc/2)));
    if(~isempty(closestIndex))
        accResults.timeHalfMaxShuffle(iter) = timeRange(closestIndex);
        accResults.accHalfMaxShuffle(iter) = maxAcc/2;
    end

    idx = findchangepts(acctimeshuffleOrig(iter,timeSelect),MaxNumChanges=1,Statistic="mean");
    if(~isempty(idx))
        accResults.timeCP1Shuffle(iter) = timeRangeSelect(idx(1)+1);
        accResults.accCP1Shuffle(iter) = acctimeshuffleOrig(iter,find(timeRange>=timeRangeSelect(idx(1)),1));
    end

end


acctimeshuffle = [];
ztimeshuffle = [];
for iter = 1:1000
    trials2select = randperm(length(decodeStructShuffle),length(decodeStruct));
    acctimeshuffle(iter,:) = mean(acctimeshuffleOrig(trials2select,:));
   
end
% acctimenorm = acctimeall - mean(acctimeshuffle,1);
% acctimeshuffnorm = acctimeshuffle - mean(acctimeshuffle,1);
acctimenorm = acctimeall;
acctimeshuffnorm = acctimeshuffle;
% running permutation test
% [clusters, p_values, t_sums, permutation_distribution ] = permutest( acctimenorm', acctimeshuffnorm',true,pVal2Cutoff,10000);
% sigClusters = find(p_values<pVal2Cutoff);
% pmask = zeros(size(timeRange));
% for iClust = 1:length(sigClusters)
%     pmask(clusters{sigClusters(iClust)}) = 1;
% end
pmask = zeros(size(timeRange));
zmean = mean(ztimeall,1);
pValTime = 1-normcdf(zmean);
%[~, ~, actClust]=timePermCluster(acctimenorm,acctimeshuffnorm(trials2select,:),1000,1,norminv(1-pVal2Cutoff));

%[~, ~, actClust]=timePermClusterAfterPerm(mean(acctimenorm,1),acctimeshuffnorm,1,norminv(1-pVal2Cutoff));
[~, actClust] = timePermClusterAfterPermPValues(pValTime, ptimeshuffleOrig, pVal2Cutoff);
for iClust=1:length(actClust.Size)
    if actClust.Size{iClust}>actClust.perm95
        pmask(actClust.Start{iClust}: ...
            actClust.Start{iClust}+(actClust.Size{iClust}-1)) ...
            = 1;
    end
end

% cutoff = prctile(acctimeall(:), perc2cutoff)
acctimeallmasked = acctimeall.*pmask;
if(sum(pmask)~=0)

for iter = 1:size(acctimeallmasked,1)
 acctimeSmooth(iter,:) = smoothdata(acctimeall(iter,:),"gaussian",10);
    acctimeSmooth(iter,:) =  acctimeSmooth(iter,:).*pmask;
    
   % timeCentroidIter = sum(acctimeall(iter,timeSelect).*timeRangeSelect)/sum(acctimeall(iter,timeSelect))
    
    [maxAcc,maxId] = max(acctimeallmasked(iter,timeSelect));

    

    diffTimeAcc = diff(acctimeSmooth(iter,timeSelect));
    
    timeCentroidIter = calculateTimeCentroid(timeRangeSelect,acctimeallmasked(iter,timeSelect));
    accResults.timeCentroid(iter) = timeCentroidIter;
    accResults.accCentroid(iter) = acctimeallmasked(iter,find(timeRange>=timeCentroidIter,1));
    accResults.timeMax(iter) = timeRangeSelect(maxId);
    accResults.accMax(iter) = maxAcc;

    [maxRiseAcc,maxRiseId] = max(diffTimeAcc);
    accResults.timeRise(iter) = timeRangeSelect(maxRiseId+1);
    accResults.accRise(iter) = acctimeSmooth(iter,find(timeRange>=timeRangeSelect(maxRiseId+1),1));

    % finding the closest value
    %[minValue,closestIndex] = min(abs(acctimeallmasked(iter,timeSelect)-(maxAcc/2)));
    closestIndex = find((acctimeall(iter,timeSelect)>=maxAcc/2),1);
    accResults.timeHalfMax(iter) = timeRangeSelect(closestIndex);
    accResults.accHalfMax(iter) = maxAcc/2;

    idx = findchangepts(acctimeallmasked(iter,timeSelect),Statistic="mean", maxNumChanges = 2 );
    if(~isempty(idx))
        accResults.timeCP1(iter) = timeRangeSelect(idx(1)+1);
        accResults.accCP1(iter) = acctimeSmooth(iter,find(timeRange>=timeRangeSelect(idx(1)),1));
    else
        accResults.timeCP1(iter) = nan;
        accResults.accCP1(iter) = nan;
    end
end
end

if(isempty(options.tileaxis))
    ax =gca;
else
    ax = options.tileaxis;    
end

 pmask = find(pmask) ;

%pValTime = decodeStruct.pValTime;
% figure;

sig1M=mean(acctimeall,1); % extract mean
% pmask = find(sig1M >= cutoff);
%sig1M = smoothdata(sig1M,"gaussian",5);
sig1S=std(acctimeall); % extract standard error

plt = plot(ax,timeRange,sig1M,'LineWidth',2,Color=options.colval);
hold on;
h = patch(ax,[timeRange,timeRange(end:-1:1)],[sig1M + sig1S, ...
sig1M(end:-1:1) - sig1S(end:-1:1)],0.75*options.colval);
set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
hold on;
scatter(ax,timeRange((pmask)),options.maxVal.*ones(1,length(pmask)),'filled',MarkerEdgeColor=plt.Color,MarkerFaceColor=plt.Color);
if(options.showAccperChance) 
    yline(0, '--','chance','LineWidth',1);
else
    yline(options.chanceVal, '--','chance','LineWidth',1);
end
if(options.showShuffle)
    sig1MShuffle=mean(acctimeshuffle,1); % extract mean
    % pmask = find(sig1M >= cutoff);
    sig1MShuffle = smoothdata(sig1MShuffle,"gaussian",5);
    sig1SShuffle=std(acctimeshuffle); % extract standard error
    
    plt = plot(ax,timeRange,sig1MShuffle,'LineWidth',1,Color=[0 0 0]);
    hold on;
    h = patch(ax,[timeRange,timeRange(end:-1:1)],[sig1MShuffle + sig1SShuffle, ...
    sig1MShuffle(end:-1:1) - sig1SShuffle(end:-1:1)],[0.75 0.75 0.75]);
    set(h,'FaceAlpha',.5,'EdgeAlpha',0,'Linestyle','none');
end
if ~isempty(pmask)
    if(options.showPeaks)
        switch options.showPeaks
            case 1 
                 boxchart(options.boxPlotPlace*ones(size(accResults.timeMax)), double(accResults.timeMax),'orientation','horizontal','BoxFaceColor',options.colval*0.75,'BoxWidth',options.boxPlotStd, 'BoxLineColor',options.colval,'MarkerStyle','none')
            case 2 
                 boxchart(options.boxPlotPlace*ones(size(accResults.timeCentroid)), double(accResults.timeCentroid),'orientation','horizontal','BoxFaceColor',options.colval*0.75,'BoxWidth',options.boxPlotStd, 'BoxLineColor',options.colval,'MarkerStyle','none')
            case 3 
                 boxchart(options.boxPlotPlace*ones(size(accResults.timeRise)), double(accResults.timeRise),'orientation','horizontal','BoxFaceColor',options.colval*0.75,'BoxWidth',options.boxPlotStd, 'BoxLineColor',options.colval,'MarkerStyle','none')
            case 4 
                 boxchart(options.boxPlotPlace*ones(size(accResults.timeHalfMax)), double(accResults.timeHalfMax),'orientation','horizontal','BoxFaceColor',options.colval*0.75,'BoxWidth',options.boxPlotStd, 'BoxLineColor',options.colval,'MarkerStyle','none')
            case 5 
                 boxchart(options.boxPlotPlace*ones(size(accResults.timeCP1)), double(accResults.timeCP1),'orientation','horizontal','BoxFaceColor',options.colval*0.75,'BoxWidth',options.boxPlotStd, 'BoxLineColor',options.colval,'MarkerStyle','none')
        
        end
    end
end


xlabel("Time from " + options.axisLabel + " onset (s)")
ylabel(options.clabel);
set(ax,'FontSize',10);
% axis square;
formatTicks(ax);

end