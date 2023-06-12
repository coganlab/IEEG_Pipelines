function visTimeGenAcc1DCluster(decodeStruct,decodeStructShuffle,options)
% Plots the result from 1D temporal generalization
% Significant time-points are marked by orange markers
% Input
% decodeStuct - result after 1D temporal generalization

arguments
    decodeStruct   
    decodeStructShuffle
    options.pVal2Cutoff double = 0.05;    
    options.timePad double = 0.1;
    options.clabel string = "Output Value"
    options.axisLabel string = ""
    options.clowLimit double = 0
    options.maxVal = 1;
    options.chanceVal = 0.1111;
    options.colval = [0 0 1]
end

pVal2Cutoff = options.pVal2Cutoff;

timeRange = decodeStruct{1}.timeRange + options.timePad;
acctimeall = [];
acctimeshuffle = [];
pValTime = decodeStruct{1}.pValTime;
%[pvalnew,~] = fdr(pValTime,pVal2Cutoff);
ptimemask = (pValTime<pValTime);
for iter = 1:length(decodeStruct)
acctimeall(iter,:) = decodeStruct{iter}.accTime;
acctimeshuffle(iter,:) = decodeStructShuffle{iter}.accTime;

ztimeall(iter,:) = norminv(1-decodeStruct{iter}.pValTime);
ztimeshuffle(iter,:) =  norminv(1-decodeStructShuffle{iter}.pValTime);
%acctimeshuffle(iter,:) = options.chanceVal.*ones(1,length(decodeStructShuffle{iter}.accTime));
end
% running permutation test
[clusters, p_values, t_sums, permutation_distribution ] = permutest( acctimeall', acctimeshuffle',false,pVal2Cutoff);
sigClusters = find(p_values<pVal2Cutoff);
pmask = zeros(size(timeRange));
for iClust = 1:length(sigClusters)
    pmask(clusters{sigClusters(iClust)}) = 1;
end
% zthresh = norminv(1-pVal2Cutoff);
% 
% [zValsRawAct, pValsRaw, actClust]=timePermCluster(acctimeall,acctimeshuffle,1000,1,zthresh);
% 
%  pmask = zeros(size(timeRange));
% for iClust=1:length(actClust.Size)
%     if actClust.Size{iClust}>actClust.perm95
%         pmask(actClust.Start{iClust}: ...
%             actClust.Start{iClust}+(actClust.Size{iClust}-1)) ...
%             = 1;
%     end
% end
%pmask = pmask & ptimemask;
%sigClusters = find(p_values<pVal2Cutoff);
%pmask = zeros(size(timeRange));
% for iClust = 1:length(sigClusters)
%     pmask(clusters{sigClusters(iClust)}) = 1;
% end

pmask = find(pmask) ;

%pValTime = decodeStruct.pValTime;
% figure;
plt = plot(timeRange,mean(acctimeall,1),'LineWidth',2,Color=options.colval);
hold on;
%[pvalnew,~] = fdr(pValTime,pVal2Cutoff);
%[pmask,pvalnew] = cluster_correction(pValTime,pVal2Cutoff);
%scatter(timeRange(pValTime<pvalnew),options.maxVal.*ones(1,sum(pValTime<pvalnew)),'filled',MarkerEdgeColor=plt.Color,MarkerFaceColor=plt.Color);
scatter(timeRange((pmask)),options.maxVal.*ones(1,length(pmask)),'filled',MarkerEdgeColor=plt.Color,MarkerFaceColor=plt.Color);
yline(options.chanceVal, '--','chance','LineWidth',1);
xlabel("Time from " + options.axisLabel + " onset (s)")
ylabel(options.clabel);
set(gca,'FontSize',10);
% axis square;
formatTicks(gca);

end