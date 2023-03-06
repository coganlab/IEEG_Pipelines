function visTimeGenAcc1D(decodeStruct,options)
% Plots the result from 1D temporal generalization
% Significant time-points are marked by orange markers
% Input
% decodeStuct - result after 1D temporal generalization

arguments
    decodeStruct struct    
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

timeRange = decodeStruct.timeRange + options.timePad;

accTime = decodeStruct.accTime;
pValTime = decodeStruct.pValTime;
% figure;
plt = plot(timeRange,accTime,'LineWidth',2,Color=options.colval);
hold on;
[pvalnew,~] = fdr(pValTime,pVal2Cutoff);
scatter(timeRange(pValTime<pvalnew),options.maxVal.*ones(1,sum(pValTime<pvalnew)),'filled',MarkerEdgeColor=plt.Color,MarkerFaceColor=plt.Color);
yline(options.chanceVal, '--','chance','LineWidth',1);
xlabel("Time from " + options.axisLabel + " onset (s)")
ylabel(options.clabel);
set(gca,'FontSize',10);
% axis square;
formatTicks(gca);

end