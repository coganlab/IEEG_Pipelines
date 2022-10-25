function visTimeGen1D(decodeStruct,options)
% Plots the result from 1D temporal generalization
% Significant time-points are marked by orange markers
% Input
% decodeStuct - result after 1D temporal generalization

arguments
    decodeStruct struct    
    options.pVal2Cutoff double = 0.01;    
    options.timePad double = 0.1;
    options.clabel string = "Output Value"
    options.axisLabel string = ""
    options.clowLimit double = 0
end

pVal2Cutoff = options.pVal2Cutoff;

timeRange = decodeStruct.timeRange + options.timePad;

r2Time = decodeStruct.r2Time;
pValTime = decodeStruct.pValTime;
figure;
plot(timeRange,r2Time,'LineWidth',2);
hold on;
%[pvalnew,~] = fdr(pValTime,pVal2Cutoff);
scatter(timeRange(pValTime<pVal2Cutoff),(max(r2Time)+0.1).*ones(1,sum(pValTime<pVal2Cutoff)),'filled');
xlabel("Time from " + options.axisLabel + " onset (s)")
ylabel('Coefficient of Determination');
set(gca,'FontSize',15);
axis square;
formatTicks(gca);

end