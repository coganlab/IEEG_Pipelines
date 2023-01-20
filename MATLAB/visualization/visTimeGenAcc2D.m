function visTimeGenAcc2D(decodeStruct,options)
% Plots the result from 2D temporal generalization
% 
% Significant time-points are marked by orange markers
% Input
% decodeStuct - result from 1D temporal generalization
arguments
    decodeStruct struct
    options.perc2cutoff double = 90;
    options.pVal2Cutoff double = 0.01;
    options.chanceVal double = 1;
    options.timePad double = 0.1;
    options.clabel string = "Output Value"
    options.axisLabel string = ""
    options.clowLimit double = 0
end

perc2cutoff = options.perc2cutoff;
pVal2Cutoff = options.pVal2Cutoff;
chanceVal = options.chanceVal;
timeRange = decodeStruct.timeRange + options.timePad;
accTime = decodeStruct.accTime./chanceVal;
pValTime = decodeStruct.pValTime;

[timeGridX,timeGridY] = meshgrid(timeRange);


figure; 
imagesc(timeRange,timeRange,accTime);
caxis([options.clowLimit max(accTime(:))+0.1]);
hold on;
cb=colorbar;
ylabel(cb,options.clabel);
set(gca,'YDir', 'normal');
xlabel(['Testing time at ' options.axisLabel ' (s)']);
ylabel(['Training time at ' options.axisLabel ' (s)']);
set(gca,'FontSize',15);
title(['Contour at ' num2str(perc2cutoff) ' percentile' ]);
cutoff = prctile(accTime(:),perc2cutoff);
[~,cont1] = contour(timeGridX,timeGridY,accTime,[cutoff ,cutoff]);
cont1.LineWidth = 2;
cont1.LineColor = 'r';
axis square;
axis equal;
formatTicks(gca)
figure; 
imagesc(timeRange,timeRange,accTime);
caxis([options.clowLimit max(accTime(:))+0.1]);
hold on;
cb=colorbar;
ylabel(cb,options.clabel);
set(gca,'YDir', 'normal');
xlabel(['Testing time at ' options.axisLabel ' (s)']);
ylabel(['Training time at ' options.axisLabel ' (s)']);
set(gca,'FontSize',15);
title(['Contour at p<' num2str(pVal2Cutoff)  ]);
pmask = zeros(size(pValTime));
for iTrain = 1:size(pValTime,1)
    [cutoff,pmasked] = fdr(pValTime(iTrain,:),pVal2Cutoff);
    pmask(iTrain,:) = pmasked;
end

[~,cont2] = contour(timeGridX,timeGridY,pmask,[1 ,1]);
%[~,cont2] = contour(timeGridX,timeGridY,pValTime,[pVal2Cutoff ,pVal2Cutoff]);

cont2.LineWidth = 2;
cont2.LineColor = 'r';
axis square;
axis equal;
formatTicks(gca)
end