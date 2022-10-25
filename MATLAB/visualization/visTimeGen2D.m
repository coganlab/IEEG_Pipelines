function visTimeGen2D(decodeStruct,options)
% Plots the result from 2D temporal generalization
% 
% Significant time-points are marked by orange markers
% Input
% decodeStuct - result from 1D temporal generalization

arguments
    decodeStruct struct
    options.perc2cutoff double = 90;
    options.pVal2Cutoff double = 0.01;
    options.chanceVal double = 0.25;
    options.timePad double = 0.1;
    options.clabel string = "Coefficient of Determination"
    options.axisLabel string = ""
    options.clowLimit double = 0
end

perc2cutoff = options.perc2cutoff;
pVal2Cutoff = options.pVal2Cutoff;
chanceVal = options.chanceVal;
timeRange = decodeStruct.timeRange + options.timePad;
r2Time = decodeStruct.r2Time;
pValTime = decodeStruct.pValTime;

[timeGridX,timeGridY] = meshgrid(timeRange);


figure; 
imagesc(timeRange,timeRange,r2Time);
caxis([options.clowLimit max(r2Time(:))+0.1]);
hold on;
cb=colorbar;
ylabel(cb,options.clabel);
set(gca,'YDir', 'normal');
xlabel(['Testing time at ' options.axisLabel ' (s)']);
ylabel(['Training time at ' options.axisLabel ' (s)']);
set(gca,'FontSize',15);
title(['Contour at ' num2str(perc2cutoff) ' percentile' ]);
cutoff = prctile(r2Time(:),perc2cutoff);
[~,cont1] = contour(timeGridX,timeGridY,r2Time,[cutoff ,cutoff]);
cont1.LineWidth = 2;
cont1.LineColor = 'r';
axis square;
axis equal;
formatTicks(gca);

figure; 
imagesc(timeRange,timeRange,r2Time);
caxis([0 max(r2Time(:))+0.1]);
hold on;
cb=colorbar;
ylabel(cb,options.clabel);
set(gca,'YDir', 'normal');
xlabel(['Testing time at ' options.axisLabel ' (s)']);
ylabel(['Training time at ' options.axisLabel ' (s)']);
set(gca,'FontSize',15);
title(['Contour at p<' num2str(pVal2Cutoff)  ]);

[cutoff,~] = fdr(pValTime(:),pVal2Cutoff);
%[~,cont2] = contour(timeGridX,timeGridY,r2Time,[cutoff ,cutoff]);
[~,cont2] = contour(timeGridX,timeGridY,pValTime,[cutoff ,cutoff]);

cont2.LineWidth = 2;
cont2.LineColor = 'r';
axis square;
axis equal;
formatTicks(gca);
end