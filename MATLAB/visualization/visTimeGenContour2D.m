function visTimeGenContour2D(decodeStructArray,options)
% Plots the result from 2D temporal generalization
% 
% Significant time-points are marked by orange markers
% Input
% decodeStuct - result from 1D temporal generalization
arguments
    decodeStructArray
    
    options.pVal2Cutoff double = 0.05;
    
    options.timePad double = 0.1;
    options.clabel string = "Coefficient of Determination"
    options.axisLabel string = ""    
end

numArray = length(decodeStructArray);
colors = lines(numArray);
figure;
for iArray = 1:numArray
    decodeStruct = decodeStructArray{iArray};
    timeRange = decodeStruct.timeRange + options.timePad;
    pValTime = decodeStruct.pValTime;
    [timeGridX,timeGridY] = meshgrid(timeRange);
    pVal2Cutoff = options.pVal2Cutoff;
    [cutoff,~] = fdr(pValTime(:),pVal2Cutoff)
    [~,cont2] = contour(timeGridX,timeGridY,pValTime,[cutoff ,cutoff]);
    set(gca,'YDir', 'normal');
    xlabel(['Testing time at ' options.axisLabel ' (s)']);
    ylabel(['Training time at ' options.axisLabel ' (s)']);
    cont2.LineWidth = 2;
    cont2.LineColor = colors(iArray,:);
    axis square;
    axis equal;
    formatTicks(gca);
    hold on;
end

end