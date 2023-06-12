function [valChannelSmooth,b] = chanView(val2disp,chanMap, options)
% figure;
arguments
    val2disp double{mustBeVector} 
    chanMap double
    options.selectedChannels double = sort(chanMap(~isnan(chanMap)))';
    options.nonChan logical = isnan(chanMap);
    options.titl {mustBeTextScalar} = 'channel map activation'
    options.cval double = []
    options.pixRange double = []
    options.isSmooth logical = 0
end
%selectedChannels = sort(chanMap(~isnan(chanMap)))';
%val2disp(~ismember(chanMap(:),options.selectedChannels)) = [];
chanMap(~ismember(chanMap(:),options.selectedChannels))= nan;
valChannel = zeros(size(chanMap,1),size(chanMap,2));
valChannelNan = nan(size(chanMap,1),size(chanMap,2));
for iChan = 1 : length(options.selectedChannels)
        [cIndR, cIndC] = find(ismember(chanMap,options.selectedChannels(iChan)));
        valChannel(cIndR,cIndC) = val2disp(iChan);
        valChannelNan(cIndR,cIndC) = val2disp(iChan);
end
%nonChan = isnan(valChannelNan);
if(options.isSmooth)
    valChannelSmooth = imgaussfilt(valChannel,options.isSmooth);
else
    valChannelSmooth = valChannelNan;
end
valChannelSmooth(options.nonChan) = nan;
if(isempty(options.pixRange))
b = imagesc(valChannelSmooth); 
else
    valChannelSmooth = valChannelSmooth(options.pixRange(1):options.pixRange(2),options.pixRange(3):options.pixRange(4));
    b = imagesc(valChannelSmooth); 
end
axis equal
axis tight
if(~isempty(options.cval))
caxis(options.cval);
end
set(b,'AlphaData',~isnan(valChannelSmooth));
title( options.titl);
colormap((parula(4096)));
%colorbar;