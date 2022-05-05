function [valChannelSmooth,b] = chanView(val2disp,chanMap,selectedChannels,nonChan,titl,cval,pixRange,isSmooth)
% figure;
valChannel = zeros(size(chanMap,1),size(chanMap,2));
valChannelNan = nan(size(chanMap,1),size(chanMap,2));
for iChan = 1 : length(selectedChannels)
        [cIndR, cIndC] = find(ismember(chanMap,selectedChannels(iChan)));
        valChannel(cIndR,cIndC) = val2disp(iChan);
        valChannelNan(cIndR,cIndC) = val2disp(iChan);
end
%nonChan = isnan(valChannelNan);
if(isSmooth)
valChannelSmooth = imgaussfilt(valChannel,isSmooth);
else
valChannelSmooth = valChannelNan;
end
valChannelSmooth(nonChan) = nan;
if(isempty(pixRange))
b = imagesc(valChannelSmooth); 
else
    valChannelSmooth = valChannelSmooth(pixRange(1):pixRange(2),pixRange(3):pixRange(4));
    b = imagesc(valChannelSmooth); 
end
axis equal
axis tight
if(~isempty(cval))
caxis(cval);
end
set(b,'AlphaData',~isnan(valChannelSmooth));
title(titl);
colormap((parula(4096)));
%colorbar;