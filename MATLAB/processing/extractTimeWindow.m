function ieegTime = extractTimeWindow(ieegData,winSize,winRes)
%EXTRACTTIMEWINDOW Summary of this function goes here
%   Detailed explanation goes here
nTime = size(ieegData,2)/winRes;

for iTime = 1:nTime
    if(iTime*winSize>size(ieegData,2))
        break
    else
        ieegTime(iTime,:,:) = ieegData(:,(iTime-1)*winSize+1:iTime*winSize);
    end
end

