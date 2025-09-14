function ieegStructNoNorm = removeZScore(ieegStruct,normFactor)
%REMOVEZSCORE Summary of this function goes here
%   Detailed explanation goes here
ieegStructNoNorm = ieegStruct;
for iChan = 1:size(normFactor,1)
    ieegStructNoNorm.data(iChan,:,:) = ieegStruct.data(iChan,:,:).*normFactor(iChan,2) + normFactor(iChan,1);
end
ieegStructNoNorm.name = [ieegStructNoNorm.name "-normalization_removed"];
end

