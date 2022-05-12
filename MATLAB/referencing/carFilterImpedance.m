function [ieegFilt,meanIeeg] = carFilterImpedance(ieeg,noChannel)
% ieeg - channels x trials x time
goodChan = setdiff(1:size(ieeg,1),noChannel);
ieegFilt = ieeg - mean(ieeg(goodChan,:,:),1);
meanIeeg = squeeze(mean(ieeg(goodChan,:,:),1));
end