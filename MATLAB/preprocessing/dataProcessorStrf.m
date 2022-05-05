function [XMatrix,YMatrix] = dataProcessorStrf(SMel,ieeg,fsIeeg,lag)
XMatrix = []; YMatrix = [];
for iTrial = 1:size(ieeg,2)
    iTrial
    SMelTemp = squeeze(SMel(iTrial,:,:));
    ieegTemp = squeeze(ieeg(:,iTrial,:))';
    size(SMelTemp)
    size(ieegTemp)
    for iTime = 1:(size(ieegTemp,2)-(lag*fsIeeg-1))
       
        if(isfinite(SMelTemp(:,iTime+lag*fsIeeg-1)))
            YMatrix = [YMatrix ieegTemp(:,iTime)];
            XMatrix = cat(3,XMatrix,SMelTemp(:,iTime:iTime+lag*fsIeeg-1));
        else
            continue;
        end
    end
end
end