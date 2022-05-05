function [XMatrix,YMatrix,XMatCell,YMatCell, YMatChanCell] = dataProcessorStrfNoLag(WMel,SMellog,ieeg)
XMatrix = []; YMatrix = []; XMatCell = []; YMatCell = []; YMatChanCell = [];
for iTrial = 1:size(ieeg,2)
    iTrial
    WMelTemp = squeeze(WMel(iTrial,:,:));
    SMelLogTemp = squeeze(SMellog(iTrial,:,:));
    ieegTemp = squeeze(ieeg(:,iTrial,:));
   
    xmattemp = [];
    ymattemp = [];
    for iTime = 1:(size(ieegTemp,2))
       
        if(isfinite(SMelLogTemp(:,iTime)))
            YMatrix = cat(2,YMatrix, ieegTemp(:,iTime));
            XMatrix = cat(2,XMatrix,WMelTemp(:,iTime));
            ymattemp = cat(2,ymattemp, ieegTemp(:,iTime));
            xmattemp = cat(2,xmattemp,WMelTemp(:,iTime));
        else
            continue;
        end
    end
    infVal = (isinf(xmattemp));
    meanX = mean(xmattemp(infVal==0));
    xmattemp(infVal)=meanX;
    XMatCell{iTrial} = xmattemp';
    YMatCell{iTrial} = ymattemp';
    for iChan = 1:size(ymattemp,1)
        YMatChanCell{iTrial,iChan} = ymattemp(iChan,:)';
    end
end
    infVal = (isinf(XMatrix));
    meanX = mean(XMatrix(infVal==0));
    XMatrix(infVal)=meanX;
XMatrix = XMatrix';
YMatrix = YMatrix';
end