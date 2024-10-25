function [accResultsPhoneme] = plotPhonemeSyllableSequence(decodeTimeStruct1D,decodeTimeStruct1Dshuffle,syllableType)
if(syllableType==1)
    chanceRange = [0.2 0.25 0.2];
else
    chanceRange = [0.25 0.2 0.25];
end

colvals = lines(3);
for iPhon = 1:3
    
    if(iPhon==1)
        [ax,accResultsTemp] = visTimeGenAcc1DCluster(decodeTimeStruct1D(iPhon,:),decodeTimeStruct1Dshuffle(iPhon,:),pVal2Cutoff=0.01,...
            axisLabel = 'Response',clowLimit = 0,timePad = 0.2,...
             maxVal = -(iPhon)*0.1, boxPlotPlace=(iPhon)*0.1+0.4, chanceVal = chanceRange(iPhon),clabel = 'Accuracy',...
             colval=colvals(iPhon,:),showShuffle=1,showPeaks=2,searchRange=[-0.15 1.5],showAccperChance=1);

    else
       [axtemp,accResultsTemp] =  visTimeGenAcc1DCluster(decodeTimeStruct1D(iPhon,:),decodeTimeStruct1Dshuffle(iPhon,:),pVal2Cutoff=0.01,...
            axisLabel = 'Response',clowLimit = 0,timePad =0.2,...
             maxVal = -(iPhon)*0.1, boxPlotPlace=(iPhon)*0.1+0.4, chanceVal = chanceRange(iPhon),clabel = 'Accuracy',...
             colval=colvals(iPhon,:),tileaxis = ax,showPeaks=2,searchRange=[-0.15 1.5],showAccperChance=1);
    end
    accResultsPhoneme{iPhon} = accResultsTemp;

end
end

