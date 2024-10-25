function [accResultsPhoneme] = plotPhonemeSequence(decodeTimeStruct1D,decodeTimeStruct1Dshuffle)
%PLOTPHONEMESEQUENCE 
colvals = lines(3);
for iPhon = 1:3
    
    if(iPhon==1)
        [ax,accResultsTemp] = visTimeGenAcc1DCluster(decodeTimeStruct1D(iPhon,:),decodeTimeStruct1Dshuffle(iPhon,:),pVal2Cutoff=0.05,...
            axisLabel = 'Response',clowLimit = 0,timePad = 0.2,...
             maxVal = -(iPhon)*0.05, boxPlotPlace=(iPhon)*0.1+0.3, chanceVal = 0.1111,clabel = 'Accuracy',...
             colval=colvals(iPhon,:),showShuffle=1,showPeaks=5,searchRange=[-1 1.5],showAccperChance=1);

    else
       [axtemp,accResultsTemp] =  visTimeGenAcc1DCluster(decodeTimeStruct1D(iPhon,:),decodeTimeStruct1Dshuffle(iPhon,:),pVal2Cutoff=0.05,...
            axisLabel = 'Response',clowLimit = 0,timePad =0.2,...
             maxVal = -(iPhon)*0.05, boxPlotPlace=(iPhon)*0.1+0.3, chanceVal = 0.1111,clabel = 'Accuracy',...
             colval=colvals(iPhon,:),tileaxis = ax,showPeaks=5,searchRange=[-1 1.5],showAccperChance=1);
    end
    accResultsPhoneme{iPhon} = accResultsTemp;

end
end

