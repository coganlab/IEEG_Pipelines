function meanFreqChanOut = extractSpecNorm(spec,tw,etw)

tspec = linspace(tw(1),tw(2),size(spec{1},2));
meanFreqChanOut = [];
for iChan = 1 : length(spec)
         spec2Analyze = spec{iChan};
         meanFreq = [];
    for iFreq = 1:size(spec2Analyze,3)
        meanFreq(iFreq) = mean2(squeeze((spec2Analyze(:,tspec>=etw(1)&tspec<= etw(2),iFreq))));
    end
    meanFreqChanOut(iChan,:) = meanFreq;
           
end

end