function [noisyBroadChan,noisyGammaChan] = extractBadChannels(ieegChan,fs)
for iCh =1:size(ieegChan,1)
    
    tmp = squeeze(ieegChan(iCh,:,:));
    tmpGamma = eegfilt(double(tmp),fs,70,150,0,200);
    tmp = detrend(tmp);
    stdChan = std(tmp');
    stdAll(iCh) = std(tmp(:));
    stdGammaAll(iCh) = std(tmpGamma(:)); % square and take the log
end
noisyBroadChan = isoutlier(stdAll,'quartiles');
noisyGammaChan = isoutlier(stdGammaAll,'quartiles');

end