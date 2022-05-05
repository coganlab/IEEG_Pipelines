function psdPower =  getPsd(ieegSplit,fs,tw,etw1,etw2)
time = linspace(tw(1),tw(2),size(ieegSplit,3));
eTimeWindow1 = time>=etw1(1)&time<=etw1(2);
eTimeWindow2 = time>=etw2(1)&time<=etw2(2);
for iChan = 1:size(ieegSplit,1)
    for iTrial = 1:size(ieegSplit,2)
        ieegTemp = squeeze(ieegSplit(iChan,iTrial,:));
        L=length(ieegTemp(eTimeWindow1));
                nsc=floor(L/3);
                nov=floor(nsc/2);
                nff=max(256,2^nextpow2(nsc));
        psdPower1(iChan,iTrial,:) = (pwelch(ieegTemp(eTimeWindow1),nsc,nov,nff,fs));
        psdPower2(iChan,iTrial,:) = (pwelch(ieegTemp(eTimeWindow2),nsc,nov,nff,fs));
    end
end
psdPower = cat(2,psdPower1,psdPower2);
end